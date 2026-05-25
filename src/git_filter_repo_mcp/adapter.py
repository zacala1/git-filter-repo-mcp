"""git-filter-repo adapter - wraps git-filter-repo commands."""

import base64
import contextlib
import datetime
import json
import logging
import platform
import random
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

# Timeout constants (seconds)
TIMEOUT_FAST = 5       # Quick operations (single file read)
TIMEOUT_DEFAULT = 30   # Standard git operations
TIMEOUT_LONG = 300     # filter-repo operations

# Limit constants
MAX_FILES_LIMIT = 1000
MAX_PREVIEW_COMMITS = 20
MAX_FINDINGS_LIMIT = 50
MAX_FILES_TO_SCAN = 200

# Date randomization: probability of advancing to the next day between commits
DATE_ADVANCE_PROBABILITY = 0.3


def _parse_lines(output: str) -> list[str]:
    """Parse stdout into non-empty lines."""
    return [line for line in output.strip().split("\n") if line]


def _safe_int(value: str, default: int = 0) -> int:
    """Safely parse int from string."""
    try:
        return int(value.strip())
    except (ValueError, AttributeError):
        return default


@contextlib.contextmanager
def _temp_file(content: str, suffix: str) -> Iterator[str]:
    """Write content to a named temp file and yield its path, deleting on exit.

    Uses ``delete=False`` so the file is closed before being handed to git
    subprocesses (required on Windows where the open handle would block
    other readers).
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as f:
        f.write(content)
        path = f.name
    try:
        yield path
    finally:
        Path(path).unlink(missing_ok=True)


# --- git-filter-repo --commit-callback bodies ---
#
# These are Python source snippets exec'd by git-filter-repo with ``commit``
# (and any decoded payload variables) in scope. Kept as module-level
# constants so the embedded Python is easy to read and lint-friendly.

_REWRITE_MESSAGES_BODY = """\
_orig_id = commit.original_id.decode() if commit.original_id else None
_msg_str = commit.message.decode("utf-8") if isinstance(commit.message, bytes) else commit.message
_new_msg = None
if _orig_id and _orig_id in _HASH_MAP:
    _new_msg = _HASH_MAP[_orig_id]
elif _msg_str.strip() in _MSG_MAP:
    _new_msg = _MSG_MAP[_msg_str.strip()]
if _new_msg is not None:
    commit.message = _new_msg.encode("utf-8")"""

_REWRITE_SINGLE_BODY = """\
_TARGET = _PAYLOAD["target"]
_CHANGES = _PAYLOAD["changes"]
_orig_id = commit.original_id.decode() if commit.original_id else None
if _orig_id and (_orig_id.startswith(_TARGET) or _TARGET.startswith(_orig_id)):
    if "message" in _CHANGES:
        commit.message = _CHANGES["message"].encode()
    if "author_name" in _CHANGES:
        commit.author_name = _CHANGES["author_name"].encode()
        commit.committer_name = _CHANGES["author_name"].encode()
    if "author_email" in _CHANGES:
        commit.author_email = _CHANGES["author_email"].encode()
        commit.committer_email = _CHANGES["author_email"].encode()"""

_DATE_REWRITE_BODY = """\
_commit_hash = commit.original_id.decode() if commit.original_id else None
if _commit_hash and _commit_hash in _DATE_MAP:
    _new_ts, _tz_offset = _DATE_MAP[_commit_hash]
    _new_date = f"{_new_ts} {_tz_offset}".encode()
    commit.author_date = _new_date
    commit.committer_date = _new_date"""


@dataclass
class FilterResult:
    """Result of a git-filter-repo operation."""

    success: bool
    message: str
    commits_processed: int = 0
    commits_rewritten: int = 0
    files_affected: list[str] = field(default_factory=list)
    dry_run: bool = False
    error: str | None = None


@dataclass
class CommitInfo:
    """Information about a single commit."""

    hash: str
    author_name: str
    author_email: str
    committer_name: str
    committer_email: str
    message: str
    date: str
    files: list[str] = field(default_factory=list)


class GitFilterRepoAdapter:
    """Adapter for git-filter-repo commands."""

    def __init__(self, repo_path: str):
        normalized = Path(self._normalize_path(repo_path))
        if not normalized.is_absolute():
            raise ValueError(f"Repository path must be absolute: {repo_path}")
        self.repo_path = normalized.resolve()
        self._validate_repo()
        self._check_git_filter_repo()

    @staticmethod
    def _normalize_path(path: str) -> str:
        """Normalize path for cross-platform compatibility."""
        if platform.system() != "Windows":
            return path

        # Git Bash style: /c/Users/... -> C:\Users\...
        if match := re.match(r"^/([a-zA-Z])/(.*)$", path):
            sep = "\\"
            return f"{match.group(1).upper()}:{sep}{match.group(2).replace('/', sep)}"

        # WSL paths - keep as-is
        if path.startswith(("//wsl", "\\\\wsl")):
            return path

        # Unix absolute paths (not Git Bash) - keep as-is
        if path.startswith("/") and not re.match(r"^/[a-zA-Z]/", path):
            return path

        # Windows paths with forward slashes
        if re.match(r"^[a-zA-Z]:/", path):
            return path.replace("/", "\\")

        # Relative paths with forward slashes
        if "/" in path and not path.startswith(("/", "\\\\")):
            return path.replace("/", "\\")

        return path

    def _validate_repo(self) -> None:
        """Validate that the path is a git repository."""
        if not self.repo_path.exists():
            raise ValueError(f"Repository path does not exist: {self.repo_path}")
        git_dir = self.repo_path / ".git"
        if not git_dir.exists() or not git_dir.is_dir():
            raise ValueError(f"Not a git repository: {self.repo_path}")

    def _check_git_filter_repo(self) -> None:
        """Check if git-filter-repo is installed."""
        if not shutil.which("git-filter-repo"):
            raise RuntimeError(
                "git-filter-repo is not installed. Install with: pip install git-filter-repo"
            )

    def _run_command(
        self, args: list[str], check: bool = True, timeout: int = TIMEOUT_DEFAULT
    ) -> subprocess.CompletedProcess:
        """Run a command in the repo directory."""
        try:
            return subprocess.run(
                args, cwd=self.repo_path, capture_output=True, stdin=subprocess.DEVNULL,
                check=check, timeout=timeout, encoding="utf-8", errors="replace",
            )
        except subprocess.CalledProcessError as e:
            logger.warning("%s failed (rc=%d): %s", args[0], e.returncode, (e.stderr or "")[:200])
            raise
        except subprocess.TimeoutExpired:
            logger.error("timeout %ds: %s", timeout, args[0])
            raise
        except FileNotFoundError as e:
            # The executable (typically ``git`` or ``git-filter-repo``) was not
            # found on PATH. Re-raise as ``RuntimeError`` so the server's
            # handler decorator converts it into a clean COMMAND_FAILED error
            # envelope instead of an opaque INTERNAL_ERROR.
            raise RuntimeError(
                f"{args[0]} not found on PATH. Is it installed?"
            ) from e

    def _run_git(self, *args: str, timeout: int = TIMEOUT_DEFAULT) -> subprocess.CompletedProcess:
        """Run a git command."""
        return self._run_command(["git", *args], timeout=timeout)

    def _run_git_fast(self, *args: str) -> subprocess.CompletedProcess:
        """Run a quick git command with short timeout."""
        return self._run_command(["git", *args], timeout=TIMEOUT_FAST)

    def _run_filter_repo(self, *args: str, dry_run: bool = False, force: bool = False) -> subprocess.CompletedProcess:
        """Run git-filter-repo."""
        cmd = ["git-filter-repo"]
        if dry_run:
            cmd.append("--dry-run")
        if force:
            cmd.append("--force")
        cmd.extend(args)
        return self._run_command(cmd, check=False, timeout=TIMEOUT_LONG)

    @staticmethod
    def _build_callback(payload: dict[str, dict | list], body: str) -> str:
        """Compose a git-filter-repo --commit-callback Python source string.

        ``payload`` maps variable name to JSON-serialisable data; each entry is
        base64-encoded and decoded at the top of the generated source so the
        body can reference them by name. ``body`` is appended verbatim and
        runs with ``commit`` plus the decoded variables in scope.
        """
        lines = ["import base64, json"]
        for var, data in payload.items():
            encoded = base64.b64encode(json.dumps(data).encode()).decode()
            lines.append(f'{var} = json.loads(base64.b64decode("{encoded}").decode())')
        lines.append(body)
        return "\n".join(lines)

    # Record separator that won't appear in commit messages/names
    _FIELD_SEP = "\x1e"

    # Record separator for multi-record git log output (rendered by git via %x00)
    _RECORD_SEP = "\x00"

    def get_commits(self, branch: str = "HEAD", max_count: int | None = None) -> list[CommitInfo]:
        """Get commit information from the repository.

        Args:
            branch: Git ref to read (branch name, tag, or HEAD).
            max_count: Limit number of commits returned. None means all.

        Returns:
            List of CommitInfo ordered newest-first. Empty list if repo has
            no commits or branch is invalid.
        """
        self._validate_ref(branch)
        sep = self._FIELD_SEP
        # %B (full body) goes last so split(sep, 6) captures multi-line messages.
        # %x00 is used as record separator because Windows rejects literal null bytes.
        args = ["log", f"--format=%x00%H{sep}%an{sep}%ae{sep}%cn{sep}%ce{sep}%aI{sep}%B", branch]
        if max_count is not None:
            args.append(f"-n{max_count}")

        try:
            result = self._run_git(*args)
        except subprocess.CalledProcessError:
            # Empty repo or invalid branch — no commits
            return []
        commits = []
        for record in result.stdout.split(self._RECORD_SEP):
            record = record.strip()
            if not record:
                continue
            # Format fields: hash, author_name, author_email,
            # committer_name, committer_email, date(ISO), message(full body)
            parts = record.split(sep, 6)
            if len(parts) >= 7:
                h, an, ae, cn, ce, date, message = parts
                commits.append(CommitInfo(h, an, ae, cn, ce, message.strip(), date))
        return commits

    def get_commit_diff(self, commit_hash: str) -> str:
        """Get diff stat for a commit."""
        self._validate_ref(commit_hash)
        return self._run_git("show", "--stat", commit_hash).stdout or ""

    def get_commit_files(self, commit_hash: str) -> list[str]:
        """Get files changed in a commit."""
        self._validate_ref(commit_hash)
        return _parse_lines(self._run_git_fast("show", "--name-only", "--format=", commit_hash).stdout)

    def analyze_history(self, branch: str = "HEAD", max_count: int = 100) -> dict:
        """Analyze repository history and return summary statistics.

        Returns dict with keys:
            total_commits: number of commits analyzed (capped at ``max_count``)
            total_in_branch: total commits in the branch (independent of max_count);
                None if the count could not be obtained (empty repo / bad ref)
            total_authors, authors (name->count),
            commits: recent commit previews (hash, author, message, date)
        """
        commits = self.get_commits(branch, max_count)

        authors = {}
        for commit in commits:
            author_key = f"{commit.author_name} <{commit.author_email}>"
            authors[author_key] = authors.get(author_key, 0) + 1

        # True branch size (independent of max_count) — best-effort, cheap.
        try:
            total_in_branch: int | None = _safe_int(
                self._run_git_fast("rev-list", "--count", branch).stdout
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            total_in_branch = None

        return {
            "total_commits": len(commits),
            "total_in_branch": total_in_branch,
            "total_authors": len(authors),
            "authors": authors,
            "commits": [
                {
                    "hash": c.hash[:8],
                    "author": f"{c.author_name} <{c.author_email}>",
                    "message": c.message[:80] + ("..." if len(c.message) > 80 else ""),
                    "date": c.date,
                }
                for c in commits[:MAX_PREVIEW_COMMITS]
            ],
        }

    def rewrite_commit_messages(
        self,
        message_callback: Callable[[str, str], str],
        branch: str = "HEAD",
        dry_run: bool = True,
        force: bool = False,
    ) -> FilterResult:
        """
        Rewrite commit messages using a callback function.

        Args:
            message_callback: Function(original_message, commit_hash) -> new_message
            branch: Branch to rewrite
            dry_run: If True, don't actually modify the repository
            force: If True, allow running on a repo with existing filter-repo state
        """
        commits = self.get_commits(branch)
        rewrites = []

        for commit in commits:
            new_message = message_callback(commit.message, commit.hash)
            if new_message != commit.message:
                rewrites.append((commit.hash, commit.message, new_message))

        if dry_run:
            return FilterResult(
                success=True,
                message=f"Dry run: {len(rewrites)} commits would be rewritten",
                commits_processed=len(commits),
                commits_rewritten=len(rewrites),
                dry_run=True,
            )

        # Build both message-based and hash-based lookup tables.
        # Hash-based takes priority to handle duplicate commit messages correctly.
        msg_replacements = {old: new for _, old, new in rewrites}
        hash_replacements = {h: new for h, _, new in rewrites}

        callback_code = self._build_callback(
            {"_HASH_MAP": hash_replacements, "_MSG_MAP": msg_replacements},
            _REWRITE_MESSAGES_BODY,
        )

        result = self._run_filter_repo(
            "--commit-callback",
            callback_code,
            dry_run=False,
            force=force,
        )

        if result.returncode != 0:
            return FilterResult(
                success=False,
                message="Failed to rewrite commit messages",
                error=result.stderr,
            )

        return FilterResult(
            success=True,
            message=f"Successfully rewrote {len(rewrites)} commit messages",
            commits_processed=len(commits),
            commits_rewritten=len(rewrites),
        )

    def change_author(
        self,
        old_email: str,
        new_name: str,
        new_email: str,
        dry_run: bool = True,
        force: bool = False,
    ) -> FilterResult:
        """Change author/committer information for commits matching old_email.

        Uses git-filter-repo's --mailmap. If no commits match old_email,
        returns success with a message listing existing author emails.
        """
        for label, value in [("name", new_name), ("email", new_email), ("old_email", old_email)]:
            if not value or not value.strip():
                return FilterResult(
                    success=False,
                    message=f"{label} must not be empty",
                )
            if any(c in value for c in "<>\n\r\t"):
                return FilterResult(
                    success=False,
                    message=f"Invalid characters in {label}: angle brackets, tabs, and newlines are not allowed",
                )

        commits = self.get_commits()
        affected = [c for c in commits if c.author_email == old_email]

        if not affected:
            unique_emails = sorted({c.author_email for c in commits})
            return FilterResult(
                success=True,
                message=f"No commits found with email '{old_email}'. Existing authors: {', '.join(unique_emails)}",
                commits_processed=len(commits),
                commits_rewritten=0,
                dry_run=dry_run,
            )

        if dry_run:
            return FilterResult(
                success=True,
                message=f"Dry run: {len(affected)} commits would be updated",
                commits_processed=len(commits),
                commits_rewritten=len(affected),
                dry_run=True,
            )

        mailmap = f"{new_name} <{new_email}> <{old_email}>\n"
        with _temp_file(mailmap, ".mailmap") as mailmap_path:
            result = self._run_filter_repo("--mailmap", mailmap_path, dry_run=False, force=force)

            if result.returncode != 0:
                return FilterResult(
                    success=False,
                    message="Failed to change author",
                    error=result.stderr,
                )

            return FilterResult(
                success=True,
                message=f"Successfully updated {len(affected)} commits",
                commits_processed=len(commits),
                commits_rewritten=len(affected),
            )

    def remove_files(
        self,
        paths: list[str],
        dry_run: bool = True,
        force: bool = False,
    ) -> FilterResult:
        """Remove files from entire git history."""
        if not paths:
            return FilterResult(success=False, message="No file paths provided")
        self._validate_paths(paths)
        if dry_run:
            try:
                result = self._run_git("log", "--all", "--format=%H", "--", *paths)
                affected = set(_parse_lines(result.stdout))
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                affected = set()
                for path in paths:
                    try:
                        result = self._run_git("log", "--all", "--format=%H", "--", path)
                        affected.update(_parse_lines(result.stdout))
                    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                        continue

            return FilterResult(
                success=True, message=f"Dry run: {len(affected)} commits affected",
                commits_rewritten=len(affected), files_affected=paths, dry_run=True,
            )

        args = ["--invert-paths"] + [arg for path in paths for arg in ("--path", path)]
        result = self._run_filter_repo(*args, dry_run=False, force=force)

        if result.returncode != 0:
            return FilterResult(success=False, message="Failed to remove files", error=result.stderr)

        return FilterResult(success=True, message=f"Removed {len(paths)} paths", files_affected=paths)

    def remove_large_files(
        self, size_threshold_mb: float = 10.0, dry_run: bool = True, force: bool = False,
    ) -> FilterResult:
        """Find and remove files larger than size_threshold_mb from history.

        Uses git cat-file --batch-check for efficient bulk size queries,
        with per-object fallback if batch mode fails.
        """
        try:
            result = self._run_git("rev-list", "--objects", "--all")
        except subprocess.CalledProcessError:
            return FilterResult(success=True, message="No objects found (empty repository?)", dry_run=dry_run)
        size_bytes = int(size_threshold_mb * 1024 * 1024)

        objects_with_paths = {}
        object_hashes = []
        for line in _parse_lines(result.stdout):
            parts = line.split(" ", 1)
            if len(parts) == 2:
                objects_with_paths[parts[0]] = parts[1]
                object_hashes.append(parts[0])

        if not object_hashes:
            return FilterResult(success=True, message="No files found", files_affected=[], dry_run=dry_run)

        large_files = []
        try:
            batch_result = subprocess.run(
                ["git", "cat-file", "--batch-check=%(objectsize)"],
                cwd=self.repo_path, input="\n".join(object_hashes),
                capture_output=True, check=True, encoding="utf-8", errors="replace",
                timeout=TIMEOUT_DEFAULT,
            )
            size_lines = batch_result.stdout.strip().split("\n")
            for blob_hash, size_str in zip(object_hashes, size_lines):
                size = _safe_int(size_str)
                if size > size_bytes:
                    large_files.append((objects_with_paths.get(blob_hash, blob_hash), size / (1024 * 1024)))
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.warning("batch failed, falling back to per-object cat-file: %s", e)
            for blob_hash in object_hashes:
                try:
                    size = _safe_int(self._run_git_fast("cat-file", "-s", blob_hash).stdout)
                    if size > size_bytes:
                        large_files.append((objects_with_paths.get(blob_hash, blob_hash), size / (1024 * 1024)))
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                    continue

        if dry_run:
            return FilterResult(
                success=True,
                message=f"Dry run: {len(large_files)} large files found",
                files_affected=[f"{p} ({s:.2f}MB)" for p, s in large_files],
                dry_run=True,
            )

        if not large_files:
            return FilterResult(
                success=True,
                message="No large files found",
            )

        paths = [p for p, _ in large_files]
        return self.remove_files(paths, dry_run=False, force=force)

    def filter_paths(
        self,
        include_paths: list[str] | None = None,
        exclude_paths: list[str] | None = None,
        dry_run: bool = True,
        force: bool = False,
    ) -> FilterResult:
        """Filter repository to include/exclude specific paths.

        Note: include_paths and exclude_paths cannot be used together because
        git-filter-repo's --invert-paths is a global flag that inverts ALL
        path selections. Use one or the other per invocation.
        """
        self._validate_paths((include_paths or []) + (exclude_paths or []))
        if not include_paths and not exclude_paths:
            return FilterResult(
                success=False,
                message="No paths specified",
                error="Provide include_paths or exclude_paths.",
            )

        if include_paths and exclude_paths:
            return FilterResult(
                success=False,
                message="Cannot use include_paths and exclude_paths together",
                error="git-filter-repo's --invert-paths is global and would invert all path "
                "selections. Use include_paths OR exclude_paths per invocation.",
            )

        args = []

        if include_paths:
            for path in include_paths:
                args.extend(["--path", path])

        if exclude_paths:
            args.append("--invert-paths")
            for path in exclude_paths:
                args.extend(["--path", path])

        if dry_run:
            return FilterResult(
                success=True,
                message="Dry run: path filtering would be applied",
                files_affected=(include_paths or []) + (exclude_paths or []),
                dry_run=True,
            )

        result = self._run_filter_repo(*args, dry_run=False, force=force)

        if result.returncode != 0:
            return FilterResult(
                success=False,
                message="Failed to filter paths",
                error=result.stderr,
            )

        return FilterResult(
            success=True,
            message="Successfully filtered paths",
            files_affected=(include_paths or []) + (exclude_paths or []),
        )

    def create_backup(self) -> str:
        """Create a backup branch before rewriting."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup_branch = f"backup_{timestamp}"

        self._run_git("branch", backup_branch)
        return backup_branch

    def restore_backup(self, backup_branch: str) -> FilterResult:
        """Restore from a backup branch."""
        if not backup_branch.startswith("backup_") or not re.match(r'^backup_[0-9_]+$', backup_branch):
            return FilterResult(
                success=False,
                message=f"Invalid backup branch: {backup_branch!r}",
                error="Backup branch must match format 'backup_YYYYMMDD_HHMMSS_ffffff'",
            )
        try:
            # Verify the backup branch exists before destructive reset
            self._run_git("rev-parse", "--verify", f"refs/heads/{backup_branch}")

            result = self._run_git("rev-parse", "--abbrev-ref", "HEAD")
            current_branch = result.stdout.strip()

            self._run_git("reset", "--hard", backup_branch)

            return FilterResult(
                success=True,
                message=f"Restored {current_branch} from {backup_branch}",
            )
        except subprocess.CalledProcessError as e:
            return FilterResult(
                success=False,
                message="Failed to restore backup",
                error=str(e),
            )

    def collect_commit_files(
        self, commits: list[CommitInfo], branch: str, max_commits: int,
    ) -> dict[str, list[str]]:
        """Collect file lists for commits in a single git log call.

        Returns {commit_hash: [file_paths]}. Falls back to per-commit
        get_commit_files (first 20) if the bulk call fails.
        """
        try:
            result = self._run_git(
                "log", "--name-only", "--format=%H", f"-n{max_commits}", branch,
            )
            commit_files_map: dict[str, list[str]] = {}
            current_hash = None
            for line in _parse_lines(result.stdout):
                if len(line) in (40, 64) and all(c in "0123456789abcdef" for c in line):
                    current_hash = line
                    commit_files_map[current_hash] = []
                elif current_hash:
                    commit_files_map[current_hash].append(line)
            return commit_files_map
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return {c.hash: self.get_commit_files(c.hash) for c in commits[:20]}

    def _scan_file_contents(
        self, files_to_scan: list[tuple[str, str]],
    ) -> list[dict]:
        """Scan file contents for secrets.

        Args:
            files_to_scan: List of (commit_hash, file_path) tuples.

        Returns:
            List of finding dicts with keys: type, description, severity,
            file, commit, line, matched. Capped at MAX_FINDINGS_LIMIT.
        """
        from .secrets import scan_content

        findings: list[dict] = []
        for commit_hash, file_path in files_to_scan:
            if len(findings) >= MAX_FINDINGS_LIMIT:
                break
            try:
                content = self._run_git_fast("show", f"{commit_hash}:{file_path}").stdout
                if not content:
                    continue
                for f in scan_content(content, file_path, commit_hash):
                    findings.append({
                        "type": f.pattern_name, "description": f.description, "severity": f.severity,
                        "file": f.file_path, "commit": f.commit_hash[:8], "line": f.line_number, "matched": f.matched_text,
                    })
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                continue
        return findings

    def scan_secrets(
        self,
        branch: str = "HEAD",
        max_commits: int = 100,
    ) -> dict:
        """Scan repository history for secrets and sensitive files.

        Returns dict with keys: message (summary), commits_scanned, secrets_found,
        sensitive_files (count), findings (list), sensitive_file_list, files_scanned.
        """
        from .secrets import get_file_risk_level, is_sensitive_file

        commits = self.get_commits(branch, max_commits)
        commit_files_map = self.collect_commit_files(commits, branch, max_commits)

        sensitive_files = []
        seen_sensitive: set[str] = set()
        files_to_scan: list[tuple[str, str]] = []
        for commit in commits:
            for file_path in commit_files_map.get(commit.hash, []):
                if is_sensitive_file(file_path) and file_path not in seen_sensitive:
                    seen_sensitive.add(file_path)
                    sensitive_files.append({
                        "file": file_path, "commit": commit.hash[:8], "risk": get_file_risk_level(file_path),
                    })
                if len(files_to_scan) < MAX_FILES_TO_SCAN:
                    files_to_scan.append((commit.hash, file_path))

        findings = self._scan_file_contents(files_to_scan)

        total_findings = len(findings)
        total_sensitive = len(sensitive_files)
        summary_parts = []
        if total_findings:
            summary_parts.append(f"{total_findings} secret(s) found")
        if total_sensitive:
            summary_parts.append(f"{total_sensitive} sensitive file(s)")
        message = "; ".join(summary_parts) if summary_parts else "No secrets or sensitive files found"

        return {
            "message": message,
            "commits_scanned": len(commits), "secrets_found": total_findings,
            "sensitive_files": total_sensitive, "findings": findings[:MAX_FINDINGS_LIMIT],
            "sensitive_file_list": sensitive_files[:MAX_PREVIEW_COMMITS], "files_scanned": len(files_to_scan),
        }

    @staticmethod
    def _validate_paths(paths: list[str]) -> None:
        """Validate paths don't contain git-filter-repo option injection."""
        for path in paths:
            if not path:
                raise ValueError("Invalid path: empty string")
            if path.startswith("-"):
                raise ValueError(f"Invalid path (must not start with '-'): {path!r}")

    @staticmethod
    def _validate_commit_hash(commit_hash: str) -> None:
        """Validate commit hash is safe hex string (prevents code injection in callbacks)."""
        if not re.match(r'^[0-9a-fA-F]+$', commit_hash):
            raise ValueError(f"Invalid commit hash (must be hex): {commit_hash!r}")

    def rewrite_single_commit(
        self,
        commit_hash: str,
        new_message: str | None = None,
        new_author_name: str | None = None,
        new_author_email: str | None = None,
        force: bool = True,
    ) -> FilterResult:
        """Rewrite a single commit's message and/or author in one filter-repo pass.

        Note: When author fields are changed, committer fields are also updated
        to match, since in most workflows they should be identical.
        """
        self._validate_commit_hash(commit_hash)
        changes = {}
        if new_message is not None:
            changes["message"] = new_message
        if new_author_name is not None:
            changes["author_name"] = new_author_name
        if new_author_email is not None:
            changes["author_email"] = new_author_email

        if not changes:
            return FilterResult(success=False, message="No changes specified")

        callback_code = self._build_callback(
            {"_PAYLOAD": {"target": commit_hash, "changes": changes}},
            _REWRITE_SINGLE_BODY,
        )

        result = self._run_filter_repo(
            "--commit-callback",
            callback_code,
            dry_run=False,
            force=force,
        )
        if result.returncode != 0:
            return FilterResult(
                success=False,
                message="Failed to rewrite commit",
                error=result.stderr,
            )

        changes_made = []
        if new_message is not None:
            changes_made.append("message")
        if new_author_name is not None or new_author_email is not None:
            changes_made.append("author")

        return FilterResult(
            success=True,
            message=f"Updated commit {commit_hash[:8]}: {', '.join(changes_made)}",
            commits_rewritten=1,
        )

    def list_all_files_in_history(self, limit: int = MAX_FILES_LIMIT) -> list[str]:
        """List all files that have ever existed."""
        files = set()
        for line in _parse_lines(self._run_git("log", "--all", "--name-only", "--format=").stdout):
            files.add(line)
            if len(files) >= limit:
                break
        return sorted(files)[:limit]

    def get_file_history(self, file_path: str) -> list[dict]:
        """Get commit history for a specific file."""
        self._validate_paths([file_path])
        sep = self._FIELD_SEP
        history = []
        for line in _parse_lines(self._run_git("log", "--follow", f"--format=%H{sep}%an{sep}%ae{sep}%s{sep}%aI", "--", file_path).stdout):
            parts = line.split(sep, 4)
            if len(parts) >= 5:
                history.append({"hash": parts[0][:8], "author": f"{parts[1]} <{parts[2]}>", "message": parts[3], "date": parts[4]})
        return history

    @staticmethod
    def _validate_ref(ref: str) -> None:
        """Validate a git ref is safe (not an option injection)."""
        if ref.startswith("-"):
            raise ValueError(f"Invalid ref (must not start with '-'): {ref!r}")

    def squash_commits(self, start_commit: str, end_commit: str = "HEAD", new_message: str | None = None, dry_run: bool = True) -> FilterResult:
        """Squash commits between start_commit (exclusive) and end_commit (inclusive).

        Uses ``git reset --soft`` + ``git commit``, so end_commit must resolve
        to the current HEAD.  If end_commit is not HEAD the operation is
        rejected to avoid silent data loss.
        """
        self._validate_ref(start_commit)
        self._validate_ref(end_commit)
        try:
            commit_count = _safe_int(self._run_git("rev-list", "--count", f"{start_commit}..{end_commit}").stdout)
        except subprocess.CalledProcessError:
            return FilterResult(success=False, message=f"Invalid commit range: {start_commit}..{end_commit}")
        if commit_count == 0:
            return FilterResult(success=False, message=f"No commits in range: {start_commit}..{end_commit}")

        if dry_run:
            return FilterResult(success=True, message=f"Dry run: would squash {commit_count} commits", commits_processed=commit_count, dry_run=True)

        # Ensure end_commit points to HEAD to avoid silent data loss
        try:
            head_hash = self._run_git("rev-parse", "HEAD").stdout.strip()
            end_hash = self._run_git("rev-parse", end_commit).stdout.strip()
        except subprocess.CalledProcessError:
            return FilterResult(success=False, message=f"Cannot resolve commits: {end_commit}")

        if head_hash != end_hash:
            return FilterResult(
                success=False,
                message=f"end_commit must be HEAD (got {end_commit}). "
                "Squashing a range that does not end at HEAD is not supported.",
            )

        try:
            if new_message is None:
                messages = _parse_lines(self._run_git("log", "--format=%s", f"{start_commit}..{end_commit}").stdout)
                new_message = "Squashed commits:\n" + "\n".join(f"- {m}" for m in messages) if messages else "Squashed commits"

            self._run_git("reset", "--soft", start_commit)
            try:
                self._run_git("commit", "-m", new_message)
            except subprocess.CalledProcessError:
                # Commit failed after reset — restore HEAD to avoid leaving dirty state
                self._run_git("reset", "--soft", head_hash)
                return FilterResult(success=False, message="Failed to create squash commit; HEAD restored", error="git commit failed after reset --soft")
            return FilterResult(success=True, message=f"Squashed {commit_count} commits", commits_processed=commit_count, commits_rewritten=1)
        except subprocess.CalledProcessError as e:
            return FilterResult(success=False, message="Failed to squash", error=str(e))

    def replace_text_in_history(
        self, old_text: str, new_text: str, file_pattern: str | None = None, dry_run: bool = True, force: bool = False,
    ) -> FilterResult:
        """Replace text throughout repository history."""
        if not old_text:
            return FilterResult(success=False, message="old_text must not be empty")
        if "\n" in old_text or "\n" in new_text:
            return FilterResult(
                success=False,
                message="Text cannot contain newlines",
                error="Newlines in old_text or new_text would corrupt the expressions file",
            )

        if dry_run:
            # Search git history (not working directory) for affected files — no temp file needed
            try:
                git_args = ["log", "--all", "-S", old_text, "--name-only", "--format="]
                if file_pattern:
                    git_args.extend(["--", file_pattern])
                result = self._run_git(*git_args)
                files_with_matches = sorted(set(_parse_lines(result.stdout)))
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                files_with_matches = []
            return FilterResult(success=True, message=f"Dry run: {len(files_with_matches)} files in history", files_affected=files_with_matches[:20], dry_run=True)

        safe_old_text = re.escape(old_text).replace("==>", "\\=\\=\\>")
        safe_new_text = new_text.replace("==>", "\\=\\=\\>")
        expression = f"regex:{safe_old_text}==>{safe_new_text}\n"
        with _temp_file(expression, ".txt") as expressions_path:
            args = ["--replace-text", expressions_path]
            if file_pattern:
                args.extend(["--path-glob", file_pattern])
            result = self._run_filter_repo(*args, dry_run=False, force=force)
            if result.returncode != 0:
                return FilterResult(success=False, message="Failed to replace text", error=result.stderr)
            return FilterResult(success=True, message="Replaced text in history")

    _TIME_RANGE_PRESETS = {
        "evening": (19, 0, 23, 0),
        "night": (22, 0, 2, 0),
        "weekend": (10, 0, 22, 0),
        "random": (0, 0, 23, 59),
    }

    def _parse_time_range(self, time_range: str) -> tuple[int, int, int, int] | FilterResult:
        """Parse a time range string into (start_hour, start_min, end_hour, end_min)."""
        if time_range in self._TIME_RANGE_PRESETS:
            return self._TIME_RANGE_PRESETS[time_range]

        if "-" in time_range:
            try:
                start_str, end_str = time_range.split("-")
                start_parts = start_str.strip().split(":")
                end_parts = end_str.strip().split(":")
                start_hour = int(start_parts[0])
                start_min = int(start_parts[1]) if len(start_parts) > 1 else 0
                end_hour = int(end_parts[0])
                end_min = int(end_parts[1]) if len(end_parts) > 1 else 0
                if not (0 <= start_hour <= 23 and 0 <= end_hour <= 23
                        and 0 <= start_min <= 59 and 0 <= end_min <= 59):
                    return FilterResult(
                        success=False,
                        message=f"Time values out of range: {time_range}",
                        error="Hours must be 0-23, minutes must be 0-59",
                    )
                return (start_hour, start_min, end_hour, end_min)
            except (ValueError, IndexError):
                return FilterResult(
                    success=False,
                    message=f"Invalid time range format: {time_range}",
                    error="Use preset (evening, night, weekend, random) or custom format 'HH:MM-HH:MM'",
                )

        return FilterResult(
            success=False,
            message=f"Unknown time range: {time_range}",
            error="Use preset (evening, night, weekend, random) or custom format 'HH:MM-HH:MM'",
        )

    @staticmethod
    def _pick_random_time(
        start_hour: int, start_min: int, end_hour: int, end_min: int,
    ) -> tuple[int, int, int]:
        """Pick a random (hour, minute, second) inside the given range.

        Handles wrap-around ranges (e.g. 22:00-02:00) by uniformly choosing
        between the pre-midnight and post-midnight halves.
        """
        second = random.randint(0, 59)
        if end_hour >= start_hour:
            hour = random.randint(start_hour, end_hour)
            if hour == start_hour == end_hour:
                minute = random.randint(start_min, end_min)
            elif hour == end_hour:
                minute = random.randint(0, end_min)
            elif hour == start_hour:
                minute = random.randint(start_min, 59)
            else:
                minute = random.randint(0, 59)
        else:
            # Wrap-around range
            if random.random() < 0.5:
                hour = random.randint(start_hour, 23)
                minute = random.randint(start_min, 59) if hour == start_hour else random.randint(0, 59)
            else:
                hour = random.randint(0, end_hour)
                minute = random.randint(0, end_min) if hour == end_hour else random.randint(0, 59)
        return hour, minute, second

    @staticmethod
    def _compose_datetime(
        current_date: datetime.datetime, hour: int, minute: int, second: int,
        start_hour: int, end_hour: int,
    ) -> datetime.datetime:
        """Place (hour, minute, second) onto current_date's day, advancing to
        the next day for wrap-around early-morning hours so the timestamp
        doesn't land before current_date.
        """
        day_offset = 1 if (end_hour < start_hour and hour <= end_hour) else 0
        return (current_date + datetime.timedelta(days=day_offset)).replace(
            hour=hour, minute=minute, second=second, microsecond=0,
        )

    @staticmethod
    def _parse_start_date(start_date: str | None, fallback: datetime.datetime) -> datetime.datetime | FilterResult:
        """Parse a YYYY-MM-DD start_date, returning fallback if not provided."""
        if not start_date:
            return fallback
        try:
            return datetime.datetime.strptime(start_date, "%Y-%m-%d").replace(
                tzinfo=datetime.timezone.utc,
            )
        except ValueError:
            return FilterResult(
                success=False,
                message=f"Invalid start_date format: {start_date}",
                error="Use YYYY-MM-DD format",
            )

    def _generate_date_mappings(
        self,
        commits: list[CommitInfo],
        start_hour: int,
        start_min: int,
        end_hour: int,
        end_min: int,
        weekend_only: bool,
        preserve_order: bool,
        start_date: str | None,
    ) -> dict[str, tuple[int, str]] | FilterResult:
        """Generate randomized date mappings for commits within time constraints.

        Supports wrap-around ranges (e.g. 22:00-02:00). When preserve_order is True,
        ensures each commit timestamp is strictly after the previous one.

        Returns:
            {commit_hash: (unix_timestamp, tz_offset_str)} on success,
            or FilterResult on validation error (e.g. bad start_date).
        """
        commit_dates: list[tuple[str, datetime.datetime]] = []
        for commit in commits:
            try:
                orig_dt = datetime.datetime.fromisoformat(commit.date.replace("Z", "+00:00"))
                commit_dates.append((commit.hash, orig_dt))
            except ValueError:
                continue
        commit_dates.sort(key=lambda x: x[1])

        default_base = commit_dates[0][1] if commit_dates else datetime.datetime.now(datetime.timezone.utc)
        base = self._parse_start_date(start_date, default_base)
        if isinstance(base, FilterResult):
            return base

        date_mappings: dict[str, tuple[int, str]] = {}
        current_date = base
        prev_timestamp: datetime.datetime | None = None

        for commit_hash, orig_dt in commit_dates:
            tz_offset = orig_dt.strftime("%z") or "+0000"

            new_dt = current_date  # default in case all 100 attempts fail
            found_valid = False
            hour = minute = second = 0  # last picked values (used by preserve_order)
            for _ in range(100):
                hour, minute, second = self._pick_random_time(
                    start_hour, start_min, end_hour, end_min,
                )
                new_dt = self._compose_datetime(
                    current_date, hour, minute, second, start_hour, end_hour,
                )

                if weekend_only and new_dt.weekday() < 5:
                    current_date += datetime.timedelta(days=5 - new_dt.weekday())
                    continue

                found_valid = True
                break

            if preserve_order and prev_timestamp and new_dt <= prev_timestamp:
                new_dt = prev_timestamp + datetime.timedelta(minutes=random.randint(5, 60))
                if weekend_only and new_dt.weekday() < 5:
                    new_dt += datetime.timedelta(days=5 - new_dt.weekday())
                new_dt = new_dt.replace(hour=hour, minute=minute, second=second)
                if new_dt <= prev_timestamp:
                    new_dt = prev_timestamp + datetime.timedelta(minutes=random.randint(1, 10))

            if not found_valid and prev_timestamp:
                new_dt = prev_timestamp + datetime.timedelta(minutes=random.randint(5, 30))
                # The 100-attempt loop exhausted without finding a valid weekend
                # slot — push forward to the next Saturday rather than silently
                # writing a weekday timestamp.
                if weekend_only and new_dt.weekday() < 5:
                    new_dt += datetime.timedelta(days=5 - new_dt.weekday())

            # Track ``new_dt`` so subsequent commits advance forward in time
            # (used to live inside the preserve_order branch, which meant
            # later commits piled up around ``base`` without it).
            current_date = new_dt

            date_mappings[commit_hash] = (int(new_dt.timestamp()), tz_offset)
            prev_timestamp = new_dt

            if random.random() < DATE_ADVANCE_PROBABILITY:
                current_date += datetime.timedelta(days=1)

        return date_mappings

    def _create_date_callback_code(self, date_mappings: dict[str, tuple[int, str]]) -> str:
        """Build git-filter-repo --commit-callback code for date rewriting.

        The generated code is exec'd by git-filter-repo with ``commit`` in
        scope. It looks up ``commit.original_id`` in a JSON-encoded date
        mapping dict.
        """
        serializable = {h: [ts, tz] for h, (ts, tz) in date_mappings.items()}
        return self._build_callback({"_DATE_MAP": serializable}, _DATE_REWRITE_BODY)

    def change_commit_dates(
        self,
        time_range: str = "evening",
        weekend_only: bool = False,
        preserve_order: bool = True,
        start_date: str | None = None,
        dry_run: bool = True,
        force: bool = False,
    ) -> FilterResult:
        """Change commit dates to specified time range.

        Args:
            time_range: Preset ('evening', 'night', 'weekend', 'random') or custom 'HH:MM-HH:MM'
            weekend_only: If True, move all commits to weekends
            preserve_order: If True, maintain relative commit order
            start_date: Start date for commits (YYYY-MM-DD)
            dry_run: If True, only show what would be changed
            force: If True, allow running on repo with existing filter-repo state
        """
        commits = self.get_commits()
        if not commits:
            return FilterResult(success=True, message="No commits to modify", commits_processed=0)

        parsed = self._parse_time_range(time_range)
        if isinstance(parsed, FilterResult):
            return parsed
        start_hour, start_min, end_hour, end_min = parsed

        # The "weekend" preset implies weekend-only days; otherwise the name
        # is misleading because it would also place commits on weekdays.
        effective_weekend_only = weekend_only or time_range == "weekend"

        mappings = self._generate_date_mappings(
            commits, start_hour, start_min, end_hour, end_min,
            effective_weekend_only, preserve_order, start_date,
        )
        if isinstance(mappings, FilterResult):
            return mappings

        if dry_run:
            commit_by_hash = {c.hash: c for c in commits}
            preview = []
            for commit_hash, (new_ts, tz) in list(mappings.items())[:10]:
                orig_commit = commit_by_hash.get(commit_hash)
                if orig_commit:
                    new_dt = datetime.datetime.fromtimestamp(new_ts, tz=datetime.timezone.utc)
                    preview.append(
                        f"{commit_hash[:8]}: {orig_commit.date[:19]} -> {new_dt.isoformat()[:19]}"
                    )
            return FilterResult(
                success=True,
                message=f"Dry run: {len(mappings)} commits would have dates changed\n\nPreview:\n" + "\n".join(preview),
                commits_processed=len(commits),
                commits_rewritten=len(mappings),
                dry_run=True,
            )

        callback_code = self._create_date_callback_code(mappings)
        result = self._run_filter_repo(
            "--commit-callback", callback_code,
            dry_run=False, force=force,
        )
        if result.returncode != 0:
            return FilterResult(success=False, message="Failed to change commit dates", error=result.stderr)
        return FilterResult(
            success=True,
            message=f"Successfully changed dates for {len(mappings)} commits",
            commits_processed=len(commits),
            commits_rewritten=len(mappings),
        )
