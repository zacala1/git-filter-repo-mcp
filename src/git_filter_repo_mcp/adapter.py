"""git-filter-repo adapter - wraps git-filter-repo commands."""

import base64
import datetime
import json
import logging
import platform
import re
import shutil
import subprocess
import tempfile
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


def _parse_lines(output: str) -> list[str]:
    """Parse stdout into non-empty lines."""
    return [line for line in output.strip().split("\n") if line]


def _safe_int(value: str, default: int = 0) -> int:
    """Safely parse int from string."""
    try:
        return int(value.strip())
    except (ValueError, AttributeError):
        return default


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
        self.repo_path = Path(self._normalize_path(repo_path)).resolve()
        self._validate_repo()
        self._check_git_filter_repo()

    @staticmethod
    def _normalize_path(path: str) -> str:
        """Normalize path for cross-platform compatibility."""
        if platform.system() != "Windows":
            return path

        # Git Bash style: /c/Users/... -> C:\Users\...
        if match := re.match(r"^/([a-zA-Z])/(.*)$", path):
            return f"{match.group(1).upper()}:\\{match.group(2).replace('/', '\\')}"

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
        git_dir = self.repo_path / ".git"
        if not git_dir.exists():
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
                args, cwd=self.repo_path, capture_output=True, text=True, check=check, timeout=timeout
            )
        except subprocess.TimeoutExpired:
            logger.error(f"timeout {timeout}s: {args[0]}")
            raise

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

    # Record separator that won't appear in commit messages/names
    _FIELD_SEP = "\x1e"

    def get_commits(self, branch: str = "HEAD", max_count: int | None = None) -> list[CommitInfo]:
        """Get commit information from the repository."""
        sep = self._FIELD_SEP
        args = ["log", f"--format=%H{sep}%an{sep}%ae{sep}%cn{sep}%ce{sep}%s{sep}%aI", branch]
        if max_count:
            args.append(f"-n{max_count}")

        result = self._run_git(*args)
        commits = []
        for line in _parse_lines(result.stdout):
            parts = line.split(sep, 6)
            if len(parts) >= 7:
                commits.append(CommitInfo(*parts[:7]))
        return commits

    def get_commit_diff(self, commit_hash: str) -> str:
        """Get diff for a commit."""
        return self._run_git_fast("show", "--stat", commit_hash).stdout or ""

    def get_commit_files(self, commit_hash: str) -> list[str]:
        """Get files changed in a commit."""
        return _parse_lines(self._run_git_fast("show", "--name-only", "--format=", commit_hash).stdout)

    def analyze_history(self, branch: str = "HEAD", max_count: int = 100) -> dict:
        """Analyze repository history for potential rewrites."""
        commits = self.get_commits(branch, max_count)

        # Collect statistics
        authors = {}
        for commit in commits:
            author_key = f"{commit.author_name} <{commit.author_email}>"
            authors[author_key] = authors.get(author_key, 0) + 1

        # Skip file count for performance - can be slow on large repos

        return {
            "total_commits": len(commits),
            "authors": authors,
            "commits": [
                {
                    "hash": c.hash[:8],
                    "author": f"{c.author_name} <{c.author_email}>",
                    "message": c.message[:80],
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

        # Create message-callback script with safe base64-encoded data
        # This avoids exec() with user-controlled content
        replacements = {old: new for _, old, new in rewrites}
        encoded_replacements = base64.b64encode(json.dumps(replacements).encode()).decode()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            # Write self-contained script with embedded data
            # git-filter-repo exec's this with `message` (bytes) in scope;
            # we must reassign `message` for changes to take effect.
            f.write(f'''import base64, json
_DATA = "{encoded_replacements}"
_REPLACEMENTS = json.loads(base64.b64decode(_DATA).decode())
_msg_str = message.decode('utf-8') if isinstance(message, bytes) else message
_new_msg = _REPLACEMENTS.get(_msg_str.strip(), _msg_str)
message = _new_msg.encode('utf-8') if isinstance(message, bytes) else _new_msg
''')
            script_path = f.name

        try:
            # Use --message-callback with file: prefix (safer than exec)
            result = self._run_filter_repo(
                "--message-callback",
                f"filename:{script_path}",
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
        finally:
            Path(script_path).unlink(missing_ok=True)

    def change_author(
        self,
        old_email: str,
        new_name: str,
        new_email: str,
        dry_run: bool = True,
        force: bool = False,
    ) -> FilterResult:
        """Change author/committer information for commits."""
        # Count affected commits
        commits = self.get_commits()
        affected = [c for c in commits if c.author_email == old_email]

        if dry_run:
            return FilterResult(
                success=True,
                message=f"Dry run: {len(affected)} commits would be updated",
                commits_processed=len(commits),
                commits_rewritten=len(affected),
                dry_run=True,
            )

        # Create mailmap file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mailmap", delete=False) as f:
            f.write(f"{new_name} <{new_email}> <{old_email}>\n")
            mailmap_path = f.name

        try:
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
        finally:
            Path(mailmap_path).unlink(missing_ok=True)

    def remove_files(
        self,
        paths: list[str],
        dry_run: bool = True,
        force: bool = False,
    ) -> FilterResult:
        """Remove files from entire git history."""
        if dry_run:
            try:
                result = self._run_git("log", "--all", "--format=%H", "--", *paths)
                affected = set(_parse_lines(result.stdout))
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                affected = set()
                for path in paths:
                    result = self._run_git("log", "--all", "--format=%H", "--", path)
                    affected.update(_parse_lines(result.stdout))

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
        """Remove files larger than threshold from history."""
        result = self._run_git("rev-list", "--objects", "--all")
        size_bytes = int(size_threshold_mb * 1024 * 1024)

        # Parse objects with paths
        objects_with_paths = {}
        object_hashes = []
        for line in _parse_lines(result.stdout):
            parts = line.split(" ", 1)
            if len(parts) == 2:
                objects_with_paths[parts[0]] = parts[1]
                object_hashes.append(parts[0])

        if not object_hashes:
            return FilterResult(success=True, message="No files found", files_affected=[], dry_run=dry_run)

        # Batch size check
        large_files = []
        try:
            batch_result = subprocess.run(
                ["git", "cat-file", "--batch-check=%(objectsize)"],
                cwd=self.repo_path, input="\n".join(object_hashes),
                capture_output=True, text=True, timeout=TIMEOUT_DEFAULT,
            )
            for blob_hash, size_str in zip(object_hashes, _parse_lines(batch_result.stdout)):
                size = _safe_int(size_str)
                if size > size_bytes:
                    large_files.append((objects_with_paths.get(blob_hash, blob_hash), size / (1024 * 1024)))
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.warning(f"batch failed: {e}")
            for blob_hash in object_hashes[:100]:
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

        # Remove large files
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
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_branch = f"backup_{timestamp}"

        self._run_git("branch", backup_branch)
        return backup_branch

    def restore_backup(self, backup_branch: str) -> FilterResult:
        """Restore from a backup branch."""
        try:
            # Get current branch
            result = self._run_git("rev-parse", "--abbrev-ref", "HEAD")
            current_branch = result.stdout.strip()

            # Reset to backup
            self._run_git("reset", "--hard", backup_branch)
            self._run_git("branch", "-D", backup_branch)

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

    def _collect_commit_files(
        self, commits: list[CommitInfo], branch: str, max_commits: int,
    ) -> dict[str, list[str]]:
        """Collect file lists for commits, bulk-fetched with per-commit fallback."""
        try:
            result = self._run_git(
                "log", "--name-only", "--format=%H", f"-n{max_commits}", branch,
            )
            commit_files_map: dict[str, list[str]] = {}
            current_hash = None
            for line in _parse_lines(result.stdout):
                if len(line) == 40 and all(c in "0123456789abcdef" for c in line):
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
        """Scan file contents for secrets, returning finding dicts."""
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
        """Scan repository history for potential secrets."""
        from .secrets import get_file_risk_level, is_sensitive_file

        commits = self.get_commits(branch, max_commits)
        commit_files_map = self._collect_commit_files(commits, branch, max_commits)

        sensitive_files = []
        files_to_scan: list[tuple[str, str]] = []
        for commit in commits:
            for file_path in commit_files_map.get(commit.hash, []):
                if is_sensitive_file(file_path):
                    sensitive_files.append({
                        "file": file_path, "commit": commit.hash[:8], "risk": get_file_risk_level(file_path),
                    })
                if len(files_to_scan) < MAX_FILES_TO_SCAN:
                    files_to_scan.append((commit.hash, file_path))

        findings = self._scan_file_contents(files_to_scan)

        return {
            "commits_scanned": len(commits), "secrets_found": len(findings),
            "sensitive_files": len(sensitive_files), "findings": findings[:MAX_FINDINGS_LIMIT],
            "sensitive_file_list": sensitive_files[:MAX_PREVIEW_COMMITS], "files_scanned": len(files_to_scan),
        }

    def rewrite_single_commit(
        self,
        commit_hash: str,
        new_message: str | None = None,
        new_author_name: str | None = None,
        new_author_email: str | None = None,
        force: bool = True,
    ) -> FilterResult:
        """Rewrite a single commit's message and/or author in one filter-repo pass."""
        changes = {}
        if new_message:
            changes["message"] = new_message
        if new_author_name:
            changes["author_name"] = new_author_name
        if new_author_email:
            changes["author_email"] = new_author_email

        if not changes:
            return FilterResult(success=False, message="No changes specified")

        encoded_changes = base64.b64encode(json.dumps(changes).encode()).decode()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            # git-filter-repo exec's this with `commit` object in scope.
            # We check original_id to only modify the target commit.
            f.write(f'''import base64, json
_DATA = "{encoded_changes}"
_CHANGES = json.loads(base64.b64decode(_DATA).decode())
_TARGET = "{commit_hash}"
_orig_id = commit.original_id.decode() if commit.original_id else None
if _orig_id and (_orig_id.startswith(_TARGET) or _TARGET.startswith(_orig_id)):
    if "message" in _CHANGES:
        commit.message = _CHANGES["message"].encode()
    if "author_name" in _CHANGES:
        commit.author_name = _CHANGES["author_name"].encode()
    if "author_email" in _CHANGES:
        commit.author_email = b"<" + _CHANGES["author_email"].encode() + b">"
    if "author_name" in _CHANGES:
        commit.committer_name = _CHANGES["author_name"].encode()
    if "author_email" in _CHANGES:
        commit.committer_email = b"<" + _CHANGES["author_email"].encode() + b">"
''')
            script_path = f.name

        try:
            result = self._run_filter_repo(
                "--commit-callback",
                f"filename:{script_path}",
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
            if new_message:
                changes_made.append("message")
            if new_author_name or new_author_email:
                changes_made.append("author")

            return FilterResult(
                success=True,
                message=f"Updated commit {commit_hash[:8]}: {', '.join(changes_made)}",
                commits_rewritten=1,
            )
        finally:
            Path(script_path).unlink(missing_ok=True)

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
        sep = self._FIELD_SEP
        history = []
        for line in _parse_lines(self._run_git("log", "--follow", f"--format=%H{sep}%an{sep}%ae{sep}%s{sep}%aI", "--", file_path).stdout):
            parts = line.split(sep, 4)
            if len(parts) >= 5:
                history.append({"hash": parts[0][:8], "author": f"{parts[1]} <{parts[2]}>", "message": parts[3], "date": parts[4]})
        return history

    def squash_commits(self, start_commit: str, end_commit: str = "HEAD", new_message: str | None = None, dry_run: bool = True) -> FilterResult:
        """Squash a range of commits into one."""
        commit_count = _safe_int(self._run_git("rev-list", "--count", f"{start_commit}..{end_commit}").stdout)
        if commit_count == 0:
            return FilterResult(success=False, message=f"Invalid commit range: {start_commit}..{end_commit}")

        if dry_run:
            return FilterResult(success=True, message=f"Dry run: would squash {commit_count} commits", commits_processed=commit_count, dry_run=True)

        try:
            if not new_message:
                messages = _parse_lines(self._run_git("log", "--format=%s", f"{start_commit}..{end_commit}").stdout)
                new_message = "Squashed commits:\n" + "\n".join(f"- {m}" for m in messages) if messages else "Squashed commits"

            self._run_git("reset", "--soft", start_commit)
            self._run_git("commit", "-m", new_message)
            return FilterResult(success=True, message=f"Squashed {commit_count} commits", commits_processed=commit_count, commits_rewritten=1)
        except subprocess.CalledProcessError as e:
            return FilterResult(success=False, message="Failed to squash", error=str(e))

    def replace_text_in_history(
        self, old_text: str, new_text: str, file_pattern: str | None = None, dry_run: bool = True, force: bool = False,
    ) -> FilterResult:
        """Replace text throughout repository history."""
        # Escape ==> in new_text to prevent git-filter-repo expression parsing issues
        safe_new_text = new_text.replace("==>", "\\=\\=\\>")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(f"regex:{re.escape(old_text)}==>{safe_new_text}\n")
            expressions_path = f.name

        if dry_run:
            # Search git history (not working directory) for affected files
            try:
                git_args = ["log", "--all", "-S", old_text, "--name-only", "--format="]
                if file_pattern:
                    git_args.extend(["--", file_pattern])
                result = self._run_git(*git_args)
                files_with_matches = sorted(set(_parse_lines(result.stdout)))
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                files_with_matches = []
            Path(expressions_path).unlink(missing_ok=True)
            return FilterResult(success=True, message=f"Dry run: {len(files_with_matches)} files in history", files_affected=files_with_matches[:20], dry_run=True)

        try:
            args = ["--replace-text", expressions_path]
            if file_pattern:
                args.extend(["--path-glob", file_pattern])
            result = self._run_filter_repo(*args, dry_run=False, force=force)
            if result.returncode != 0:
                return FilterResult(success=False, message="Failed to replace text", error=result.stderr)
            return FilterResult(success=True, message="Replaced text in history")
        finally:
            Path(expressions_path).unlink(missing_ok=True)

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
        """Generate new date mappings for commits within the specified time constraints."""
        import random

        commit_dates = []
        for commit in commits:
            try:
                orig_dt = datetime.datetime.fromisoformat(commit.date.replace("Z", "+00:00"))
                commit_dates.append((commit.hash, orig_dt))
            except ValueError:
                continue

        commit_dates.sort(key=lambda x: x[1])

        if start_date:
            try:
                base_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
            except ValueError:
                return FilterResult(
                    success=False,
                    message=f"Invalid start_date format: {start_date}",
                    error="Use YYYY-MM-DD format",
                )
        else:
            base_date = commit_dates[0][1] if commit_dates else datetime.datetime.now()

        date_mappings: dict[str, tuple[int, str]] = {}
        current_date = base_date
        prev_timestamp = None

        for commit_hash, orig_dt in commit_dates:
            tz_offset = orig_dt.strftime("%z") or "+0000"

            found_valid = False
            for _ in range(100):
                if end_hour >= start_hour:
                    hour = random.randint(start_hour, end_hour)
                    if hour == end_hour:
                        minute = random.randint(0, end_min)
                    elif hour == start_hour:
                        minute = random.randint(start_min, 59)
                    else:
                        minute = random.randint(0, 59)
                else:
                    if random.random() < 0.5:
                        hour = random.randint(start_hour, 23)
                    else:
                        hour = random.randint(0, end_hour)
                    minute = random.randint(0, 59)

                second = random.randint(0, 59)
                new_dt = current_date.replace(hour=hour, minute=minute, second=second, microsecond=0)

                if weekend_only and new_dt.weekday() < 5:
                    days_until_saturday = (5 - new_dt.weekday()) % 7
                    if days_until_saturday == 0:
                        days_until_saturday = 7
                    current_date = current_date + datetime.timedelta(days=days_until_saturday)
                    continue

                if preserve_order and prev_timestamp:
                    if new_dt <= prev_timestamp:
                        new_dt = prev_timestamp + datetime.timedelta(minutes=random.randint(5, 60))
                        current_date = new_dt

                found_valid = True
                break

            if not found_valid:
                if prev_timestamp:
                    new_dt = prev_timestamp + datetime.timedelta(minutes=random.randint(5, 30))

            date_mappings[commit_hash] = (int(new_dt.timestamp()), tz_offset)
            prev_timestamp = new_dt

            if random.random() < 0.3:
                current_date = current_date + datetime.timedelta(days=1)

        return date_mappings

    def _create_date_callback_script(self, date_mappings: dict[str, tuple[int, str]]) -> str:
        """Create a temporary callback script for date rewriting. Caller must delete."""
        serializable = {h: [ts, tz] for h, (ts, tz) in date_mappings.items()}
        encoded = base64.b64encode(json.dumps(serializable).encode()).decode()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(f'''import base64, json
_DATA = "{encoded}"
_DATE_MAP = json.loads(base64.b64decode(_DATA).decode())
_commit_hash = commit.original_id.decode() if commit.original_id else None
if _commit_hash and _commit_hash in _DATE_MAP:
    _new_ts, _tz_offset = _DATE_MAP[_commit_hash]
    _new_date = f"{{_new_ts}} {{_tz_offset}}".encode()
    commit.author_date = _new_date
    commit.committer_date = _new_date
''')
            return f.name

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

        mappings = self._generate_date_mappings(
            commits, start_hour, start_min, end_hour, end_min,
            weekend_only, preserve_order, start_date,
        )
        if isinstance(mappings, FilterResult):
            return mappings

        if dry_run:
            preview = []
            for commit_hash, (new_ts, tz) in list(mappings.items())[:10]:
                orig_commit = next((c for c in commits if c.hash == commit_hash), None)
                if orig_commit:
                    new_dt = datetime.datetime.fromtimestamp(new_ts)
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

        script_path = self._create_date_callback_script(mappings)
        try:
            result = self._run_filter_repo(
                "--commit-callback", f"filename:{script_path}",
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
        finally:
            Path(script_path).unlink(missing_ok=True)
