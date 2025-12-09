"""git-filter-repo adapter - wraps git-filter-repo commands."""

import base64
import datetime
import json
import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

# Constants
DEFAULT_TIMEOUT = 300  # 5 minutes
MAX_FILES_LIMIT = 1000
MAX_PREVIEW_COMMITS = 20
MAX_FINDINGS_LIMIT = 50


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
        self.repo_path = Path(repo_path).resolve()
        self._validate_repo()
        self._check_git_filter_repo()

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
        self, args: list[str], check: bool = True, timeout: int = DEFAULT_TIMEOUT
    ) -> subprocess.CompletedProcess:
        """Run a command in the repo directory."""
        try:
            return subprocess.run(
                args,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=check,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out after {timeout}s: {' '.join(args)}")
            raise

    def _run_git(self, *args: str) -> subprocess.CompletedProcess:
        """Run a git command."""
        return self._run_command(["git", *args])

    def _run_filter_repo(
        self, *args: str, dry_run: bool = False, force: bool = False
    ) -> subprocess.CompletedProcess:
        """Run git-filter-repo with given arguments."""
        cmd = ["git-filter-repo"]
        if dry_run:
            cmd.append("--dry-run")
        if force:
            cmd.append("--force")
        cmd.extend(args)
        return self._run_command(cmd, check=False)

    def get_commits(self, branch: str = "HEAD", max_count: int | None = None) -> list[CommitInfo]:
        """Get commit information from the repository."""
        format_str = "%H|%an|%ae|%cn|%ce|%s|%aI"
        args = ["log", f"--format={format_str}", branch]
        if max_count:
            args.append(f"-n{max_count}")

        result = self._run_git(*args)
        commits = []

        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split("|", 6)
            if len(parts) >= 7:
                commits.append(
                    CommitInfo(
                        hash=parts[0],
                        author_name=parts[1],
                        author_email=parts[2],
                        committer_name=parts[3],
                        committer_email=parts[4],
                        message=parts[5],
                        date=parts[6],
                    )
                )

        return commits

    def get_commit_diff(self, commit_hash: str) -> str:
        """Get the diff for a specific commit."""
        result = self._run_git("show", "--stat", commit_hash)
        return result.stdout

    def get_commit_files(self, commit_hash: str) -> list[str]:
        """Get list of files changed in a commit."""
        result = self._run_git("show", "--name-only", "--format=", commit_hash)
        return [f for f in result.stdout.strip().split("\n") if f]

    def analyze_history(self, branch: str = "HEAD", max_count: int = 100) -> dict:
        """Analyze repository history for potential rewrites."""
        commits = self.get_commits(branch, max_count)

        # Collect statistics
        authors = {}
        files_changed = set()

        for commit in commits:
            author_key = f"{commit.author_name} <{commit.author_email}>"
            authors[author_key] = authors.get(author_key, 0) + 1
            files_changed.update(self.get_commit_files(commit.hash))

        return {
            "total_commits": len(commits),
            "authors": authors,
            "unique_files": len(files_changed),
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
            f.write(f'''import base64, json
_DATA = "{encoded_replacements}"
REPLACEMENTS = json.loads(base64.b64decode(_DATA).decode())

def message_callback(message):
    msg_str = message.decode('utf-8') if isinstance(message, bytes) else message
    new_msg = REPLACEMENTS.get(msg_str.strip(), msg_str)
    return new_msg.encode('utf-8') if isinstance(message, bytes) else new_msg
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
            # Check which commits would be affected
            affected_commits = set()
            for path in paths:
                result = self._run_git("log", "--all", "--format=%H", "--", path)
                affected_commits.update(result.stdout.strip().split("\n"))
            affected_commits.discard("")

            return FilterResult(
                success=True,
                message=f"Dry run: {len(affected_commits)} commits would be affected",
                commits_rewritten=len(affected_commits),
                files_affected=paths,
                dry_run=True,
            )

        args = []
        for path in paths:
            args.extend(["--path", path, "--invert-paths"])

        result = self._run_filter_repo(*args, dry_run=False, force=force)

        if result.returncode != 0:
            return FilterResult(
                success=False,
                message="Failed to remove files",
                error=result.stderr,
            )

        return FilterResult(
            success=True,
            message=f"Successfully removed {len(paths)} paths from history",
            files_affected=paths,
        )

    def remove_large_files(
        self,
        size_threshold_mb: float = 10.0,
        dry_run: bool = True,
        force: bool = False,
    ) -> FilterResult:
        """Remove files larger than threshold from history."""
        # Find large blobs
        result = self._run_git("rev-list", "--objects", "--all")

        large_files = []
        size_bytes = int(size_threshold_mb * 1024 * 1024)

        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) < 2:
                continue
            blob_hash, path = parts[0], parts[1]

            # Get blob size
            size_result = self._run_git("cat-file", "-s", blob_hash)
            try:
                blob_size = int(size_result.stdout.strip())
                if blob_size > size_bytes:
                    large_files.append((path, blob_size / (1024 * 1024)))
            except ValueError:
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
        """Filter repository to include/exclude specific paths."""
        args = []

        if include_paths:
            for path in include_paths:
                args.extend(["--path", path])

        if exclude_paths:
            for path in exclude_paths:
                args.extend(["--path", path, "--invert-paths"])

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

    def scan_secrets(
        self,
        branch: str = "HEAD",
        max_commits: int = 100,
    ) -> dict:
        """Scan repository history for potential secrets."""
        from .secrets import get_file_risk_level, is_sensitive_file, scan_content

        commits = self.get_commits(branch, max_commits)
        findings = []
        sensitive_files = []

        for commit in commits:
            files = self.get_commit_files(commit.hash)

            for file_path in files:
                # Check if it's a sensitive file by name
                if is_sensitive_file(file_path):
                    sensitive_files.append(
                        {
                            "file": file_path,
                            "commit": commit.hash[:8],
                            "risk": get_file_risk_level(file_path),
                        }
                    )

                # Get file content at that commit
                try:
                    result = self._run_git("show", f"{commit.hash}:{file_path}")
                    content = result.stdout

                    # Scan for secrets
                    file_findings = scan_content(content, file_path, commit.hash)
                    for finding in file_findings:
                        findings.append(
                            {
                                "type": finding.pattern_name,
                                "description": finding.description,
                                "severity": finding.severity,
                                "file": finding.file_path,
                                "commit": finding.commit_hash[:8],
                                "line": finding.line_number,
                                "matched": finding.matched_text,
                            }
                        )
                except subprocess.CalledProcessError:
                    # File might not exist at this commit
                    continue

        return {
            "commits_scanned": len(commits),
            "secrets_found": len(findings),
            "sensitive_files": len(sensitive_files),
            "findings": findings[:MAX_FINDINGS_LIMIT],
            "sensitive_file_list": sensitive_files[:MAX_PREVIEW_COMMITS],
        }

    def get_file_at_commit(self, commit_hash: str, file_path: str) -> str | None:
        """Get file content at a specific commit."""
        try:
            result = self._run_git("show", f"{commit_hash}:{file_path}")
            return result.stdout
        except subprocess.CalledProcessError:
            return None

    def list_all_files_in_history(self, limit: int = MAX_FILES_LIMIT) -> list[str]:
        """List all files that have ever existed in the repository."""
        result = self._run_git("log", "--all", "--name-only", "--format=")
        files = set()
        for line in result.stdout.strip().split("\n"):
            if line:
                files.add(line)
                if len(files) >= limit:
                    break
        return sorted(files)[:limit]

    def get_file_history(self, file_path: str) -> list[dict]:
        """Get commit history for a specific file."""
        result = self._run_git("log", "--follow", "--format=%H|%an|%ae|%s|%aI", "--", file_path)

        history = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split("|", 4)
            if len(parts) >= 5:
                history.append(
                    {
                        "hash": parts[0][:8],
                        "author": f"{parts[1]} <{parts[2]}>",
                        "message": parts[3],
                        "date": parts[4],
                    }
                )

        return history

    def squash_commits(
        self,
        start_commit: str,
        end_commit: str = "HEAD",
        new_message: str | None = None,
        dry_run: bool = True,
    ) -> FilterResult:
        """Squash a range of commits into one."""
        # Get commits in range
        result = self._run_git("rev-list", "--count", f"{start_commit}..{end_commit}")
        commit_count = int(result.stdout.strip())

        if dry_run:
            return FilterResult(
                success=True,
                message=f"Dry run: would squash {commit_count} commits",
                commits_processed=commit_count,
                dry_run=True,
            )

        # Perform squash using reset and commit
        try:
            # Get the message from commits if not provided
            if not new_message:
                result = self._run_git("log", "--format=%s", f"{start_commit}..{end_commit}")
                messages = result.stdout.strip().split("\n")
                new_message = "Squashed commits:\n" + "\n".join(f"- {m}" for m in messages)

            # Soft reset to start commit
            self._run_git("reset", "--soft", start_commit)

            # Commit with new message
            self._run_git("commit", "-m", new_message)

            return FilterResult(
                success=True,
                message=f"Squashed {commit_count} commits",
                commits_processed=commit_count,
                commits_rewritten=1,
            )
        except subprocess.CalledProcessError as e:
            return FilterResult(
                success=False,
                message="Failed to squash commits",
                error=str(e),
            )

    def replace_text_in_history(
        self,
        old_text: str,
        new_text: str,
        file_pattern: str | None = None,
        dry_run: bool = True,
        force: bool = False,
    ) -> FilterResult:
        """Replace text throughout repository history."""
        # Create expressions file for git-filter-repo
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            # Use regex format: literal==>replacement
            import re

            escaped_old = re.escape(old_text)
            f.write(f"regex:{escaped_old}==>{new_text}\n")
            expressions_path = f.name

        if dry_run:
            # Count occurrences
            grep_args = ["grep", "-r", "-l", old_text, "."]
            if file_pattern:
                grep_args.extend(["--include", file_pattern])

            try:
                result = self._run_command(grep_args, check=False)
                files_with_matches = [f for f in result.stdout.strip().split("\n") if f]
            except Exception:
                files_with_matches = []

            Path(expressions_path).unlink(missing_ok=True)

            return FilterResult(
                success=True,
                message=f"Dry run: would replace in {len(files_with_matches)} files",
                files_affected=files_with_matches[:20],
                dry_run=True,
            )

        try:
            args = ["--replace-text", expressions_path]
            if file_pattern:
                args.extend(["--path-glob", file_pattern])

            result = self._run_filter_repo(*args, dry_run=False, force=force)

            if result.returncode != 0:
                return FilterResult(
                    success=False,
                    message="Failed to replace text",
                    error=result.stderr,
                )

            return FilterResult(
                success=True,
                message="Successfully replaced text in history",
            )
        finally:
            Path(expressions_path).unlink(missing_ok=True)
