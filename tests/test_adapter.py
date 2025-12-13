"""Adapter tests."""

import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


class TestPathNormalization:
    def test_linux_absolute_path_preserved_on_windows(self):
        from git_filter_repo_mcp.adapter import GitFilterRepoAdapter

        with patch("platform.system", return_value="Windows"):
            assert GitFilterRepoAdapter._normalize_path("/root/test-repo") == "/root/test-repo"
            assert GitFilterRepoAdapter._normalize_path("/home/user/repo") == "/home/user/repo"
            assert GitFilterRepoAdapter._normalize_path("/tmp/test") == "/tmp/test"

    def test_git_bash_path_converted_on_windows(self):
        from git_filter_repo_mcp.adapter import GitFilterRepoAdapter

        with patch("platform.system", return_value="Windows"):
            assert GitFilterRepoAdapter._normalize_path("/c/Users/test") == "C:\\Users\\test"
            assert GitFilterRepoAdapter._normalize_path("/d/Projects/repo") == "D:\\Projects\\repo"

    def test_windows_path_with_forward_slashes(self):
        from git_filter_repo_mcp.adapter import GitFilterRepoAdapter

        with patch("platform.system", return_value="Windows"):
            assert GitFilterRepoAdapter._normalize_path("C:/Users/test") == "C:\\Users\\test"

    def test_wsl_paths_preserved(self):
        from git_filter_repo_mcp.adapter import GitFilterRepoAdapter

        with patch("platform.system", return_value="Windows"):
            assert GitFilterRepoAdapter._normalize_path("//wsl$/Ubuntu/home/user") == "//wsl$/Ubuntu/home/user"

    def test_paths_unchanged_on_linux(self):
        from git_filter_repo_mcp.adapter import GitFilterRepoAdapter

        with patch("platform.system", return_value="Linux"):
            assert GitFilterRepoAdapter._normalize_path("/c/Users/test") == "/c/Users/test"
            assert GitFilterRepoAdapter._normalize_path("/root/test") == "/root/test"


# Marker for tests requiring git-filter-repo
_requires_git_filter_repo = pytest.mark.skipif(
    not any(
        (Path(p) / "git-filter-repo").exists() or (Path(p) / "git-filter-repo.exe").exists()
        for p in os.environ.get("PATH", "").split(os.pathsep)
    ),
    reason="git-filter-repo not installed",
)


@pytest.fixture
def temp_git_repo():
    """Create a temporary git repository for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir) / "test_repo"
        repo_path.mkdir()

        # Initialize git repo
        subprocess.run(["git", "init"], cwd=repo_path, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"], cwd=repo_path, capture_output=True
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"], cwd=repo_path, capture_output=True
        )

        # Create initial commit
        (repo_path / "README.md").write_text("# Test Repo")
        subprocess.run(["git", "add", "."], cwd=repo_path, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Initial commit"], cwd=repo_path, capture_output=True
        )

        # Create more commits
        (repo_path / "main.py").write_text("print('hello')")
        subprocess.run(["git", "add", "."], cwd=repo_path, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Add main.py"], cwd=repo_path, capture_output=True)

        (repo_path / "config.json").write_text('{"key": "value"}')
        subprocess.run(["git", "add", "."], cwd=repo_path, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Add config"], cwd=repo_path, capture_output=True)

        yield repo_path


@_requires_git_filter_repo
class TestGitFilterRepoAdapter:
    def test_validate_repo(self, temp_git_repo):
        from git_filter_repo_mcp.adapter import GitFilterRepoAdapter

        # Should not raise
        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        assert adapter.repo_path == temp_git_repo.resolve()

    def test_invalid_repo_raises(self):
        from git_filter_repo_mcp.adapter import GitFilterRepoAdapter

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Not a git repository"):
                GitFilterRepoAdapter(tmpdir)

    def test_get_commits(self, temp_git_repo):
        from git_filter_repo_mcp.adapter import GitFilterRepoAdapter

        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        commits = adapter.get_commits()

        assert len(commits) == 3
        assert commits[0].message == "Add config"
        assert commits[1].message == "Add main.py"
        assert commits[2].message == "Initial commit"

    def test_get_commits_with_limit(self, temp_git_repo):
        from git_filter_repo_mcp.adapter import GitFilterRepoAdapter

        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        commits = adapter.get_commits(max_count=2)

        assert len(commits) == 2

    def test_get_commit_files(self, temp_git_repo):
        from git_filter_repo_mcp.adapter import GitFilterRepoAdapter

        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        commits = adapter.get_commits()

        # Latest commit should have config.json
        files = adapter.get_commit_files(commits[0].hash)
        assert "config.json" in files

    def test_analyze_history(self, temp_git_repo):
        from git_filter_repo_mcp.adapter import GitFilterRepoAdapter

        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        analysis = adapter.analyze_history()

        assert analysis["total_commits"] == 3
        assert "Test User <test@example.com>" in analysis["authors"]
        assert analysis["authors"]["Test User <test@example.com>"] == 3

    def test_create_backup(self, temp_git_repo):
        from git_filter_repo_mcp.adapter import GitFilterRepoAdapter

        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        backup_branch = adapter.create_backup()

        assert backup_branch.startswith("backup_")

        # Verify branch exists
        result = subprocess.run(
            ["git", "branch", "--list", backup_branch],
            cwd=temp_git_repo,
            capture_output=True,
            text=True,
        )
        assert backup_branch in result.stdout

    def test_list_all_files_in_history(self, temp_git_repo):
        from git_filter_repo_mcp.adapter import GitFilterRepoAdapter

        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        files = adapter.list_all_files_in_history()

        assert "README.md" in files
        assert "main.py" in files
        assert "config.json" in files

    def test_get_file_history(self, temp_git_repo):
        from git_filter_repo_mcp.adapter import GitFilterRepoAdapter

        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        history = adapter.get_file_history("README.md")

        assert len(history) == 1
        assert history[0]["message"] == "Initial commit"


@_requires_git_filter_repo
class TestDryRunOperations:
    def test_rewrite_messages_dry_run(self, temp_git_repo):
        from git_filter_repo_mcp.adapter import GitFilterRepoAdapter

        adapter = GitFilterRepoAdapter(str(temp_git_repo))

        def callback(msg, hash):
            return f"[REWRITTEN] {msg}"

        result = adapter.rewrite_commit_messages(callback, dry_run=True)

        assert result.success is True
        assert result.dry_run is True
        assert result.commits_rewritten == 3

        # Verify commits are NOT actually changed
        commits = adapter.get_commits()
        assert not commits[0].message.startswith("[REWRITTEN]")

    def test_change_author_dry_run(self, temp_git_repo):
        from git_filter_repo_mcp.adapter import GitFilterRepoAdapter

        adapter = GitFilterRepoAdapter(str(temp_git_repo))

        result = adapter.change_author(
            old_email="test@example.com",
            new_name="New Name",
            new_email="new@example.com",
            dry_run=True,
        )

        assert result.success is True
        assert result.dry_run is True
        assert result.commits_rewritten == 3

        # Verify author is NOT actually changed
        commits = adapter.get_commits()
        assert commits[0].author_email == "test@example.com"

    def test_squash_commits_dry_run(self, temp_git_repo):
        from git_filter_repo_mcp.adapter import GitFilterRepoAdapter

        adapter = GitFilterRepoAdapter(str(temp_git_repo))

        commits = adapter.get_commits()
        start_commit = commits[2].hash  # Initial commit

        result = adapter.squash_commits(start_commit=start_commit, dry_run=True)

        assert result.success is True
        assert result.dry_run is True
        assert result.commits_processed == 2  # 2 commits after initial

        # Verify commits are NOT actually squashed
        commits_after = adapter.get_commits()
        assert len(commits_after) == 3

    def test_change_commit_dates_dry_run(self, temp_git_repo):
        from git_filter_repo_mcp.adapter import GitFilterRepoAdapter

        adapter = GitFilterRepoAdapter(str(temp_git_repo))

        result = adapter.change_commit_dates(
            time_range="evening",
            dry_run=True,
        )

        assert result.success is True
        assert result.dry_run is True
        assert result.commits_rewritten == 3
        assert "Preview:" in result.message

    def test_change_commit_dates_custom_range_dry_run(self, temp_git_repo):
        from git_filter_repo_mcp.adapter import GitFilterRepoAdapter

        adapter = GitFilterRepoAdapter(str(temp_git_repo))

        result = adapter.change_commit_dates(
            time_range="20:00-23:00",
            dry_run=True,
        )

        assert result.success is True
        assert result.dry_run is True

    def test_change_commit_dates_weekend_only_dry_run(self, temp_git_repo):
        from git_filter_repo_mcp.adapter import GitFilterRepoAdapter

        adapter = GitFilterRepoAdapter(str(temp_git_repo))

        result = adapter.change_commit_dates(
            time_range="random",
            weekend_only=True,
            dry_run=True,
        )

        assert result.success is True
        assert result.dry_run is True

    def test_change_commit_dates_invalid_range(self, temp_git_repo):
        from git_filter_repo_mcp.adapter import GitFilterRepoAdapter

        adapter = GitFilterRepoAdapter(str(temp_git_repo))

        result = adapter.change_commit_dates(
            time_range="invalid",
            dry_run=True,
        )

        assert result.success is False
        assert "Unknown time range" in result.message
