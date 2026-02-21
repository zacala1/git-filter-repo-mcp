"""Adapter tests."""

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from git_filter_repo_mcp.adapter import FilterResult, GitFilterRepoAdapter
from tests.conftest import requires_git_filter_repo


class TestPathNormalization:
    def test_linux_absolute_path_preserved_on_windows(self):

        with patch("platform.system", return_value="Windows"):
            assert GitFilterRepoAdapter._normalize_path("/root/test-repo") == "/root/test-repo"
            assert GitFilterRepoAdapter._normalize_path("/home/user/repo") == "/home/user/repo"
            assert GitFilterRepoAdapter._normalize_path("/tmp/test") == "/tmp/test"

    def test_git_bash_path_converted_on_windows(self):

        with patch("platform.system", return_value="Windows"):
            assert GitFilterRepoAdapter._normalize_path("/c/Users/test") == "C:\\Users\\test"
            assert GitFilterRepoAdapter._normalize_path("/d/Projects/repo") == "D:\\Projects\\repo"

    def test_windows_path_with_forward_slashes(self):

        with patch("platform.system", return_value="Windows"):
            assert GitFilterRepoAdapter._normalize_path("C:/Users/test") == "C:\\Users\\test"

    def test_wsl_paths_preserved(self):

        with patch("platform.system", return_value="Windows"):
            assert GitFilterRepoAdapter._normalize_path("//wsl$/Ubuntu/home/user") == "//wsl$/Ubuntu/home/user"

    def test_paths_unchanged_on_linux(self):

        with patch("platform.system", return_value="Linux"):
            assert GitFilterRepoAdapter._normalize_path("/c/Users/test") == "/c/Users/test"
            assert GitFilterRepoAdapter._normalize_path("/root/test") == "/root/test"


@requires_git_filter_repo
class TestGitFilterRepoAdapter:
    def test_validate_repo(self, temp_git_repo):

        # Should not raise
        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        assert adapter.repo_path == temp_git_repo.resolve()

    def test_invalid_repo_raises(self):

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Not a git repository"):
                GitFilterRepoAdapter(tmpdir)

    def test_get_commits(self, temp_git_repo):

        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        commits = adapter.get_commits()

        assert len(commits) == 3
        assert commits[0].message == "Add config"
        assert commits[1].message == "Add main.py"
        assert commits[2].message == "Initial commit"

    def test_get_commits_with_limit(self, temp_git_repo):

        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        commits = adapter.get_commits(max_count=2)

        assert len(commits) == 2

    def test_get_commit_files(self, temp_git_repo):

        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        commits = adapter.get_commits()

        # Latest commit should have config.json
        files = adapter.get_commit_files(commits[0].hash)
        assert "config.json" in files

    def test_analyze_history(self, temp_git_repo):

        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        analysis = adapter.analyze_history()

        assert analysis["total_commits"] == 3
        assert "Test User <test@example.com>" in analysis["authors"]
        assert analysis["authors"]["Test User <test@example.com>"] == 3

    def test_create_backup(self, temp_git_repo):

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

        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        files = adapter.list_all_files_in_history()

        assert "README.md" in files
        assert "main.py" in files
        assert "config.json" in files

    def test_get_file_history(self, temp_git_repo):

        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        history = adapter.get_file_history("README.md")

        assert len(history) == 1
        assert history[0]["message"] == "Initial commit"


@requires_git_filter_repo
class TestDryRunOperations:
    def test_rewrite_messages_dry_run(self, temp_git_repo):

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

        adapter = GitFilterRepoAdapter(str(temp_git_repo))

        result = adapter.change_commit_dates(
            time_range="20:00-23:00",
            dry_run=True,
        )

        assert result.success is True
        assert result.dry_run is True

    def test_change_commit_dates_weekend_only_dry_run(self, temp_git_repo):

        adapter = GitFilterRepoAdapter(str(temp_git_repo))

        result = adapter.change_commit_dates(
            time_range="random",
            weekend_only=True,
            dry_run=True,
        )

        assert result.success is True
        assert result.dry_run is True

    def test_change_commit_dates_invalid_range(self, temp_git_repo):

        adapter = GitFilterRepoAdapter(str(temp_git_repo))

        result = adapter.change_commit_dates(
            time_range="invalid",
            dry_run=True,
        )

        assert result.success is False
        assert "Unknown time range" in result.message

    def test_filter_paths_include_exclude_rejected(self, temp_git_repo):

        adapter = GitFilterRepoAdapter(str(temp_git_repo))

        result = adapter.filter_paths(
            include_paths=["src/"],
            exclude_paths=["tests/"],
            dry_run=True,
        )

        assert result.success is False
        assert "Cannot use include_paths and exclude_paths together" in result.message

    def test_replace_text_dry_run_searches_history(self, temp_git_repo):

        adapter = GitFilterRepoAdapter(str(temp_git_repo))

        result = adapter.replace_text_in_history(
            old_text="hello",
            new_text="world",
            dry_run=True,
        )

        assert result.success is True
        assert result.dry_run is True
        assert "history" in result.message


def _make_mock_adapter():
    """Create a GitFilterRepoAdapter instance without repo validation."""
    adapter = object.__new__(GitFilterRepoAdapter)
    adapter.repo_path = Path("/fake/repo")
    return adapter


class TestChangeDateHelpers:
    """Test private helpers extracted from change_commit_dates."""

    def test_parse_time_range_preset_evening(self):
        adapter = _make_mock_adapter()
        assert adapter._parse_time_range("evening") == (19, 0, 23, 0)

    def test_parse_time_range_preset_night(self):
        adapter = _make_mock_adapter()
        assert adapter._parse_time_range("night") == (22, 0, 2, 0)

    def test_parse_time_range_preset_weekend(self):
        adapter = _make_mock_adapter()
        assert adapter._parse_time_range("weekend") == (10, 0, 22, 0)

    def test_parse_time_range_preset_random(self):
        adapter = _make_mock_adapter()
        assert adapter._parse_time_range("random") == (0, 0, 23, 59)

    def test_parse_time_range_custom_valid(self):
        adapter = _make_mock_adapter()
        assert adapter._parse_time_range("18:30-22:00") == (18, 30, 22, 0)

    def test_parse_time_range_custom_hours_only(self):
        adapter = _make_mock_adapter()
        assert adapter._parse_time_range("9-17") == (9, 0, 17, 0)

    def test_parse_time_range_invalid_format(self):
        adapter = _make_mock_adapter()
        result = adapter._parse_time_range("abc-def")
        assert isinstance(result, FilterResult)
        assert result.success is False
        assert "Invalid time range" in result.message

    def test_parse_time_range_unknown_preset(self):
        adapter = _make_mock_adapter()
        result = adapter._parse_time_range("invalid")
        assert isinstance(result, FilterResult)
        assert result.success is False
        assert "Unknown time range" in result.message


@requires_git_filter_repo
class TestRealExecution:
    """Test non-dry-run adapter operations (requires git-filter-repo)."""

    def test_remove_files_actually_removes(self, temp_git_repo):
        adapter = GitFilterRepoAdapter(str(temp_git_repo))

        # config.json should exist in history
        files_before = adapter.list_all_files_in_history()
        assert "config.json" in files_before

        result = adapter.remove_files(["config.json"], dry_run=False, force=True)
        assert result.success is True
        assert result.dry_run is False

        # Re-create adapter to pick up new history
        adapter2 = GitFilterRepoAdapter(str(temp_git_repo))
        files_after = adapter2.list_all_files_in_history()
        assert "config.json" not in files_after

    def test_change_author_actually_changes(self, temp_git_repo):
        adapter = GitFilterRepoAdapter(str(temp_git_repo))

        result = adapter.change_author(
            old_email="test@example.com",
            new_name="Changed User",
            new_email="changed@example.com",
            dry_run=False,
            force=True,
        )
        assert result.success is True

        adapter2 = GitFilterRepoAdapter(str(temp_git_repo))
        commits = adapter2.get_commits()
        for c in commits:
            assert c.author_email == "changed@example.com"
            assert c.author_name == "Changed User"

    def test_rewrite_messages_actually_changes(self, temp_git_repo):
        adapter = GitFilterRepoAdapter(str(temp_git_repo))

        def callback(msg, _hash):
            return f"[PREFIXED] {msg}"

        result = adapter.rewrite_commit_messages(callback, dry_run=False, force=True)
        assert result.success is True
        assert result.commits_rewritten == 3

        adapter2 = GitFilterRepoAdapter(str(temp_git_repo))
        commits = adapter2.get_commits()
        for c in commits:
            assert c.message.startswith("[PREFIXED]")

    def test_replace_text_actually_replaces(self, temp_git_repo):
        adapter = GitFilterRepoAdapter(str(temp_git_repo))

        result = adapter.replace_text_in_history(
            old_text="hello", new_text="world", dry_run=False, force=True,
        )
        assert result.success is True

        # Verify text is replaced in working tree
        content = (temp_git_repo / "main.py").read_text()
        assert "world" in content
        assert "hello" not in content

    def test_restore_backup_works(self, temp_git_repo):
        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        backup_branch = adapter.create_backup()
        commits_before = adapter.get_commits()

        # Make a non-filter-repo change (add a new commit)
        (temp_git_repo / "extra.txt").write_text("extra")
        subprocess.run(["git", "add", "."], cwd=temp_git_repo, capture_output=True)
        subprocess.run(["git", "commit", "-m", "extra"], cwd=temp_git_repo, capture_output=True)

        adapter2 = GitFilterRepoAdapter(str(temp_git_repo))
        assert len(adapter2.get_commits()) == 4  # 3 original + 1 extra

        # Restore to backup (which is at 3 commits)
        result = adapter2.restore_backup(backup_branch)
        assert result.success is True

        adapter3 = GitFilterRepoAdapter(str(temp_git_repo))
        commits_after = adapter3.get_commits()
        assert len(commits_after) == 3
        assert commits_after[0].hash == commits_before[0].hash

    def test_scan_secrets_with_real_secret(self, temp_git_repo):
        # Write a fake secret and commit it
        (temp_git_repo / "secret.env").write_text("AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")
        subprocess.run(["git", "add", "."], cwd=temp_git_repo, capture_output=True)
        subprocess.run(["git", "commit", "-m", "add secret"], cwd=temp_git_repo, capture_output=True)

        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        scan_result = adapter.scan_secrets()
        # scan_secrets returns "secrets_found" and "sensitive_files" keys
        assert scan_result["secrets_found"] > 0 or scan_result["sensitive_files"] > 0


@requires_git_filter_repo
class TestEdgeCases:
    """Test edge cases with various repo states."""

    def test_empty_repo_get_commits(self, empty_git_repo):
        adapter = GitFilterRepoAdapter(str(empty_git_repo))
        commits = adapter.get_commits()
        assert commits == []

    def test_empty_repo_analyze_history(self, empty_git_repo):
        adapter = GitFilterRepoAdapter(str(empty_git_repo))
        analysis = adapter.analyze_history()
        assert analysis["total_commits"] == 0
        assert analysis["authors"] == {}

    def test_single_commit_squash_noop(self, single_commit_repo):
        adapter = GitFilterRepoAdapter(str(single_commit_repo))
        commits = adapter.get_commits()
        # Squashing from the only commit should produce no range
        result = adapter.squash_commits(start_commit=commits[0].hash, dry_run=True)
        assert result.commits_processed == 0

    def test_single_commit_change_dates(self, single_commit_repo):
        adapter = GitFilterRepoAdapter(str(single_commit_repo))
        result = adapter.change_commit_dates(time_range="evening", dry_run=True)
        assert result.success is True
        assert result.commits_rewritten == 1

    def test_unicode_commit_message_preserved(self, unicode_git_repo):
        adapter = GitFilterRepoAdapter(str(unicode_git_repo))
        commits = adapter.get_commits()
        assert len(commits) == 1
        assert "\u2728" in commits[0].message

    def test_unicode_rewrite_dry_run(self, unicode_git_repo):
        adapter = GitFilterRepoAdapter(str(unicode_git_repo))

        def callback(msg, _hash):
            return f"[prefix] {msg}"

        result = adapter.rewrite_commit_messages(callback, dry_run=True)
        assert result.success is True
        assert result.commits_rewritten == 1

    def test_nonexistent_file_get_history(self, temp_git_repo):
        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        history = adapter.get_file_history("nonexistent.txt")
        assert history == []

    def test_remove_nonexistent_file_dry_run(self, temp_git_repo):
        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        result = adapter.remove_files(["nonexistent.txt"], dry_run=True)
        assert result.success is True
        assert result.commits_rewritten == 0
