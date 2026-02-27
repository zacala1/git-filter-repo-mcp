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

    def test_parse_time_range_out_of_bounds_hour(self):
        adapter = _make_mock_adapter()
        result = adapter._parse_time_range("25:00-30:00")
        assert isinstance(result, FilterResult)
        assert result.success is False
        assert "out of range" in result.message

    def test_parse_time_range_out_of_bounds_minute(self):
        adapter = _make_mock_adapter()
        result = adapter._parse_time_range("10:70-22:00")
        assert isinstance(result, FilterResult)
        assert result.success is False
        assert "out of range" in result.message


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

        # Verify backup branch is preserved (not deleted)
        branch_result = subprocess.run(
            ["git", "branch", "--list", backup_branch],
            cwd=temp_git_repo, capture_output=True, text=True,
        )
        assert backup_branch in branch_result.stdout

    def test_rewrite_single_commit_message(self, temp_git_repo):
        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        commits = adapter.get_commits()
        target = commits[0]  # "Add config"

        result = adapter.rewrite_single_commit(
            commit_hash=target.hash,
            new_message="Updated config message",
            force=True,
        )
        assert result.success is True
        assert result.commits_rewritten == 1
        assert "message" in result.message

        adapter2 = GitFilterRepoAdapter(str(temp_git_repo))
        new_commits = adapter2.get_commits()
        assert new_commits[0].message == "Updated config message"
        # Other commits unchanged
        assert new_commits[1].message == "Add main.py"

    def test_rewrite_single_commit_author(self, temp_git_repo):
        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        commits = adapter.get_commits()
        target = commits[1]  # "Add main.py"

        result = adapter.rewrite_single_commit(
            commit_hash=target.hash,
            new_author_name="New Author",
            new_author_email="newauthor@test.com",
            force=True,
        )
        assert result.success is True
        assert "author" in result.message

        adapter2 = GitFilterRepoAdapter(str(temp_git_repo))
        new_commits = adapter2.get_commits()
        # Find the commit that was "Add main.py"
        rewritten = [c for c in new_commits if c.message == "Add main.py"][0]
        assert rewritten.author_name == "New Author"
        assert rewritten.author_email == "newauthor@test.com"

    def test_rewrite_single_commit_no_changes(self, temp_git_repo):
        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        commits = adapter.get_commits()

        result = adapter.rewrite_single_commit(commit_hash=commits[0].hash)
        assert result.success is False
        assert "No changes specified" in result.message

    def test_change_commit_dates_actually_changes(self, temp_git_repo):
        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        commits_before = adapter.get_commits()
        dates_before = [c.date for c in commits_before]

        result = adapter.change_commit_dates(
            time_range="night", dry_run=False, force=True,
        )
        assert result.success is True
        assert result.commits_rewritten == 3

        adapter2 = GitFilterRepoAdapter(str(temp_git_repo))
        commits_after = adapter2.get_commits()
        # Dates should have changed (exact values depend on randomization)
        dates_after = [c.date for c in commits_after]
        assert dates_before != dates_after

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


@requires_git_filter_repo
class TestFilterPaths:
    """Test filter_paths dry-run operations."""

    def test_filter_paths_include_dry_run(self, temp_git_repo):
        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        result = adapter.filter_paths(include_paths=["main.py"], dry_run=True)
        assert result.success is True
        assert result.dry_run is True

    def test_filter_paths_exclude_dry_run(self, temp_git_repo):
        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        result = adapter.filter_paths(exclude_paths=["config.json"], dry_run=True)
        assert result.success is True
        assert result.dry_run is True

    def test_filter_paths_no_paths_rejected(self, temp_git_repo):
        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        result = adapter.filter_paths(dry_run=True)
        assert result.success is False
        assert "No paths specified" in result.message

    def test_filter_paths_include_and_exclude_rejected(self, temp_git_repo):
        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        result = adapter.filter_paths(
            include_paths=["src/"], exclude_paths=["tests/"], dry_run=True,
        )
        assert result.success is False
        assert "Cannot use include_paths and exclude_paths together" in result.message

    def test_filter_paths_include_actually_filters(self, temp_git_repo):
        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        result = adapter.filter_paths(
            include_paths=["README.md"], dry_run=False, force=True,
        )
        assert result.success is True

        adapter2 = GitFilterRepoAdapter(str(temp_git_repo))
        files = adapter2.list_all_files_in_history()
        assert "README.md" in files
        assert "config.json" not in files
        assert "main.py" not in files


@requires_git_filter_repo
class TestAdditionalEdgeCases:
    """Additional edge case coverage."""

    def test_list_all_files_with_limit(self, temp_git_repo):
        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        files = adapter.list_all_files_in_history(limit=2)
        assert len(files) == 2

    def test_get_commits_with_branch(self, temp_git_repo):
        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        commits = adapter.get_commits("HEAD")
        assert len(commits) == 3

    def test_analyze_history_has_all_fields(self, temp_git_repo):
        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        analysis = adapter.analyze_history()
        assert "total_commits" in analysis
        assert "authors" in analysis
        assert "commits" in analysis
        assert analysis["total_commits"] == 3
        assert len(analysis["commits"]) == 3

    def test_get_commit_files_returns_list(self, temp_git_repo):
        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        commits = adapter.get_commits()
        files = adapter.get_commit_files(commits[2].hash)  # Initial commit
        assert "README.md" in files

    def test_empty_repo_list_files(self, empty_git_repo):
        adapter = GitFilterRepoAdapter(str(empty_git_repo))
        files = adapter.list_all_files_in_history()
        assert files == []

    def test_squash_commits_dry_run_count(self, temp_git_repo):
        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        commits = adapter.get_commits()
        start = commits[2].hash  # Initial commit
        result = adapter.squash_commits(start_commit=start, dry_run=True)
        assert result.success is True
        assert result.dry_run is True
        assert result.commits_processed == 2

    def test_squash_invalid_range(self, temp_git_repo):
        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        result = adapter.squash_commits(start_commit="0000000000000000000000000000000000000000", dry_run=True)
        assert result.success is False
        assert "Invalid commit range" in result.message


class TestCommitHashValidation:
    """Test that commit hash injection is prevented."""

    def test_valid_hex_hash_accepted(self):
        adapter = _make_mock_adapter()
        # Should not raise
        adapter._validate_commit_hash("abc123def456")
        adapter._validate_commit_hash("0" * 40)
        adapter._validate_commit_hash("ABCDEF")

    def test_injection_attempt_rejected(self):
        adapter = _make_mock_adapter()
        with pytest.raises(ValueError, match="Invalid commit hash"):
            adapter._validate_commit_hash('"; import os; os.system("rm -rf /"); "')

    def test_spaces_rejected(self):
        adapter = _make_mock_adapter()
        with pytest.raises(ValueError, match="Invalid commit hash"):
            adapter._validate_commit_hash("abc 123")

    def test_empty_string_rejected(self):
        adapter = _make_mock_adapter()
        with pytest.raises(ValueError, match="Invalid commit hash"):
            adapter._validate_commit_hash("")


@requires_git_filter_repo
class TestMultilineCommitMessage:
    """Test that multi-line commit messages are fully preserved."""

    def test_multiline_message_preserved(self, multiline_commit_repo):
        adapter = GitFilterRepoAdapter(str(multiline_commit_repo))
        commits = adapter.get_commits()
        assert len(commits) == 1
        assert "feat: add app" in commits[0].message
        assert "main application" in commits[0].message
        assert "multiple lines" in commits[0].message

    def test_multiline_rewrite_dry_run(self, multiline_commit_repo):
        adapter = GitFilterRepoAdapter(str(multiline_commit_repo))

        def callback(msg, _hash):
            return f"[REWRITTEN] {msg}"

        result = adapter.rewrite_commit_messages(callback, dry_run=True)
        assert result.success is True
        assert result.commits_rewritten == 1


class TestPathValidation:
    """Test path validation in _validate_repo."""

    def test_relative_path_rejected(self):
        with pytest.raises(ValueError, match="must be absolute"):
            GitFilterRepoAdapter("relative/path/repo")

    def test_nonexistent_path_rejected(self):
        import platform
        path = "C:\\nonexistent\\path\\to\\repo" if platform.system() == "Windows" else "/nonexistent/path/to/repo"
        with pytest.raises(ValueError, match="does not exist"):
            GitFilterRepoAdapter(path)


@requires_git_filter_repo
class TestMaxCountZero:
    """Test that max_count=0 returns zero commits, not all."""

    def test_max_count_zero_returns_empty(self, temp_git_repo):
        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        commits = adapter.get_commits(max_count=0)
        assert commits == []

    def test_max_count_none_returns_all(self, temp_git_repo):
        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        commits = adapter.get_commits(max_count=None)
        assert len(commits) == 3


@requires_git_filter_repo
class TestCollectCommitFiles:
    """Test bulk collect_commit_files method."""

    def test_bulk_collect_returns_files_per_commit(self, temp_git_repo):
        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        commits = adapter.get_commits()
        result = adapter.collect_commit_files(commits, "HEAD", len(commits))

        assert isinstance(result, dict)
        # Should have entries for each commit hash
        for commit in commits:
            assert commit.hash in result
            assert isinstance(result[commit.hash], list)

        # "Add config" commit should include config.json
        config_commit = [c for c in commits if c.message == "Add config"][0]
        assert "config.json" in result[config_commit.hash]

    def test_bulk_collect_matches_individual(self, temp_git_repo):
        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        commits = adapter.get_commits()
        bulk = adapter.collect_commit_files(commits, "HEAD", len(commits))

        # Compare with individual get_commit_files calls
        for commit in commits:
            individual = adapter.get_commit_files(commit.hash)
            assert sorted(bulk.get(commit.hash, [])) == sorted(individual)


@requires_git_filter_repo
class TestRemoveLargeFiles:
    """Test remove_large_files dry-run and real execution."""

    def test_dry_run_no_large_files(self, temp_git_repo):
        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        result = adapter.remove_large_files(size_threshold_mb=10.0, dry_run=True)
        assert result.success is True
        assert result.dry_run is True
        assert result.files_affected == []

    def test_dry_run_finds_large_file(self, temp_git_repo):
        # Write a file > 1 byte to detect with a very low threshold
        (temp_git_repo / "big.bin").write_bytes(b"x" * 2048)
        subprocess.run(["git", "add", "."], cwd=temp_git_repo, capture_output=True)
        subprocess.run(["git", "commit", "-m", "add big file"], cwd=temp_git_repo, capture_output=True)

        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        # Threshold of 0.001 MB = ~1KB, so 2KB file should be found
        result = adapter.remove_large_files(size_threshold_mb=0.001, dry_run=True)
        assert result.success is True
        assert result.dry_run is True
        assert len(result.files_affected) >= 1
        assert any("big.bin" in f for f in result.files_affected)

    def test_actual_removal(self, temp_git_repo):
        (temp_git_repo / "huge.bin").write_bytes(b"y" * 4096)
        subprocess.run(["git", "add", "."], cwd=temp_git_repo, capture_output=True)
        subprocess.run(["git", "commit", "-m", "add huge"], cwd=temp_git_repo, capture_output=True)

        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        result = adapter.remove_large_files(size_threshold_mb=0.001, dry_run=False, force=True)
        assert result.success is True

        adapter2 = GitFilterRepoAdapter(str(temp_git_repo))
        files = adapter2.list_all_files_in_history()
        assert "huge.bin" not in files

    def test_high_threshold_finds_nothing(self, temp_git_repo):
        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        # 100MB threshold - nothing in test repo that big
        result = adapter.remove_large_files(size_threshold_mb=100.0, dry_run=False)
        assert result.success is True
        assert "No large files found" in result.message


@requires_git_filter_repo
class TestGetCommitDiff:
    """Test get_commit_diff method."""

    def test_returns_diff_stat(self, temp_git_repo):
        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        commits = adapter.get_commits()
        diff = adapter.get_commit_diff(commits[0].hash)
        assert isinstance(diff, str)
        assert "config.json" in diff

    def test_initial_commit_diff(self, temp_git_repo):
        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        commits = adapter.get_commits()
        # Initial commit
        diff = adapter.get_commit_diff(commits[-1].hash)
        assert "README.md" in diff
