"""Adapter tests."""

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from git_filter_repo_mcp.adapter import CommitInfo, FilterResult, GitFilterRepoAdapter
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


class TestMailmapInjection:
    """Test that mailmap injection is prevented."""

    @requires_git_filter_repo
    def test_newline_in_name_rejected(self, temp_git_repo):
        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        result = adapter.change_author(
            old_email="test@example.com",
            new_name="Evil\nName",
            new_email="evil@example.com",
            dry_run=False,
        )
        assert result.success is False
        assert "Invalid characters" in result.message

    @requires_git_filter_repo
    def test_angle_bracket_in_email_rejected(self, temp_git_repo):
        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        result = adapter.change_author(
            old_email="test@example.com",
            new_name="Evil",
            new_email="evil@example.com> <extra@inject.com",
            dry_run=False,
        )
        assert result.success is False
        assert "Invalid characters" in result.message

    @requires_git_filter_repo
    def test_sanitization_runs_in_dry_run(self, temp_git_repo):
        """Sanitization must run even in dry_run mode to catch invalid input early."""
        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        result = adapter.change_author(
            old_email="test@example.com",
            new_name="Evil\nName",
            new_email="evil@example.com",
            dry_run=True,
        )
        assert result.success is False
        assert "Invalid characters" in result.message

    @requires_git_filter_repo
    def test_clean_inputs_accepted(self, temp_git_repo):
        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        result = adapter.change_author(
            old_email="test@example.com",
            new_name="Clean Name",
            new_email="clean@example.com",
            dry_run=True,
        )
        assert result.success is True


class TestPathInjection:
    """Test that path option injection is prevented."""

    def test_dash_path_rejected(self):
        adapter = _make_mock_adapter()
        with pytest.raises(ValueError, match="must not start with"):
            adapter._validate_paths(["--commit-callback"])

    def test_empty_path_rejected(self):
        adapter = _make_mock_adapter()
        with pytest.raises(ValueError, match="empty string"):
            adapter._validate_paths(["valid.py", ""])

    def test_normal_paths_accepted(self):
        adapter = _make_mock_adapter()
        adapter._validate_paths(["src/main.py", "README.md", "docs/"])

    @requires_git_filter_repo
    def test_remove_files_rejects_option_injection(self, temp_git_repo):
        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        with pytest.raises(ValueError, match="must not start with"):
            adapter.remove_files(["--commit-callback", "evil code"], dry_run=True)

    @requires_git_filter_repo
    def test_filter_paths_rejects_option_injection(self, temp_git_repo):
        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        with pytest.raises(ValueError, match="must not start with"):
            adapter.filter_paths(include_paths=["--force"], dry_run=True)


@requires_git_filter_repo
class TestSquashEndCommitValidation:
    """Test squash_commits validates end_commit == HEAD."""

    def test_squash_non_head_end_commit_rejected(self, temp_git_repo):
        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        commits = adapter.get_commits()
        # start=initial, end=middle (not HEAD)
        result = adapter.squash_commits(
            start_commit=commits[2].hash,
            end_commit=commits[1].hash,
            dry_run=False,
        )
        assert result.success is False
        assert "must be HEAD" in result.message

    def test_squash_head_end_commit_accepted_dry_run(self, temp_git_repo):
        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        commits = adapter.get_commits()
        result = adapter.squash_commits(
            start_commit=commits[2].hash,
            end_commit="HEAD",
            dry_run=True,
        )
        assert result.success is True

    def test_squash_zero_range_message(self, temp_git_repo):
        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        commits = adapter.get_commits()
        result = adapter.squash_commits(
            start_commit=commits[0].hash,
            end_commit="HEAD",
            dry_run=True,
        )
        assert result.success is False
        assert "No commits in range" in result.message


@requires_git_filter_repo
class TestRemoveLargeFilesEmptyRepo:
    """Test remove_large_files on empty repo."""

    def test_empty_repo_returns_gracefully(self, empty_git_repo):
        adapter = GitFilterRepoAdapter(str(empty_git_repo))
        result = adapter.remove_large_files(dry_run=True)
        assert result.success is True


class TestBackupTimestampUniqueness:
    """Test that backup branch names include microseconds."""

    @requires_git_filter_repo
    def test_backup_includes_microseconds(self, temp_git_repo):
        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        backup = adapter.create_backup()
        # Format: backup_YYYYMMDD_HHMMSS_ffffff
        parts = backup.split("_")
        assert len(parts) == 4  # backup, date, time, microseconds
        assert len(parts[3]) == 6  # microseconds


class TestReplaceTextNewlineRejection:
    """Test that newlines in old_text/new_text are rejected."""

    @requires_git_filter_repo
    def test_newline_in_old_text_rejected(self, temp_git_repo):
        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        result = adapter.replace_text_in_history(
            old_text="hello\nworld", new_text="replaced", dry_run=True,
        )
        assert result.success is False
        assert "newlines" in result.message.lower()

    @requires_git_filter_repo
    def test_newline_in_new_text_rejected(self, temp_git_repo):
        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        result = adapter.replace_text_in_history(
            old_text="hello", new_text="re\nplaced", dry_run=True,
        )
        assert result.success is False
        assert "newlines" in result.message.lower()

    @requires_git_filter_repo
    def test_clean_text_accepted(self, temp_git_repo):
        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        result = adapter.replace_text_in_history(
            old_text="hello", new_text="world", dry_run=True,
        )
        assert result.success is True

    @requires_git_filter_repo
    def test_replace_text_with_separator_escaped(self, temp_git_repo):
        """Test that ==> in new_text is escaped properly."""
        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        result = adapter.replace_text_in_history(
            old_text="hello", new_text="a==>b", dry_run=True,
        )
        assert result.success is True


class TestMailmapTabInjection:
    """Test that tab characters in mailmap input are rejected."""

    @requires_git_filter_repo
    def test_tab_in_name_rejected(self, temp_git_repo):
        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        result = adapter.change_author(
            old_email="test@example.com",
            new_name="Evil\tName",
            new_email="evil@example.com",
            dry_run=True,
        )
        assert result.success is False
        assert "Invalid characters" in result.message


class TestRestoreBackupValidation:
    """Test that restore_backup validates branch name prefix."""

    @requires_git_filter_repo
    def test_non_backup_branch_rejected(self, temp_git_repo):
        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        result = adapter.restore_backup("main")
        assert result.success is False
        assert "Invalid backup branch" in result.message

    @requires_git_filter_repo
    def test_backup_branch_accepted(self, temp_git_repo):
        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        backup = adapter.create_backup()
        result = adapter.restore_backup(backup)
        assert result.success is True


class TestGenerateDateMappings:
    """Test _generate_date_mappings directly for edge cases."""

    def test_preserve_order_weekend_only(self):
        """Dates adjusted by preserve_order should still have valid times."""
        import datetime

        adapter = _make_mock_adapter()
        # Create commits with unique hashes, close together so preserve_order kicks in
        commits = [
            CommitInfo(
                hash=f"{i:0>40x}",
                author_name="Test",
                author_email="t@e.com",
                committer_name="Test",
                committer_email="t@e.com",
                message=f"commit {i}",
                date="2024-01-06T10:00:00+00:00",  # Saturday
            )
            for i in range(5)
        ]
        # Use evening time range (19-23) on weekends only
        result = adapter._generate_date_mappings(
            commits,
            start_hour=19, start_min=0, end_hour=23, end_min=0,
            weekend_only=True, preserve_order=True, start_date=None,
        )
        assert isinstance(result, dict)
        assert len(result) == 5

        prev_ts = None
        for _hash, (ts, _tz) in result.items():
            dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
            # Should be on a weekend
            assert dt.weekday() >= 5, f"Expected weekend, got weekday {dt.weekday()}"
            # Should maintain order
            if prev_ts is not None:
                assert ts > prev_ts, "Order not preserved"
            prev_ts = ts

    def test_empty_commits_returns_empty(self):
        adapter = _make_mock_adapter()
        result = adapter._generate_date_mappings(
            [], 19, 0, 23, 0, False, True, None,
        )
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_invalid_start_date_returns_error(self):
        adapter = _make_mock_adapter()
        commits = [
            CommitInfo("a" * 40, "T", "t@e.com", "T", "t@e.com", "msg", "2024-01-01T00:00:00+00:00"),
        ]
        result = adapter._generate_date_mappings(
            commits, 19, 0, 23, 0, False, True, "not-a-date",
        )
        assert isinstance(result, FilterResult)
        assert result.success is False


class TestRestoreBackupBranchValidation:
    """Test restore_backup with stricter regex validation."""

    @requires_git_filter_repo
    def test_injection_attempt_rejected(self, temp_git_repo):
        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        result = adapter.restore_backup("backup_; rm -rf /")
        assert result.success is False
        assert "Invalid backup branch" in result.message

    @requires_git_filter_repo
    def test_nonexistent_backup_rejected(self, temp_git_repo):
        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        result = adapter.restore_backup("backup_99991231_235959_000000")
        assert result.success is False

    @requires_git_filter_repo
    def test_hyphens_rejected(self, temp_git_repo):
        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        result = adapter.restore_backup("backup_2024-01-01")
        assert result.success is False


class TestAnalyzeHistoryResponse:
    """Test that analyze_history includes all expected fields."""

    @requires_git_filter_repo
    def test_total_authors_present(self, temp_git_repo):
        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        result = adapter.analyze_history()
        assert "total_authors" in result
        assert result["total_authors"] == 1

    @requires_git_filter_repo
    def test_total_commits_matches(self, temp_git_repo):
        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        result = adapter.analyze_history()
        assert result["total_commits"] == 3


class TestScanSecretsSensitiveDedup:
    """Test that sensitive files are not duplicated across commits."""

    @requires_git_filter_repo
    def test_sensitive_file_in_multiple_commits_deduplicated(self, temp_git_repo):
        import subprocess
        # Create .env in two commits
        (temp_git_repo / ".env").write_text("SECRET=abc")
        subprocess.run(["git", "add", "."], cwd=temp_git_repo, capture_output=True)
        subprocess.run(["git", "commit", "-m", "add env"], cwd=temp_git_repo, capture_output=True)
        (temp_git_repo / ".env").write_text("SECRET=def")
        subprocess.run(["git", "add", "."], cwd=temp_git_repo, capture_output=True)
        subprocess.run(["git", "commit", "-m", "update env"], cwd=temp_git_repo, capture_output=True)

        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        result = adapter.scan_secrets()
        # .env should appear only once in sensitive_file_list
        env_entries = [f for f in result["sensitive_file_list"] if f["file"] == ".env"]
        assert len(env_entries) == 1


class TestValidateRef:
    """Test _validate_ref dash-prefix rejection."""

    def test_normal_ref_accepted(self):
        adapter = _make_mock_adapter()
        adapter._validate_ref("HEAD")
        adapter._validate_ref("abc123")
        adapter._validate_ref("main")

    def test_dash_ref_rejected(self):
        adapter = _make_mock_adapter()
        with pytest.raises(ValueError, match="must not start with"):
            adapter._validate_ref("--exec=evil")

    def test_single_dash_rejected(self):
        adapter = _make_mock_adapter()
        with pytest.raises(ValueError, match="must not start with"):
            adapter._validate_ref("-n5")


class TestReplaceTextEmptyOldText:
    """Test that empty old_text is rejected."""

    @requires_git_filter_repo
    def test_empty_old_text_rejected(self, temp_git_repo):
        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        result = adapter.replace_text_in_history(old_text="", new_text="x", dry_run=True)
        assert result.success is False
        assert "empty" in result.message.lower()


@requires_git_filter_repo
class TestSquashCommitsRealExecution:
    """Test squash_commits actual execution (not dry_run)."""

    def test_squash_actually_squashes(self, temp_git_repo):
        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        commits = adapter.get_commits()
        assert len(commits) == 3
        # commits[2] = Initial commit (oldest), commits[0] = Add config (newest/HEAD)
        # squash_commits(start=exclusive, end=inclusive) so commits AFTER start get squashed
        start = commits[2].hash  # Initial commit is excluded

        result = adapter.squash_commits(start_commit=start, end_commit="HEAD", new_message="squashed two", dry_run=False)
        assert result.success is True
        assert result.commits_rewritten == 1
        assert result.commits_processed == 2  # 2 commits after initial

        adapter2 = GitFilterRepoAdapter(str(temp_git_repo))
        commits_after = adapter2.get_commits()
        # Initial commit + 1 squashed commit = 2 total
        assert len(commits_after) == 2
        assert commits_after[0].message == "squashed two"
        assert commits_after[1].message == "Initial commit"

    def test_squash_without_custom_message(self, temp_git_repo):
        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        commits = adapter.get_commits()
        start = commits[2].hash  # Initial commit (exclusive)

        result = adapter.squash_commits(start_commit=start, end_commit="HEAD", dry_run=False)
        assert result.success is True

        adapter2 = GitFilterRepoAdapter(str(temp_git_repo))
        commits_after = adapter2.get_commits()
        assert len(commits_after) == 2  # Initial + squashed
        assert "Squashed commits:" in commits_after[0].message
        assert "Add config" in commits_after[0].message


@requires_git_filter_repo
class TestRewriteMessagesDuplicateMessages:
    """Test that rewrite_commit_messages handles duplicate messages via hash-based lookup."""

    def test_duplicate_messages_both_rewritten(self, temp_git_repo):
        """Two commits with same message should both get rewritten."""
        # Add another commit with same "Add config" message
        (temp_git_repo / "config2.json").write_text('{"key2": "val2"}')
        subprocess.run(["git", "add", "."], cwd=temp_git_repo, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Add config"], cwd=temp_git_repo, capture_output=True)

        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        commits = adapter.get_commits()
        dup_msgs = [c for c in commits if c.message == "Add config"]
        assert len(dup_msgs) == 2, "Should have 2 commits with same message"

        def callback(msg, _hash):
            if msg == "Add config":
                return "chore: add configuration"
            return msg

        result = adapter.rewrite_commit_messages(callback, dry_run=False, force=True)
        assert result.success is True

        adapter2 = GitFilterRepoAdapter(str(temp_git_repo))
        commits_after = adapter2.get_commits()
        rewritten = [c for c in commits_after if c.message == "chore: add configuration"]
        assert len(rewritten) == 2, f"Both duplicates should be rewritten, got {[c.message for c in commits_after]}"


@requires_git_filter_repo
class TestChangeDatesNightRange:
    """Test change_commit_dates with night wrap-around range (22:00-02:00)."""

    def test_night_range_produces_valid_times(self, temp_git_repo):
        import datetime

        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        result = adapter.change_commit_dates(time_range="night", dry_run=False, force=True)
        assert result.success is True

        adapter2 = GitFilterRepoAdapter(str(temp_git_repo))
        commits = adapter2.get_commits()
        for c in commits:
            dt = datetime.datetime.fromisoformat(c.date)
            hour = dt.hour
            # Night range: 22:00-02:00 means hour in {22, 23, 0, 1, 2}
            assert hour >= 22 or hour <= 2, f"Hour {hour} outside night range for commit {c.hash[:8]}"


@requires_git_filter_repo
class TestScanSecretsPatternMatching:
    """Test scan_secrets actually detects various secret patterns."""

    def test_detects_aws_access_key(self, temp_git_repo):
        (temp_git_repo / "creds.txt").write_text("key = AKIAIOSFODNN7EXAMPLE")
        subprocess.run(["git", "add", "."], cwd=temp_git_repo, capture_output=True)
        subprocess.run(["git", "commit", "-m", "add creds"], cwd=temp_git_repo, capture_output=True)

        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        result = adapter.scan_secrets()
        assert result["secrets_found"] >= 1
        aws_findings = [f for f in result["findings"] if f["type"] == "aws_access_key"]
        assert len(aws_findings) >= 1

    def test_detects_github_token(self, temp_git_repo):
        (temp_git_repo / "token.txt").write_text("ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmn")
        subprocess.run(["git", "add", "."], cwd=temp_git_repo, capture_output=True)
        subprocess.run(["git", "commit", "-m", "add token"], cwd=temp_git_repo, capture_output=True)

        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        result = adapter.scan_secrets()
        gh_findings = [f for f in result["findings"] if f["type"] == "github_token"]
        assert len(gh_findings) >= 1

    def test_detects_private_key(self, temp_git_repo):
        (temp_git_repo / "key.pem").write_text("-----BEGIN RSA PRIVATE KEY-----\nfake\n-----END RSA PRIVATE KEY-----")
        subprocess.run(["git", "add", "."], cwd=temp_git_repo, capture_output=True)
        subprocess.run(["git", "commit", "-m", "add key"], cwd=temp_git_repo, capture_output=True)

        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        result = adapter.scan_secrets()
        pk_findings = [f for f in result["findings"] if f["type"] == "private_key"]
        assert len(pk_findings) >= 1

    def test_no_false_positive_on_clean_repo(self, single_commit_repo):
        adapter = GitFilterRepoAdapter(str(single_commit_repo))
        result = adapter.scan_secrets()
        assert result["secrets_found"] == 0


class TestChangeAuthorEmptyFields:
    """Test that empty author fields are rejected."""

    @requires_git_filter_repo
    def test_empty_name_rejected(self, temp_git_repo):
        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        result = adapter.change_author("test@example.com", "", "new@example.com", dry_run=True)
        assert result.success is False
        assert "empty" in result.message.lower()

    @requires_git_filter_repo
    def test_whitespace_only_name_rejected(self, temp_git_repo):
        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        result = adapter.change_author("test@example.com", "  ", "new@example.com", dry_run=True)
        assert result.success is False

    @requires_git_filter_repo
    def test_empty_old_email_rejected(self, temp_git_repo):
        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        result = adapter.change_author("", "Name", "new@example.com", dry_run=True)
        assert result.success is False


class TestChangeAuthorNoMatch:
    """Test that change_author shows helpful message when email not found."""

    @requires_git_filter_repo
    def test_no_match_shows_existing_emails(self, temp_git_repo):
        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        result = adapter.change_author("nonexistent@email.com", "Name", "new@e.com", dry_run=True)
        assert result.success is True
        assert result.commits_rewritten == 0
        assert "nonexistent@email.com" in result.message
        assert "test@example.com" in result.message  # Should show existing email


class TestGetCommitsDashInjection:
    """Test that get_commits rejects dash-prefixed branch names."""

    @requires_git_filter_repo
    def test_dash_branch_rejected(self, temp_git_repo):
        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        with pytest.raises(ValueError, match="must not start with"):
            adapter.get_commits(branch="--exec=evil")

    @requires_git_filter_repo
    def test_normal_branch_accepted(self, temp_git_repo):
        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        commits = adapter.get_commits(branch="HEAD")
        assert len(commits) == 3


class TestRemoveFilesEmptyPaths:
    """Test that remove_files rejects empty paths list."""

    @requires_git_filter_repo
    def test_empty_list_rejected(self, temp_git_repo):
        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        result = adapter.remove_files([], dry_run=True)
        assert result.success is False
        assert "No file paths" in result.message


class TestAnalyzeHistoryTruncation:
    """Test that long commit messages get '...' indicator."""

    @requires_git_filter_repo
    def test_long_message_truncated_with_ellipsis(self, temp_git_repo):
        long_msg = "feat: " + "a" * 100
        (temp_git_repo / "long.txt").write_text("content")
        subprocess.run(["git", "add", "."], cwd=temp_git_repo, capture_output=True)
        subprocess.run(["git", "commit", "-m", long_msg], cwd=temp_git_repo, capture_output=True)

        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        result = adapter.analyze_history()
        latest = result["commits"][0]
        assert latest["message"].endswith("...")
        assert len(latest["message"]) == 83  # 80 + "..."

    @requires_git_filter_repo
    def test_short_message_no_ellipsis(self, temp_git_repo):
        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        result = adapter.analyze_history()
        for c in result["commits"]:
            if len(c["message"]) < 80:
                assert not c["message"].endswith("...")


class TestScanSecretsMessageField:
    """Test that scan_secrets response includes a summary message."""

    @requires_git_filter_repo
    def test_clean_repo_has_no_secrets_message(self, single_commit_repo):
        adapter = GitFilterRepoAdapter(str(single_commit_repo))
        result = adapter.scan_secrets()
        assert "message" in result
        assert "No secrets" in result["message"]

    @requires_git_filter_repo
    def test_repo_with_secrets_has_count_message(self, temp_git_repo):
        (temp_git_repo / "key.txt").write_text("AKIAIOSFODNN7EXAMPLE")
        subprocess.run(["git", "add", "."], cwd=temp_git_repo, capture_output=True)
        subprocess.run(["git", "commit", "-m", "add key"], cwd=temp_git_repo, capture_output=True)

        adapter = GitFilterRepoAdapter(str(temp_git_repo))
        result = adapter.scan_secrets()
        assert "message" in result
        assert "secret" in result["message"].lower()
