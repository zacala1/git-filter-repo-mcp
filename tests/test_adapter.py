"""Tests for ``GitFilterRepoAdapter``.

Organisation:

- **Unit slice** (top of file): pure-Python tests with no real-git fixtures.
  Path normalisation, static validators, time-range parsing, and the
  ``_generate_date_mappings`` algorithm all run here.
- **Integration slice** (bottom of file): tests using ``temp_git_repo`` /
  ``adapter`` fixtures — these spawn real git/git-filter-repo subprocesses
  and are auto-tagged ``integration`` by ``conftest.pytest_collection_modifyitems``
  so ``pytest -m "not integration"`` runs only the fast unit slice.

A test belongs in the unit slice when it can be expressed purely with
``_make_mock_adapter()`` or static methods on the class. Anything that
needs a repository must be integration.
"""

import datetime
import platform
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from git_filter_repo_mcp.adapter import (
    MAX_FILES_TO_SCAN,
    MAX_FINDINGS_LIMIT,
    CommitInfo,
    FilterResult,
    GitFilterRepoAdapter,
)
from tests.conftest import add_commit, requires_git_filter_repo


# Apply ``requires_git_filter_repo`` to every integration test in this module.
# Integration classes opt in by listing this as a class-level pytest marker
# (set by inheriting from ``_IntegrationBase``).


def _make_mock_adapter() -> GitFilterRepoAdapter:
    """Build a ``GitFilterRepoAdapter`` without running ``__init__``.

    Used by unit tests that exercise static/instance methods without
    needing a real git repo on disk.
    """
    obj = object.__new__(GitFilterRepoAdapter)
    obj.repo_path = Path("/fake/repo")  # type: ignore[attr-defined]
    return obj


def _commit_info(hash_int: int = 0, date: str = "2024-01-01T10:00:00+00:00") -> CommitInfo:
    """Concise CommitInfo factory for unit tests."""
    return CommitInfo(
        hash=f"{hash_int:0>40x}",
        author_name="Test", author_email="t@e.com",
        committer_name="Test", committer_email="t@e.com",
        message=f"commit {hash_int}",
        date=date,
    )


# =========================================================================
# Unit slice — no fixtures, no subprocess
# =========================================================================


class TestPathNormalization:
    """``_normalize_path`` is a pure function; mock platform.system."""

    @pytest.mark.parametrize(
        "system,input_path,expected",
        [
            # Linux-style paths preserved when system is Windows (e.g. Docker/WSL bind mounts)
            ("Windows", "/root/test-repo", "/root/test-repo"),
            ("Windows", "/home/user/repo", "/home/user/repo"),
            # Git Bash style /c/Users/... → C:\Users\...
            ("Windows", "/c/Users/test", "C:\\Users\\test"),
            ("Windows", "/d/Projects/repo", "D:\\Projects\\repo"),
            # Windows path with forward slashes
            ("Windows", "C:/Users/test", "C:\\Users\\test"),
            # WSL paths preserved
            ("Windows", "//wsl$/Ubuntu/home/user", "//wsl$/Ubuntu/home/user"),
            # On Linux, every input is returned untouched
            ("Linux", "/c/Users/test", "/c/Users/test"),
            ("Linux", "/root/test", "/root/test"),
        ],
    )
    def test_normalise(self, system: str, input_path: str, expected: str) -> None:
        with patch("platform.system", return_value=system):
            assert GitFilterRepoAdapter._normalize_path(input_path) == expected


class TestStaticValidators:
    """Adapter-level input validators that don't touch the repo."""

    @pytest.mark.parametrize("hash_", ["abc123def456", "0" * 40, "ABCDEF"])
    def test_valid_commit_hash_accepted(self, hash_: str) -> None:
        _make_mock_adapter()._validate_commit_hash(hash_)  # must not raise

    @pytest.mark.parametrize(
        "bad_hash",
        [
            '"; import os; os.system("rm -rf /"); "',  # code injection attempt
            "abc 123",                                  # whitespace
            "",                                         # empty
            "abc",                                      # too short / overly broad
            "abc-123",                                  # punctuation
        ],
    )
    def test_invalid_commit_hash_rejected(self, bad_hash: str) -> None:
        with pytest.raises(ValueError, match="Invalid commit hash"):
            _make_mock_adapter()._validate_commit_hash(bad_hash)

    @pytest.mark.parametrize("ref", ["HEAD", "abc123", "main", "feature/x"])
    def test_valid_ref_accepted(self, ref: str) -> None:
        _make_mock_adapter()._validate_ref(ref)

    @pytest.mark.parametrize("bad_ref", ["--exec=evil", "-n5", "", "HEAD\nmain"])
    def test_dash_prefixed_ref_rejected(self, bad_ref: str) -> None:
        with pytest.raises(ValueError, match="Invalid ref"):
            _make_mock_adapter()._validate_ref(bad_ref)

    def test_valid_paths_accepted(self) -> None:
        _make_mock_adapter()._validate_paths(["src/main.py", "README.md", "docs/"])

    @pytest.mark.parametrize(
        "paths,error_fragment",
        [
            (["--commit-callback"], "must not start with"),
            (["valid.py", ""], "empty string"),
            (["bad\npath.txt"], "control characters"),
        ],
    )
    def test_invalid_paths_rejected(self, paths: list[str], error_fragment: str) -> None:
        with pytest.raises(ValueError, match=error_fragment):
            _make_mock_adapter()._validate_paths(paths)

    def test_collect_commit_files_rejects_bad_branch(self) -> None:
        with pytest.raises(ValueError, match="Invalid ref"):
            _make_mock_adapter().collect_commit_files([], "--all", 1)

    def test_collect_commit_files_rejects_negative_limit(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            _make_mock_adapter().collect_commit_files([], "HEAD", -1)


class TestRepoInitErrors:
    """``__init__`` failure cases that don't need a working repo."""

    def test_relative_path_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be absolute"):
            GitFilterRepoAdapter("relative/path/repo")

    def test_nonexistent_path_rejected(self) -> None:
        path = (
            "C:\\nonexistent\\path\\to\\repo" if platform.system() == "Windows"
            else "/nonexistent/path/to/repo"
        )
        with pytest.raises(ValueError, match="does not exist"):
            GitFilterRepoAdapter(path)

    def test_not_a_git_repository_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Not a git repository"):
                GitFilterRepoAdapter(tmpdir)


class TestParseTimeRange:
    """``_parse_time_range`` is a pure helper — exhaustively parametrised."""

    @pytest.mark.parametrize(
        "spec,expected",
        [
            ("evening", (19, 0, 23, 0)),
            ("night", (22, 0, 2, 0)),
            ("weekend", (10, 0, 22, 0)),
            ("random", (0, 0, 23, 59)),
            ("18:30-22:00", (18, 30, 22, 0)),
            ("9-17", (9, 0, 17, 0)),  # no minutes
        ],
    )
    def test_valid_spec(self, spec: str, expected: tuple[int, int, int, int]) -> None:
        assert _make_mock_adapter()._parse_time_range(spec) == expected

    @pytest.mark.parametrize(
        "bad_spec,error_fragment",
        [
            ("abc-def", "Invalid time range"),
            ("invalid", "Unknown time range"),
            ("25:00-30:00", "out of range"),
            ("10:70-22:00", "out of range"),
            ("18:30-18:00", "Invalid time range"),
        ],
    )
    def test_invalid_spec(self, bad_spec: str, error_fragment: str) -> None:
        result = _make_mock_adapter()._parse_time_range(bad_spec)
        assert isinstance(result, FilterResult)
        assert result.success is False
        assert error_fragment in result.message


class TestGenerateDateMappings:
    """``_generate_date_mappings`` invariants without touching git."""

    def test_empty_input_returns_empty_map(self) -> None:
        result = _make_mock_adapter()._generate_date_mappings(
            [], 19, 0, 23, 0, False, True, None,
        )
        assert result == {}

    def test_invalid_start_date_returns_filter_result(self) -> None:
        result = _make_mock_adapter()._generate_date_mappings(
            [_commit_info()], 19, 0, 23, 0, False, True, "not-a-date",
        )
        assert isinstance(result, FilterResult)
        assert result.success is False

    def test_preserve_order_with_weekend_only(self) -> None:
        """Times must be monotonically increasing AND fall on Sat/Sun."""
        commits = [_commit_info(i, "2024-01-06T10:00:00+00:00") for i in range(5)]
        result = _make_mock_adapter()._generate_date_mappings(
            commits, 19, 0, 23, 0, True, True, None,
        )
        assert isinstance(result, dict) and len(result) == 5

        prev_ts = None
        for ts, _tz in result.values():
            dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
            assert dt.weekday() >= 5, f"got weekday={dt.weekday()}"
            if prev_ts is not None:
                assert ts > prev_ts
            prev_ts = ts

    def test_wrap_around_range_preserves_order_and_range(self) -> None:
        """22:00-02:00 wrap-around: each timestamp must land in the window
        AND be strictly after the previous one (regression for the
        early-morning-on-current-day bug)."""
        commits = [_commit_info(i, "2024-01-01T10:00:00+00:00") for i in range(10)]
        result = _make_mock_adapter()._generate_date_mappings(
            commits, 22, 0, 2, 0, False, True, None,
        )
        assert isinstance(result, dict) and len(result) == 10

        prev_ts = None
        for ts, _tz in result.values():
            dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
            assert dt.hour >= 22 or dt.hour <= 2, f"got hour={dt.hour}"
            if prev_ts is not None:
                assert ts > prev_ts
            prev_ts = ts

    def test_preserve_order_advances_to_next_window_not_outside_range(self) -> None:
        commits = [
            _commit_info(i, f"2024-01-0{i + 1}T10:00:00+00:00")
            for i in range(3)
        ]
        with (
            patch.object(GitFilterRepoAdapter, "_pick_random_time", return_value=(19, 0, 0)),
            patch("git_filter_repo_mcp.adapter.random.random", return_value=1.0),
        ):
            result = _make_mock_adapter()._generate_date_mappings(
                commits, 19, 0, 19, 0, False, True, None,
            )

        assert isinstance(result, dict) and len(result) == 3
        timestamps = [ts for ts, _tz in result.values()]
        assert timestamps == sorted(timestamps)
        for ts in timestamps:
            dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
            assert (dt.hour, dt.minute, dt.second) == (19, 0, 0)


class TestRewriteSingleCommitUnit:
    """Single-commit rewrite checks that do not need git-filter-repo."""

    def test_dry_run_resolves_hash_without_running_filter_repo(self) -> None:
        adapter = _make_mock_adapter()
        resolved = "a" * 40
        adapter._run_git = MagicMock(  # type: ignore[method-assign]
            return_value=subprocess.CompletedProcess(["git"], 0, stdout=f"{resolved}\n", stderr="")
        )
        adapter._run_filter_repo = MagicMock()  # type: ignore[method-assign]

        result = adapter.rewrite_single_commit("a" * 8, new_message="new", dry_run=True)

        assert result.success and result.dry_run
        assert result.commits_processed == 1
        adapter._run_filter_repo.assert_not_called()

    def test_uses_resolved_full_hash_for_callback_target(self) -> None:
        adapter = _make_mock_adapter()
        resolved = "b" * 40
        adapter._run_git = MagicMock(  # type: ignore[method-assign]
            return_value=subprocess.CompletedProcess(["git"], 0, stdout=f"{resolved}\n", stderr="")
        )
        adapter._build_callback = MagicMock(return_value="callback")  # type: ignore[method-assign]
        adapter._run_filter_repo = MagicMock(  # type: ignore[method-assign]
            return_value=subprocess.CompletedProcess(["git-filter-repo"], 0, stdout="", stderr="")
        )

        result = adapter.rewrite_single_commit("b" * 8, new_message="new", force=False)

        assert result.success
        payload = adapter._build_callback.call_args.args[0]
        assert payload["_PAYLOAD"]["target"] == resolved
        adapter._run_filter_repo.assert_called_once_with(
            "--commit-callback", "callback", dry_run=False, force=False,
        )

    def test_missing_commit_does_not_run_filter_repo(self) -> None:
        adapter = _make_mock_adapter()
        adapter._run_git = MagicMock(  # type: ignore[method-assign]
            side_effect=subprocess.CalledProcessError(
                128, ["git"], stderr="fatal: Needed a single revision"
            )
        )
        adapter._run_filter_repo = MagicMock()  # type: ignore[method-assign]

        result = adapter.rewrite_single_commit("c" * 8, new_message="new")

        assert result.success is False
        assert "not found or ambiguous" in result.message
        adapter._run_filter_repo.assert_not_called()

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"new_author_name": "Bad\nName"},
            {"new_author_email": "bad@example.com> <extra@example.com"},
            {"new_message": "bad\x00message"},
        ],
    )
    def test_invalid_rewrite_values_rejected_before_git(self, kwargs: dict) -> None:
        adapter = _make_mock_adapter()
        adapter._run_git = MagicMock()  # type: ignore[method-assign]

        result = adapter.rewrite_single_commit("d" * 8, **kwargs)

        assert result.success is False
        adapter._run_git.assert_not_called()


class TestRewriteMessagesUnit:
    """Message rewrite no-op behaviour without spawning git-filter-repo."""

    def test_no_changes_does_not_run_filter_repo(self) -> None:
        adapter = _make_mock_adapter()
        adapter.get_commits = MagicMock(return_value=[_commit_info(1)])  # type: ignore[method-assign]
        adapter._run_filter_repo = MagicMock()  # type: ignore[method-assign]

        result = adapter.rewrite_commit_messages(lambda msg, _hash: msg, dry_run=False)

        assert result.success
        assert result.commits_rewritten == 0
        adapter._run_filter_repo.assert_not_called()


class TestSquashCommitsUnit:
    """Safety checks in squash before it mutates HEAD/index."""

    @staticmethod
    def _cp(stdout: str = "") -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(["git"], 0, stdout=stdout, stderr="")

    def test_dry_run_rejects_non_head_end_commit(self) -> None:
        adapter = _make_mock_adapter()

        def run_git(*args: str, **_kwargs) -> subprocess.CompletedProcess:
            if args[:2] == ("rev-list", "--count"):
                return self._cp("2\n")
            if args == ("rev-parse", "HEAD"):
                return self._cp("head\n")
            if args == ("rev-parse", "feature"):
                return self._cp("feature\n")
            raise AssertionError(f"unexpected git args: {args}")

        adapter._run_git = MagicMock(side_effect=run_git)  # type: ignore[method-assign]

        result = adapter.squash_commits("base", "feature", dry_run=True)

        assert result.success is False
        assert "must be HEAD" in result.message

    def test_non_dry_run_requires_clean_worktree(self) -> None:
        adapter = _make_mock_adapter()

        def run_git(*args: str, **_kwargs) -> subprocess.CompletedProcess:
            if args[:2] == ("rev-list", "--count"):
                return self._cp("2\n")
            if args == ("rev-parse", "HEAD"):
                return self._cp("head\n")
            raise AssertionError(f"unexpected git args: {args}")

        adapter._run_git = MagicMock(side_effect=run_git)  # type: ignore[method-assign]
        adapter._run_git_fast = MagicMock(return_value=self._cp(" M dirty.txt\n"))  # type: ignore[method-assign]

        result = adapter.squash_commits("base", "HEAD", dry_run=False)

        assert result.success is False
        assert "Working tree must be clean" in result.message


class TestBackupBranchFormat:
    """The static format of ``create_backup``-generated branch names."""

    @requires_git_filter_repo
    def test_includes_microseconds(self, adapter: GitFilterRepoAdapter) -> None:
        backup = adapter.create_backup()
        parts = backup.split("_")
        assert len(parts) == 4         # backup_<date>_<time>_<microseconds>
        assert len(parts[3]) == 6      # ffffff


# =========================================================================
# Integration slice — fixtures spawn real git + git-filter-repo
# =========================================================================

# Every class below uses ``temp_git_repo`` or ``adapter`` (which depends on
# ``temp_git_repo``), so they are automatically tagged ``integration`` by
# ``conftest.pytest_collection_modifyitems``.


@requires_git_filter_repo
class TestReadOperations:
    """Adapter read methods over a real 3-commit repo."""

    def test_get_commits_returns_all_in_reverse_chronological_order(
        self, adapter: GitFilterRepoAdapter,
    ) -> None:
        commits = adapter.get_commits()
        assert [c.message for c in commits] == ["Add config", "Add main.py", "Initial commit"]

    @pytest.mark.parametrize(
        "kwargs,expected_count",
        [
            ({}, 3),
            ({"max_count": 2}, 2),
            ({"max_count": 0}, 0),
            ({"max_count": None}, 3),
            ({"branch": "HEAD"}, 3),
        ],
    )
    def test_get_commits_count(
        self, adapter: GitFilterRepoAdapter, kwargs: dict, expected_count: int,
    ) -> None:
        assert len(adapter.get_commits(**kwargs)) == expected_count

    def test_get_commits_rejects_dash_branch(self, adapter: GitFilterRepoAdapter) -> None:
        with pytest.raises(ValueError, match="must not start with"):
            adapter.get_commits(branch="--exec=evil")

    def test_get_commit_files_returns_changed_files(self, adapter: GitFilterRepoAdapter) -> None:
        commits = adapter.get_commits()
        assert "config.json" in adapter.get_commit_files(commits[0].hash)
        assert "README.md" in adapter.get_commit_files(commits[-1].hash)

    def test_get_commit_diff_shows_files(self, adapter: GitFilterRepoAdapter) -> None:
        commits = adapter.get_commits()
        assert "config.json" in adapter.get_commit_diff(commits[0].hash)

    def test_list_all_files_in_history(self, adapter: GitFilterRepoAdapter) -> None:
        files = adapter.list_all_files_in_history()
        assert {"README.md", "main.py", "config.json"} <= set(files)

    def test_list_all_files_respects_limit(self, adapter: GitFilterRepoAdapter) -> None:
        assert len(adapter.list_all_files_in_history(limit=2)) == 2

    def test_get_file_history(self, adapter: GitFilterRepoAdapter) -> None:
        history = adapter.get_file_history("README.md")
        assert len(history) == 1
        assert history[0]["message"] == "Initial commit"

    def test_get_file_history_for_nonexistent_file_empty(
        self, adapter: GitFilterRepoAdapter,
    ) -> None:
        assert adapter.get_file_history("nonexistent.txt") == []

    def test_validate_repo_safety_clean_repo(self, adapter: GitFilterRepoAdapter) -> None:
        result = adapter.validate_repo_safety()
        assert result["is_clean"] is True
        assert result["safe_for_rewrite"] is True
        assert result["head"]

    def test_validate_repo_safety_detects_dirty_worktree(
        self, adapter: GitFilterRepoAdapter, temp_git_repo: Path,
    ) -> None:
        (temp_git_repo / "uncommitted.txt").write_text("dirty")

        result = adapter.validate_repo_safety()

        assert result["is_clean"] is False
        assert result["safe_for_rewrite"] is False
        assert result["dirty_count"] >= 1
        assert any("uncommitted.txt" in entry for entry in result["dirty_entries"])
        assert "Working tree has uncommitted changes" in result["warnings"]

    def test_resolve_commit_ref_head(self, adapter: GitFilterRepoAdapter) -> None:
        result = adapter.resolve_commit_ref("HEAD")
        assert result is not None
        assert result["hash"] == adapter.get_commits()[0].hash
        assert result["hash_short"] == result["hash"][:8]

    def test_find_large_files_read_only_no_match(self, adapter: GitFilterRepoAdapter) -> None:
        result = adapter.find_large_files(size_threshold_mb=100.0, limit=10)
        assert result["total_large_files"] == 0
        assert result["large_files"] == []

    def test_find_large_files_returns_sorted_limited_results(
        self, adapter: GitFilterRepoAdapter, temp_git_repo: Path,
    ) -> None:
        (temp_git_repo / "large.bin").write_bytes(b"x" * 2048)
        (temp_git_repo / "huge.bin").write_bytes(b"y" * 4096)
        subprocess.run(["git", "add", "."], cwd=temp_git_repo, capture_output=True)
        subprocess.run(["git", "commit", "-m", "add large files"], cwd=temp_git_repo, capture_output=True)

        result = adapter.find_large_files(size_threshold_mb=0.001, limit=1)

        assert result["total_large_files"] >= 2
        assert result["returned"] == 1
        assert result["truncated"] is True
        assert result["large_files"][0]["path"] == "huge.bin"
        assert result["large_files"][0]["size_mb"] > 0.001


@requires_git_filter_repo
class TestAnalyzeHistory:
    """All ``analyze_history`` assertions in one place."""

    def test_summary_fields_for_full_history(self, adapter: GitFilterRepoAdapter) -> None:
        result = adapter.analyze_history()
        assert result["total_commits"] == 3
        assert result["total_authors"] == 1
        assert result["authors"]["Test User <test@example.com>"] == 3
        assert len(result["commits"]) == 3

    def test_total_in_branch_independent_of_max_count(
        self, adapter: GitFilterRepoAdapter,
    ) -> None:
        result = adapter.analyze_history(max_count=1)
        assert result["total_commits"] == 1
        assert result["total_in_branch"] == 3

    def test_long_message_truncated(
        self, adapter: GitFilterRepoAdapter, temp_git_repo: Path,
    ) -> None:
        add_commit(temp_git_repo, "long.txt", "content", "feat: " + "a" * 100)
        latest = adapter.analyze_history()["commits"][0]
        assert latest["message"].endswith("...")
        assert len(latest["message"]) == 83  # 80 + "..."

    def test_short_message_not_truncated(self, adapter: GitFilterRepoAdapter) -> None:
        for c in adapter.analyze_history()["commits"]:
            if len(c["message"]) < 80:
                assert not c["message"].endswith("...")

    def test_empty_repo(self, empty_git_repo: Path) -> None:
        result = GitFilterRepoAdapter(str(empty_git_repo)).analyze_history()
        assert result["total_commits"] == 0
        assert result["authors"] == {}


@requires_git_filter_repo
class TestCollectCommitFiles:
    """Bulk ``collect_commit_files`` parity with per-commit reads."""

    def test_returns_files_keyed_by_hash(self, adapter: GitFilterRepoAdapter) -> None:
        commits = adapter.get_commits()
        bulk = adapter.collect_commit_files(commits, "HEAD", len(commits))
        assert {c.hash for c in commits} <= bulk.keys()
        config_commit = next(c for c in commits if c.message == "Add config")
        assert "config.json" in bulk[config_commit.hash]

    def test_matches_per_commit_get_commit_files(
        self, adapter: GitFilterRepoAdapter,
    ) -> None:
        commits = adapter.get_commits()
        bulk = adapter.collect_commit_files(commits, "HEAD", len(commits))
        for c in commits:
            assert sorted(bulk[c.hash]) == sorted(adapter.get_commit_files(c.hash))


@requires_git_filter_repo
class TestDryRun:
    """Every dry-run path: assert (a) success+dry_run=True and (b) repo
    state is unchanged afterwards."""

    def test_rewrite_messages(self, adapter: GitFilterRepoAdapter) -> None:
        result = adapter.rewrite_commit_messages(
            lambda msg, _h: f"[REWRITTEN] {msg}", dry_run=True,
        )
        assert result.success and result.dry_run
        assert result.commits_rewritten == 3
        assert not adapter.get_commits()[0].message.startswith("[REWRITTEN]")

    def test_change_author(self, adapter: GitFilterRepoAdapter) -> None:
        result = adapter.change_author(
            "test@example.com", "New Name", "new@example.com", dry_run=True,
        )
        assert result.success and result.dry_run
        assert result.commits_rewritten == 3
        assert adapter.get_commits()[0].author_email == "test@example.com"

    def test_squash_commits(self, adapter: GitFilterRepoAdapter) -> None:
        commits = adapter.get_commits()
        result = adapter.squash_commits(start_commit=commits[2].hash, dry_run=True)
        assert result.success and result.dry_run
        assert result.commits_processed == 2
        assert len(adapter.get_commits()) == 3

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"time_range": "evening"},
            {"time_range": "20:00-23:00"},
            {"time_range": "random", "weekend_only": True},
        ],
        ids=["preset_evening", "custom_range", "weekend_only"],
    )
    def test_change_commit_dates(
        self, adapter: GitFilterRepoAdapter, kwargs: dict,
    ) -> None:
        result = adapter.change_commit_dates(dry_run=True, **kwargs)
        assert result.success and result.dry_run

    def test_change_commit_dates_invalid_range(
        self, adapter: GitFilterRepoAdapter,
    ) -> None:
        result = adapter.change_commit_dates(time_range="invalid", dry_run=True)
        assert result.success is False
        assert "Unknown time range" in result.message

    def test_replace_text(self, adapter: GitFilterRepoAdapter) -> None:
        result = adapter.replace_text_in_history(
            old_text="hello", new_text="world", dry_run=True,
        )
        assert result.success and result.dry_run
        assert "history" in result.message

    def test_filter_paths_include(self, adapter: GitFilterRepoAdapter) -> None:
        result = adapter.filter_paths(include_paths=["main.py"], dry_run=True)
        assert result.success and result.dry_run

    def test_filter_paths_exclude(self, adapter: GitFilterRepoAdapter) -> None:
        result = adapter.filter_paths(exclude_paths=["config.json"], dry_run=True)
        assert result.success and result.dry_run

    def test_filter_paths_no_paths_rejected(
        self, adapter: GitFilterRepoAdapter,
    ) -> None:
        result = adapter.filter_paths(dry_run=True)
        assert result.success is False
        assert "No paths specified" in result.message

    def test_filter_paths_include_and_exclude_rejected(
        self, adapter: GitFilterRepoAdapter,
    ) -> None:
        result = adapter.filter_paths(
            include_paths=["src/"], exclude_paths=["tests/"], dry_run=True,
        )
        assert result.success is False
        assert "Cannot use include_paths and exclude_paths together" in result.message

    def test_remove_large_files_no_match(self, adapter: GitFilterRepoAdapter) -> None:
        result = adapter.remove_large_files(size_threshold_mb=10.0, dry_run=True)
        assert result.success and result.dry_run
        assert result.files_affected == []

    def test_remove_large_files_finds_match(
        self, adapter: GitFilterRepoAdapter, temp_git_repo: Path,
    ) -> None:
        (temp_git_repo / "big.bin").write_bytes(b"x" * 2048)
        subprocess.run(["git", "add", "."], cwd=temp_git_repo, capture_output=True)
        subprocess.run(["git", "commit", "-m", "big"], cwd=temp_git_repo, capture_output=True)
        # 0.001 MB = ~1KB, our 2KB file should exceed it.
        result = adapter.remove_large_files(size_threshold_mb=0.001, dry_run=True)
        assert result.success and result.dry_run
        assert any("big.bin" in f for f in result.files_affected)

    def test_remove_files_nonexistent(self, adapter: GitFilterRepoAdapter) -> None:
        result = adapter.remove_files(["nonexistent.txt"], dry_run=True)
        assert result.success
        assert result.commits_rewritten == 0

    def test_squash_invalid_range(self, adapter: GitFilterRepoAdapter) -> None:
        result = adapter.squash_commits(start_commit="0" * 40, dry_run=True)
        assert result.success is False
        assert "Invalid commit range" in result.message

    def test_squash_zero_range(self, adapter: GitFilterRepoAdapter) -> None:
        commits = adapter.get_commits()
        result = adapter.squash_commits(start_commit=commits[0].hash, dry_run=True)
        assert result.success is False
        assert "No commits in range" in result.message


@requires_git_filter_repo
class TestExecution:
    """Real (non-dry-run) operations. These actually mutate the repo;
    the adapter is recreated after a mutation to pick up the new HEAD."""

    def test_remove_files_actually_removes(
        self, adapter: GitFilterRepoAdapter, temp_git_repo: Path,
    ) -> None:
        assert "config.json" in adapter.list_all_files_in_history()
        result = adapter.remove_files(["config.json"], dry_run=False, force=True)
        assert result.success and not result.dry_run
        files = GitFilterRepoAdapter(str(temp_git_repo)).list_all_files_in_history()
        assert "config.json" not in files

    def test_change_author_actually_changes(
        self, adapter: GitFilterRepoAdapter, temp_git_repo: Path,
    ) -> None:
        result = adapter.change_author(
            "test@example.com", "Changed User", "changed@example.com",
            dry_run=False, force=True,
        )
        assert result.success
        for c in GitFilterRepoAdapter(str(temp_git_repo)).get_commits():
            assert c.author_email == "changed@example.com"
            assert c.author_name == "Changed User"

    def test_rewrite_messages_actually_changes(
        self, adapter: GitFilterRepoAdapter, temp_git_repo: Path,
    ) -> None:
        result = adapter.rewrite_commit_messages(
            lambda msg, _h: f"[PREFIXED] {msg}", dry_run=False, force=True,
        )
        assert result.success
        assert result.commits_rewritten == 3
        for c in GitFilterRepoAdapter(str(temp_git_repo)).get_commits():
            assert c.message.startswith("[PREFIXED]")

    def test_replace_text_actually_replaces(
        self, adapter: GitFilterRepoAdapter, temp_git_repo: Path,
    ) -> None:
        result = adapter.replace_text_in_history(
            old_text="hello", new_text="world", dry_run=False, force=True,
        )
        assert result.success
        content = (temp_git_repo / "main.py").read_text()
        assert "world" in content and "hello" not in content

    def test_filter_paths_include_actually_filters(
        self, adapter: GitFilterRepoAdapter, temp_git_repo: Path,
    ) -> None:
        adapter.filter_paths(include_paths=["README.md"], dry_run=False, force=True)
        files = GitFilterRepoAdapter(str(temp_git_repo)).list_all_files_in_history()
        assert "README.md" in files
        assert "config.json" not in files
        assert "main.py" not in files

    def test_remove_large_files_actual_removal(
        self, adapter: GitFilterRepoAdapter, temp_git_repo: Path,
    ) -> None:
        (temp_git_repo / "huge.bin").write_bytes(b"y" * 4096)
        subprocess.run(["git", "add", "."], cwd=temp_git_repo, capture_output=True)
        subprocess.run(["git", "commit", "-m", "huge"], cwd=temp_git_repo, capture_output=True)
        result = adapter.remove_large_files(size_threshold_mb=0.001, dry_run=False, force=True)
        assert result.success
        files = GitFilterRepoAdapter(str(temp_git_repo)).list_all_files_in_history()
        assert "huge.bin" not in files

    def test_remove_large_files_high_threshold(
        self, adapter: GitFilterRepoAdapter,
    ) -> None:
        result = adapter.remove_large_files(size_threshold_mb=100.0, dry_run=False)
        assert result.success
        assert "No large files found" in result.message


@requires_git_filter_repo
class TestRewriteSingleCommit:
    """Per-commit message/author edits."""

    def test_message_change(
        self, adapter: GitFilterRepoAdapter, temp_git_repo: Path,
    ) -> None:
        target = adapter.get_commits()[0]
        result = adapter.rewrite_single_commit(
            target.hash, new_message="Updated config message", force=True,
        )
        assert result.success
        assert result.commits_rewritten == 1
        new_commits = GitFilterRepoAdapter(str(temp_git_repo)).get_commits()
        assert new_commits[0].message == "Updated config message"
        assert new_commits[1].message == "Add main.py"  # untouched

    def test_author_change(
        self, adapter: GitFilterRepoAdapter, temp_git_repo: Path,
    ) -> None:
        target = adapter.get_commits()[1]  # "Add main.py"
        result = adapter.rewrite_single_commit(
            target.hash, new_author_name="New Author",
            new_author_email="newauthor@test.com", force=True,
        )
        assert result.success
        rewritten = next(
            c for c in GitFilterRepoAdapter(str(temp_git_repo)).get_commits()
            if c.message == "Add main.py"
        )
        assert rewritten.author_name == "New Author"
        assert rewritten.author_email == "newauthor@test.com"

    def test_no_changes_returns_error(self, adapter: GitFilterRepoAdapter) -> None:
        result = adapter.rewrite_single_commit(adapter.get_commits()[0].hash)
        assert result.success is False
        assert "No changes specified" in result.message


@requires_git_filter_repo
class TestSquashCommits:
    """Multi-commit squash, including dry/exec and end_commit validation."""

    def test_squashes_actually(
        self, adapter: GitFilterRepoAdapter, temp_git_repo: Path,
    ) -> None:
        commits = adapter.get_commits()
        # commits[2] = initial (excluded), commits[0] = HEAD
        result = adapter.squash_commits(
            start_commit=commits[2].hash, end_commit="HEAD",
            new_message="squashed two", dry_run=False,
        )
        assert result.success
        assert result.commits_rewritten == 1
        assert result.commits_processed == 2
        after = GitFilterRepoAdapter(str(temp_git_repo)).get_commits()
        assert len(after) == 2
        assert after[0].message == "squashed two"
        assert after[1].message == "Initial commit"

    def test_default_message_lists_squashed_commits(
        self, adapter: GitFilterRepoAdapter, temp_git_repo: Path,
    ) -> None:
        commits = adapter.get_commits()
        result = adapter.squash_commits(
            start_commit=commits[2].hash, end_commit="HEAD", dry_run=False,
        )
        assert result.success
        head_msg = GitFilterRepoAdapter(str(temp_git_repo)).get_commits()[0].message
        assert "Squashed commits:" in head_msg
        assert "Add config" in head_msg

    def test_non_head_end_commit_rejected(self, adapter: GitFilterRepoAdapter) -> None:
        commits = adapter.get_commits()
        result = adapter.squash_commits(
            start_commit=commits[2].hash, end_commit=commits[1].hash, dry_run=False,
        )
        assert result.success is False
        assert "must be HEAD" in result.message


@requires_git_filter_repo
class TestChangeCommitDates:
    """Date rewriting end-to-end (helpers covered by ``TestParseTimeRange``
    and ``TestGenerateDateMappings`` above)."""

    def test_dates_change(
        self, adapter: GitFilterRepoAdapter, temp_git_repo: Path,
    ) -> None:
        before = [c.date for c in adapter.get_commits()]
        result = adapter.change_commit_dates(
            time_range="night", dry_run=False, force=True,
        )
        assert result.success
        assert result.commits_rewritten == 3
        after = [c.date for c in GitFilterRepoAdapter(str(temp_git_repo)).get_commits()]
        assert before != after

    def test_night_range_produces_valid_hours(
        self, adapter: GitFilterRepoAdapter, temp_git_repo: Path,
    ) -> None:
        adapter.change_commit_dates(time_range="night", dry_run=False, force=True)
        for c in GitFilterRepoAdapter(str(temp_git_repo)).get_commits():
            hour = datetime.datetime.fromisoformat(c.date).hour
            assert hour >= 22 or hour <= 2, f"hour {hour} outside 22-02"

    def test_weekend_preset_implies_weekend_only(
        self, adapter: GitFilterRepoAdapter,
    ) -> None:
        """The ``weekend`` preset name must force weekend_only behaviour
        even when the user leaves ``weekend_only=False`` (regression)."""
        result = adapter.change_commit_dates(
            time_range="weekend", weekend_only=False, dry_run=True,
        )
        assert result.success
        for line in result.message.splitlines():
            if "->" not in line:
                continue
            iso = line.rsplit("-> ", 1)[1].strip()
            assert datetime.datetime.fromisoformat(iso).weekday() >= 5

    def test_single_commit_dry_run(self, single_commit_repo: Path) -> None:
        result = GitFilterRepoAdapter(str(single_commit_repo)).change_commit_dates(
            time_range="evening", dry_run=True,
        )
        assert result.success
        assert result.commits_rewritten == 1


@requires_git_filter_repo
class TestBackupAndRestore:
    """Backup branch creation, restore, and the validation guarding restore."""

    def test_create_backup_actually_creates_branch(
        self, adapter: GitFilterRepoAdapter, temp_git_repo: Path,
    ) -> None:
        backup = adapter.create_backup()
        assert backup.startswith("backup_")
        listing = subprocess.run(
            ["git", "branch", "--list", backup],
            cwd=temp_git_repo, capture_output=True, text=True,
        )
        assert backup in listing.stdout

    def test_list_backups(self, adapter: GitFilterRepoAdapter) -> None:
        backup = adapter.create_backup()
        result = adapter.list_backups()
        assert backup in result["backups"]
        assert result["total_backups"] >= 1

    def test_restore_returns_to_backup_state(
        self, adapter: GitFilterRepoAdapter, temp_git_repo: Path,
    ) -> None:
        backup = adapter.create_backup()
        commits_before = adapter.get_commits()

        # Diverge from backup by adding a commit.
        add_commit(temp_git_repo, "extra.txt", "extra", "extra")
        assert len(GitFilterRepoAdapter(str(temp_git_repo)).get_commits()) == 4

        result = GitFilterRepoAdapter(str(temp_git_repo)).restore_backup(backup)
        assert result.success

        commits_after = GitFilterRepoAdapter(str(temp_git_repo)).get_commits()
        assert len(commits_after) == 3
        assert commits_after[0].hash == commits_before[0].hash

        # Backup branch is not deleted by restore.
        listing = subprocess.run(
            ["git", "branch", "--list", backup],
            cwd=temp_git_repo, capture_output=True, text=True,
        )
        assert backup in listing.stdout

    @pytest.mark.parametrize(
        "branch,reason",
        [
            ("main", "Invalid backup branch"),
            ("backup_; rm -rf /", "Invalid backup branch"),
            ("backup_2024-01-01", "Invalid backup branch"),
            ("backup_123", "Invalid backup branch"),
        ],
    )
    def test_invalid_branch_name_rejected(
        self, adapter: GitFilterRepoAdapter, branch: str, reason: str,
    ) -> None:
        result = adapter.restore_backup(branch)
        assert result.success is False
        assert reason in result.message

    def test_well_formed_but_missing_branch_rejected(
        self, adapter: GitFilterRepoAdapter,
    ) -> None:
        result = adapter.restore_backup("backup_99991231_235959_000000")
        assert result.success is False


@requires_git_filter_repo
class TestChangeAuthorValidation:
    """Input validation for ``change_author`` (the no-match path is here too)."""

    @pytest.mark.parametrize(
        "name,email",
        [
            ("", "new@example.com"),       # empty name
            ("  ", "new@example.com"),     # whitespace-only name
        ],
    )
    def test_empty_name_rejected(
        self, adapter: GitFilterRepoAdapter, name: str, email: str,
    ) -> None:
        result = adapter.change_author("test@example.com", name, email, dry_run=True)
        assert result.success is False
        assert "empty" in result.message.lower() or "Invalid" in result.message

    def test_empty_old_email_rejected(self, adapter: GitFilterRepoAdapter) -> None:
        result = adapter.change_author("", "Name", "new@example.com", dry_run=True)
        assert result.success is False

    @pytest.mark.parametrize(
        "name,email",
        [
            ("Evil\nName", "evil@example.com"),                       # newline
            ("Evil\tName", "evil@example.com"),                       # tab
            ("Evil", "evil@example.com> <extra@inject.com"),          # angle brackets
        ],
        ids=["newline", "tab", "angle_bracket"],
    )
    def test_injection_chars_rejected(
        self, adapter: GitFilterRepoAdapter, name: str, email: str,
    ) -> None:
        # The check must also run in dry_run so callers fail fast.
        for dry_run in (True, False):
            result = adapter.change_author(
                "test@example.com", name, email, dry_run=dry_run,
            )
            assert result.success is False
            assert "Invalid characters" in result.message

    def test_clean_inputs_pass_dry_run(self, adapter: GitFilterRepoAdapter) -> None:
        result = adapter.change_author(
            "test@example.com", "Clean Name", "clean@example.com", dry_run=True,
        )
        assert result.success

    def test_no_match_returns_helpful_message(
        self, adapter: GitFilterRepoAdapter,
    ) -> None:
        result = adapter.change_author(
            "nonexistent@email.com", "Name", "new@e.com", dry_run=True,
        )
        assert result.success
        assert result.commits_rewritten == 0
        assert "nonexistent@email.com" in result.message
        assert "test@example.com" in result.message  # existing emails listed


@requires_git_filter_repo
class TestPathInputValidation:
    """Path-list inputs to public methods reject option injection."""

    @pytest.mark.parametrize(
        "method,kwargs",
        [
            ("remove_files", {"paths": ["--commit-callback", "evil"], "dry_run": True}),
            ("filter_paths", {"include_paths": ["--force"], "dry_run": True}),
        ],
    )
    def test_option_injection_rejected(
        self, adapter: GitFilterRepoAdapter, method: str, kwargs: dict,
    ) -> None:
        with pytest.raises(ValueError, match="must not start with"):
            getattr(adapter, method)(**kwargs)

    def test_remove_files_empty_list_rejected(
        self, adapter: GitFilterRepoAdapter,
    ) -> None:
        result = adapter.remove_files([], dry_run=True)
        assert result.success is False
        assert "No file paths" in result.message


@requires_git_filter_repo
class TestReplaceTextValidation:
    """``replace_text_in_history`` argument validation."""

    @pytest.mark.parametrize(
        "old_text,new_text,error_fragment",
        [
            ("", "x", "empty"),
            ("hello\nworld", "replaced", "newlines"),
            ("hello", "re\nplaced", "newlines"),
        ],
    )
    def test_invalid_text_rejected(
        self, adapter: GitFilterRepoAdapter,
        old_text: str, new_text: str, error_fragment: str,
    ) -> None:
        result = adapter.replace_text_in_history(
            old_text=old_text, new_text=new_text, dry_run=True,
        )
        assert result.success is False
        assert error_fragment in result.message.lower()

    def test_separator_in_new_text_escaped(self, adapter: GitFilterRepoAdapter) -> None:
        """``==>`` is git-filter-repo's regex separator; the adapter must
        escape it so users can replace literal text containing ``==>``."""
        result = adapter.replace_text_in_history(
            old_text="hello", new_text="a==>b", dry_run=True,
        )
        assert result.success


@requires_git_filter_repo
class TestDuplicateMessageRewrite:
    """Hash-based lookup must rewrite all copies of a duplicated message."""

    def test_both_duplicates_rewritten(
        self, adapter: GitFilterRepoAdapter, temp_git_repo: Path,
    ) -> None:
        add_commit(temp_git_repo, "config2.json", '{"k2":"v2"}', "Add config")
        commits = adapter.get_commits()
        assert sum(c.message == "Add config" for c in commits) == 2

        adapter.rewrite_commit_messages(
            lambda msg, _h: "chore: add configuration" if msg == "Add config" else msg,
            dry_run=False, force=True,
        )
        rewritten = [
            c for c in GitFilterRepoAdapter(str(temp_git_repo)).get_commits()
            if c.message == "chore: add configuration"
        ]
        assert len(rewritten) == 2


@requires_git_filter_repo
class TestScanSecrets:
    """``scan_secrets`` pattern hits and message formatting."""

    @pytest.mark.parametrize(
        "filename,content,expected_type",
        [
            ("creds.txt", "key = AKIAIOSFODNN7EXAMPLE", "aws_access_key"),
            (
                "token.txt",
                "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmn",
                "github_token",
            ),
            (
                "key.pem",
                "-----BEGIN RSA PRIVATE KEY-----\nfake\n-----END RSA PRIVATE KEY-----",
                "private_key",
            ),
        ],
    )
    def test_pattern_detected(
        self, adapter: GitFilterRepoAdapter, temp_git_repo: Path,
        filename: str, content: str, expected_type: str,
    ) -> None:
        add_commit(temp_git_repo, filename, content, f"add {filename}")
        result = GitFilterRepoAdapter(str(temp_git_repo)).scan_secrets()
        assert any(f["type"] == expected_type for f in result["findings"])

    def test_no_secrets_on_clean_repo(self, single_commit_repo: Path) -> None:
        result = GitFilterRepoAdapter(str(single_commit_repo)).scan_secrets()
        assert result["secrets_found"] == 0
        assert "No secrets" in result["message"]

    def test_message_mentions_secret_count(
        self, adapter: GitFilterRepoAdapter, temp_git_repo: Path,
    ) -> None:
        add_commit(temp_git_repo, "key.txt", "AKIAIOSFODNN7EXAMPLE", "add key")
        result = GitFilterRepoAdapter(str(temp_git_repo)).scan_secrets()
        assert "secret" in result["message"].lower()

    def test_sensitive_file_deduplicated_across_commits(
        self, adapter: GitFilterRepoAdapter, temp_git_repo: Path,
    ) -> None:
        add_commit(temp_git_repo, ".env", "SECRET=abc", "add env")
        add_commit(temp_git_repo, ".env", "SECRET=def", "update env")
        result = GitFilterRepoAdapter(str(temp_git_repo)).scan_secrets()
        env_entries = [f for f in result["sensitive_file_list"] if f["file"] == ".env"]
        assert len(env_entries) == 1

    def test_scan_secrets_includes_limit_metadata(self, adapter: GitFilterRepoAdapter) -> None:
        result = adapter.scan_secrets()
        assert "scan_limits" in result
        assert "files_considered" in result
        assert "files_truncated" in result
        assert "findings_truncated" in result

    def test_scan_secrets_reports_file_scan_truncation(
        self, adapter: GitFilterRepoAdapter, temp_git_repo: Path,
    ) -> None:
        for i in range(MAX_FILES_TO_SCAN + 5):
            (temp_git_repo / f"candidate-{i:03d}.txt").write_text("clean")
        subprocess.run(["git", "add", "."], cwd=temp_git_repo, capture_output=True)
        subprocess.run(["git", "commit", "-m", "add many files"], cwd=temp_git_repo, capture_output=True)

        result = adapter.scan_secrets(max_commits=1)

        assert result["files_considered"] == MAX_FILES_TO_SCAN + 5
        assert result["files_scanned"] == MAX_FILES_TO_SCAN
        assert result["files_truncated"] is True
        assert result["scan_limits"]["max_files_to_scan"] == MAX_FILES_TO_SCAN

    def test_scan_secrets_reports_finding_truncation(
        self, adapter: GitFilterRepoAdapter, temp_git_repo: Path,
    ) -> None:
        secrets = "\n".join(
            f"AKIA{i:016d}"
            for i in range(MAX_FINDINGS_LIMIT + 5)
        )
        add_commit(temp_git_repo, "many-secrets.txt", secrets, "add many secrets")

        result = adapter.scan_secrets(max_commits=1)

        assert result["secrets_found"] == MAX_FINDINGS_LIMIT
        assert len(result["findings"]) == MAX_FINDINGS_LIMIT
        assert result["findings_truncated"] is True
        assert result["scan_limits"]["max_findings"] == MAX_FINDINGS_LIMIT


@requires_git_filter_repo
class TestEdgeCaseRepos:
    """Behaviour on empty / single / unicode / multiline fixtures."""

    def test_get_commits_empty_repo(self, empty_git_repo: Path) -> None:
        assert GitFilterRepoAdapter(str(empty_git_repo)).get_commits() == []

    def test_list_files_empty_repo(self, empty_git_repo: Path) -> None:
        assert GitFilterRepoAdapter(str(empty_git_repo)).list_all_files_in_history() == []

    def test_remove_large_files_empty_repo(self, empty_git_repo: Path) -> None:
        result = GitFilterRepoAdapter(str(empty_git_repo)).remove_large_files(dry_run=True)
        assert result.success

    def test_squash_on_single_commit_repo_noop(self, single_commit_repo: Path) -> None:
        a = GitFilterRepoAdapter(str(single_commit_repo))
        result = a.squash_commits(start_commit=a.get_commits()[0].hash, dry_run=True)
        assert result.commits_processed == 0

    def test_unicode_commit_message_preserved(self, unicode_git_repo: Path) -> None:
        commits = GitFilterRepoAdapter(str(unicode_git_repo)).get_commits()
        assert len(commits) == 1
        assert "✨" in commits[0].message

    def test_unicode_rewrite_dry_run(self, unicode_git_repo: Path) -> None:
        a = GitFilterRepoAdapter(str(unicode_git_repo))
        result = a.rewrite_commit_messages(lambda msg, _h: f"[prefix] {msg}", dry_run=True)
        assert result.success
        assert result.commits_rewritten == 1

    def test_multiline_message_preserved(self, multiline_commit_repo: Path) -> None:
        commits = GitFilterRepoAdapter(str(multiline_commit_repo)).get_commits()
        assert len(commits) == 1
        for fragment in ["feat: add app", "main application", "multiple lines"]:
            assert fragment in commits[0].message

    def test_multiline_rewrite_dry_run(self, multiline_commit_repo: Path) -> None:
        a = GitFilterRepoAdapter(str(multiline_commit_repo))
        result = a.rewrite_commit_messages(lambda msg, _h: f"[r] {msg}", dry_run=True)
        assert result.success
        assert result.commits_rewritten == 1
