"""Tests for the MCP server layer.

These tests exercise the handler layer in isolation: ``GitFilterRepoAdapter``
is always mocked so we never spawn subprocesses or touch the filesystem.

Organisation:

- **Pure helpers** (no patching): ``result_to_dict``, pydantic input models.
- **Protocol layer**: ``list_tools``, ``call_tool``, ``_execute_tool`` dispatch.
- **Per-tool handlers**: one class per MCP tool covering happy path,
  dry-run, validation, and backup behaviour.
- **Cross-cutting**: auto-backup, AI provider plumbing, error envelopes,
  ``main()`` lifecycle.

Common fixtures (``patched_adapter``) and factories (``make_fr``, ``make_ci``)
collapse the otherwise-verbose ``patch + MagicMock + side_effect`` ritual.
"""

import json
import subprocess as sp
from typing import Any, Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from git_filter_repo_mcp.adapter import CommitInfo, FilterResult
from git_filter_repo_mcp.server import (
    _execute_tool,
    call_tool,
    list_tools,
    result_to_dict,
)
from git_filter_repo_mcp.tools import (
    AnalyzeHistoryInput,
    RemoveLargeFilesInput,
    RewriteCommitMessagesInput,
    ErrorCode,
)


# =========================================================================
# Helpers & fixtures
# =========================================================================


def make_fr(**kwargs: Any) -> FilterResult:
    """``FilterResult`` factory with sensible defaults — tests only set what
    they care about. Replaces dozens of inline 7-line constructors."""
    defaults: dict[str, Any] = {
        "success": True,
        "message": "ok",
        "commits_processed": 0,
        "commits_rewritten": 0,
        "files_affected": [],
        "dry_run": False,
        "error": None,
    }
    defaults.update(kwargs)
    return FilterResult(**defaults)


def make_ci(hash_: str = "abc123def456", message: str = "msg") -> CommitInfo:
    """Concise ``CommitInfo`` factory for handler tests."""
    return CommitInfo(
        hash=hash_,
        author_name="Test User", author_email="test@example.com",
        committer_name="Test User", committer_email="test@example.com",
        message=message, date="2024-12-09",
    )


@pytest.fixture
def patched_adapter() -> Iterator[MagicMock]:
    """Patch ``server.GitFilterRepoAdapter`` and yield the mock instance.

    Every server handler instantiates ``GitFilterRepoAdapter(repo_path)``, so
    tests almost always need this. Yielding the *instance* (not the class)
    lets tests set up methods directly: ``patched_adapter.get_commits.return_value = [...]``.
    """
    with patch("git_filter_repo_mcp.server.GitFilterRepoAdapter") as MockAdapter:
        instance = MagicMock()
        MockAdapter.return_value = instance
        yield instance


# =========================================================================
# Pure helpers — no patching needed
# =========================================================================


class TestResultToDict:
    """``result_to_dict`` serialises a ``FilterResult`` for transport."""

    def test_round_trip_preserves_every_field(self) -> None:
        result = make_fr(
            success=True, message="Operation completed",
            commits_processed=10, commits_rewritten=5,
            files_affected=["a.py", "b.py"], dry_run=False, error=None,
        )
        d = result_to_dict(result)
        assert d == {
            "success": True,
            "message": "Operation completed",
            "commits_processed": 10,
            "commits_rewritten": 5,
            "files_affected": ["a.py", "b.py"],
            "dry_run": False,
            "error": None,
        }

    @pytest.mark.parametrize(
        "kwargs,key,expected",
        [
            ({"success": False, "error": "Something went wrong"}, "error", "Something went wrong"),
            ({"dry_run": True}, "dry_run", True),
        ],
    )
    def test_selected_field(self, kwargs: dict, key: str, expected: Any) -> None:
        assert result_to_dict(make_fr(**kwargs))[key] == expected


class TestPydanticInputValidation:
    """Input-model constraints baked into ``tools.py`` (pure pydantic)."""

    @pytest.mark.parametrize(
        "model,kwargs",
        [
            (AnalyzeHistoryInput, {"repo_path": "/tmp", "max_count": 0}),
            (AnalyzeHistoryInput, {"repo_path": "/tmp", "max_count": -1}),
            (AnalyzeHistoryInput, {"repo_path": "/tmp", "max_count": 20000}),
            (RemoveLargeFilesInput, {"repo_path": "/tmp", "size_threshold_mb": 0.0}),
            (RemoveLargeFilesInput, {"repo_path": "/tmp", "size_threshold_mb": -5.0}),
            # Regression: empty repo_path must be rejected at pydantic layer
            # rather than crashing deep inside the adapter.
            (AnalyzeHistoryInput, {"repo_path": ""}),
        ],
    )
    def test_out_of_range_rejected(self, model: type, kwargs: dict) -> None:
        with pytest.raises(Exception):  # pydantic.ValidationError
            model(**kwargs)

    def test_rewrite_messages_defaults(self) -> None:
        params = RewriteCommitMessagesInput(repo_path="/tmp")
        assert params.use_ai is False
        assert params.ai_provider is None
        assert params.dry_run is True
        assert params.style == "conventional"

    @pytest.mark.parametrize("style", ["conventional", "gitmoji", "simple", "detailed"])
    def test_all_valid_styles_accepted(self, style: str) -> None:
        params = RewriteCommitMessagesInput(repo_path="/tmp", style=style)  # type: ignore[arg-type]
        assert params.style == style


# =========================================================================
# Protocol layer — list_tools, call_tool, _execute_tool dispatch
# =========================================================================


class TestListTools:
    """``list_tools`` produces the MCP-shaped tool list."""

    async def test_includes_known_tools_with_required_fields(self) -> None:
        tools = await list_tools()
        names = {t.name for t in tools}
        # Spot-check critical ones — full set is asserted by test_tools.py.
        assert {
            "analyze_git_history", "rewrite_commit_messages",
            "change_author", "create_backup",
        } <= names
        for tool in tools:
            assert tool.name and tool.description and tool.inputSchema is not None


class TestCallTool:
    """``call_tool`` envelopes the handler output as ``TextContent``."""

    async def test_wraps_success(self) -> None:
        with patch("git_filter_repo_mcp.server._execute_tool",
                   AsyncMock(return_value={"success": True, "message": "Done"})):
            result = await call_tool("analyze_git_history", {"repo_path": "/tmp"})
        assert len(result) == 1 and result[0].type == "text"
        assert json.loads(result[0].text)["success"] is True

    async def test_wraps_unexpected_exception(self) -> None:
        with patch("git_filter_repo_mcp.server._execute_tool",
                   AsyncMock(side_effect=ValueError("Test error"))):
            result = await call_tool("analyze_git_history", {"repo_path": "/tmp"})
        data = json.loads(result[0].text)
        assert data["success"] is False
        assert "Test error" in data["error"]


class TestExecuteToolDispatch:
    """Cross-cutting behaviour of ``_execute_tool`` itself."""

    async def test_unknown_tool_returns_tool_not_found(self) -> None:
        result = await _execute_tool("unknown_tool", {})
        assert result["error_code"] == ErrorCode.TOOL_NOT_FOUND
        assert "Unknown tool" in result["error"]

    @pytest.mark.parametrize(
        "tool,args,error_fragment",
        [
            ("analyze_git_history", {}, "repo_path"),
            ("analyze_git_history",
             {"repo_path": "/tmp/repo", "max_count": "not_a_number"}, "Invalid input"),
            ("change_author", {"repo_path": "/tmp/repo"}, "Invalid input"),
        ],
    )
    async def test_pydantic_errors_propagated(
        self, tool: str, args: dict, error_fragment: str,
    ) -> None:
        result = await _execute_tool(tool, args)
        assert result["success"] is False
        assert result["error_code"] == ErrorCode.INVALID_INPUT
        assert error_fragment in result["error"]


# =========================================================================
# Read-only handlers
# =========================================================================


class TestAnalyzeHistoryHandler:
    async def test_forwards_args_and_merges_response(
        self, patched_adapter: MagicMock,
    ) -> None:
        patched_adapter.analyze_history.return_value = {
            "total_commits": 10, "authors": ["test@example.com"], "branches": ["main"],
        }
        result = await _execute_tool("analyze_git_history", {
            "repo_path": "/tmp/repo", "branch": "main", "max_count": 50,
        })
        assert result["success"] is True
        assert result["total_commits"] == 10
        patched_adapter.analyze_history.assert_called_once_with("main", 50)


class TestGetCommitDetailsHandler:
    async def test_returns_full_commit(self, patched_adapter: MagicMock) -> None:
        patched_adapter.get_commits.return_value = [make_ci("abc123def456", "Test commit")]
        patched_adapter.get_commit_files.return_value = ["file1.py", "file2.py"]
        patched_adapter.get_commit_diff.return_value = "+ added\n- removed"

        result = await _execute_tool("get_commit_details", {
            "repo_path": "/tmp/repo", "commit_hash": "abc123",
        })
        assert result["success"] is True
        assert result["commit"]["hash"] == "abc123def456"
        assert result["commit"]["files"] == ["file1.py", "file2.py"]

    async def test_not_found(self, patched_adapter: MagicMock) -> None:
        patched_adapter.get_commits.return_value = []
        result = await _execute_tool("get_commit_details", {
            "repo_path": "/tmp/repo", "commit_hash": "nonexistent",
        })
        assert result["success"] is False
        assert "not found" in result["error"]

    async def test_dash_hash_rejected(self) -> None:
        """Hash sanitisation happens in the handler — no adapter needed."""
        result = await _execute_tool("get_commit_details", {
            "repo_path": "/tmp/repo", "commit_hash": "--exec=evil",
        })
        assert result["success"] is False
        assert result["error_code"] == ErrorCode.INVALID_INPUT


class TestScanSecretsHandler:
    async def test_passes_through(self, patched_adapter: MagicMock) -> None:
        patched_adapter.scan_secrets.return_value = {
            "findings": [], "files_scanned": 50, "commits_scanned": 10,
        }
        result = await _execute_tool("scan_secrets", {
            "repo_path": "/tmp/repo", "branch": "main", "max_commits": 50,
        })
        assert result["success"] is True
        patched_adapter.scan_secrets.assert_called_once_with("main", 50)

    async def test_empty_repo(self, patched_adapter: MagicMock) -> None:
        patched_adapter.scan_secrets.return_value = {
            "commits_scanned": 0, "secrets_found": 0, "sensitive_files": 0,
            "findings": [], "sensitive_file_list": [], "files_scanned": 0,
        }
        result = await _execute_tool("scan_secrets", {"repo_path": "/tmp/repo"})
        assert result["success"] is True
        assert result["secrets_found"] == 0


class TestListAllFilesHandler:
    async def test_summary(self, patched_adapter: MagicMock) -> None:
        patched_adapter.list_all_files_in_history.return_value = [
            "file1.py", "file2.py", "dir/file3.py",
        ]
        result = await _execute_tool("list_all_files_in_history", {"repo_path": "/tmp/repo"})
        assert result["total_files"] == 3
        assert "file1.py" in result["files"]


class TestGetFileHistoryHandler:
    async def test_summary(self, patched_adapter: MagicMock) -> None:
        patched_adapter.get_file_history.return_value = [
            {"hash": "abc", "message": "Add"}, {"hash": "def", "message": "Update"},
        ]
        result = await _execute_tool("get_file_history", {
            "repo_path": "/tmp/repo", "file_path": "src/main.py",
        })
        assert result["total_commits"] == 2
        patched_adapter.get_file_history.assert_called_once_with("src/main.py")


# =========================================================================
# Backup handlers
# =========================================================================


class TestBackupHandlers:
    async def test_create_backup(self, patched_adapter: MagicMock) -> None:
        patched_adapter.create_backup.return_value = "backup-20241209-123456"
        result = await _execute_tool("create_backup", {"repo_path": "/tmp/repo"})
        assert result["backup_branch"] == "backup-20241209-123456"

    async def test_restore_backup(self, patched_adapter: MagicMock) -> None:
        patched_adapter.restore_backup.return_value = make_fr(message="Restored")
        result = await _execute_tool("restore_backup", {
            "repo_path": "/tmp/repo", "backup_branch": "backup-20241209-123456",
        })
        assert result["success"] is True
        patched_adapter.restore_backup.assert_called_once_with("backup-20241209-123456")


# =========================================================================
# Destructive handlers
# =========================================================================


class TestChangeAuthorHandler:
    async def test_dry_run_forwards_args(self, patched_adapter: MagicMock) -> None:
        patched_adapter.change_author.return_value = make_fr(
            message="Would change 5", commits_processed=5, dry_run=True,
        )
        result = await _execute_tool("change_author", {
            "repo_path": "/tmp/repo",
            "old_email": "old@example.com",
            "new_name": "New Name",
            "new_email": "new@example.com",
            "dry_run": True,
        })
        assert result["success"] and result["dry_run"]
        patched_adapter.change_author.assert_called_once_with(
            "old@example.com", "New Name", "new@example.com", True, False,
        )


class TestRemoveFilesHandler:
    async def test_forwards_paths(self, patched_adapter: MagicMock) -> None:
        patched_adapter.remove_files.return_value = make_fr(
            commits_rewritten=3, files_affected=["secret.txt"],
        )
        result = await _execute_tool("remove_files_from_history", {
            "repo_path": "/tmp/repo",
            "paths": ["secret.txt", "config.json"],
            "dry_run": False,
        })
        assert result["files_affected"] == ["secret.txt"]


class TestRemoveLargeFilesHandler:
    async def test_dry_run_forwards_threshold(self, patched_adapter: MagicMock) -> None:
        patched_adapter.remove_large_files.return_value = make_fr(
            files_affected=["big.bin", "huge.zip"], dry_run=True,
        )
        result = await _execute_tool("remove_large_files", {
            "repo_path": "/tmp/repo", "size_threshold_mb": 10.0, "dry_run": True,
        })
        assert result["dry_run"] and result["files_affected"] == ["big.bin", "huge.zip"]
        patched_adapter.remove_large_files.assert_called_once_with(10.0, True, False)


class TestFilterPathsHandler:
    async def test_include_and_exclude_propagates_error(
        self, patched_adapter: MagicMock,
    ) -> None:
        patched_adapter.filter_paths.return_value = make_fr(
            success=False, message="Cannot use include_paths and exclude_paths together",
        )
        result = await _execute_tool("filter_paths", {
            "repo_path": "/tmp/repo",
            "include_paths": ["src/"], "exclude_paths": ["tests/"], "dry_run": True,
        })
        assert result["success"] is False


class TestSquashCommitsHandler:
    async def test_dry_run(self, patched_adapter: MagicMock) -> None:
        patched_adapter.squash_commits.return_value = make_fr(
            commits_processed=3, dry_run=True,
        )
        result = await _execute_tool("squash_commits", {
            "repo_path": "/tmp/repo", "start_commit": "abc123", "end_commit": "HEAD",
            "dry_run": True,
        })
        assert result["dry_run"] and result["commits_processed"] == 3

    async def test_dash_start_commit_returns_error(
        self, patched_adapter: MagicMock,
    ) -> None:
        """Adapter raises ValueError; handler must surface a clean error."""
        patched_adapter.squash_commits.side_effect = ValueError("Invalid ref")
        result = await _execute_tool("squash_commits", {
            "repo_path": "/tmp/repo", "start_commit": "--exec=evil", "dry_run": True,
        })
        assert result["success"] is False


class TestReplaceTextHandler:
    async def test_dry_run(self, patched_adapter: MagicMock) -> None:
        patched_adapter.replace_text_in_history.return_value = make_fr(
            files_affected=["a.py", "b.py", "c.py"], dry_run=True,
        )
        result = await _execute_tool("replace_text_in_history", {
            "repo_path": "/tmp/repo", "old_text": "old_value",
            "new_text": "new_value", "dry_run": True,
        })
        assert result["success"] and result["dry_run"]

    async def test_empty_old_text_propagates(self, patched_adapter: MagicMock) -> None:
        patched_adapter.replace_text_in_history.return_value = make_fr(
            success=False, message="old_text must not be empty",
        )
        result = await _execute_tool("replace_text_in_history", {
            "repo_path": "/tmp/repo", "old_text": "", "new_text": "x", "dry_run": True,
        })
        assert result["success"] is False


class TestChangeDatesHandler:
    @pytest.mark.parametrize(
        "extra,expected_call_args",
        [
            ({"time_range": "evening"},
             ("evening", False, True, None)),
            ({"time_range": "random", "weekend_only": True},
             ("random", True, True, None)),
        ],
        ids=["evening", "weekend_only"],
    )
    async def test_forwards_args(
        self, patched_adapter: MagicMock,
        extra: dict, expected_call_args: tuple,
    ) -> None:
        patched_adapter.change_commit_dates.return_value = make_fr(
            commits_rewritten=3, dry_run=True,
        )
        await _execute_tool("change_commit_dates", {
            "repo_path": "/tmp/repo", "dry_run": True, **extra,
        })
        patched_adapter.change_commit_dates.assert_called_once_with(
            *expected_call_args, dry_run=True, force=False,
        )


# =========================================================================
# rewrite_commit_messages — combine manual_mappings / AI / errors
# =========================================================================


class TestRewriteCommitMessages:
    """``rewrite_commit_messages`` has the most branches; group them here."""

    async def test_manual_mappings_dry_run(self, patched_adapter: MagicMock) -> None:
        patched_adapter.rewrite_commit_messages.return_value = make_fr(
            commits_processed=5, dry_run=True,
        )
        result = await _execute_tool("rewrite_commit_messages", {
            "repo_path": "/tmp/repo", "use_ai": False,
            "manual_mappings": {"old": "new"}, "dry_run": True,
        })
        assert result["success"] and result["dry_run"]

    async def test_no_ai_no_mappings_rejected(self, patched_adapter: MagicMock) -> None:
        result = await _execute_tool("rewrite_commit_messages", {
            "repo_path": "/tmp/repo", "use_ai": False,
        })
        assert result["success"] is False
        assert result["error_code"] == ErrorCode.INVALID_INPUT
        assert "manual_mappings" in result["error"]

    async def test_use_ai_with_manual_mappings_rejected(
        self, patched_adapter: MagicMock,
    ) -> None:
        result = await _execute_tool("rewrite_commit_messages", {
            "repo_path": "/tmp/repo", "use_ai": True,
            "manual_mappings": {"a": "b"}, "dry_run": True,
        })
        assert result["success"] is False
        assert result["error_code"] == ErrorCode.INVALID_INPUT
        assert "manual_mappings" in result["error"]

    async def test_manual_mappings_real_creates_backup(
        self, patched_adapter: MagicMock,
    ) -> None:
        with patch("git_filter_repo_mcp.server.get_config") as mock_config:
            mock_config.return_value.server.auto_backup = True
            patched_adapter.create_backup.return_value = "backup_test"
            patched_adapter.rewrite_commit_messages.return_value = make_fr(
                commits_processed=3, commits_rewritten=1,
            )
            result = await _execute_tool("rewrite_commit_messages", {
                "repo_path": "/tmp/repo", "use_ai": False,
                "manual_mappings": {"old": "new"}, "dry_run": False,
            })
        assert result["backup_branch"] == "backup_test"
        patched_adapter.create_backup.assert_called_once()

    async def test_manual_mappings_dry_run_skips_backup(
        self, patched_adapter: MagicMock,
    ) -> None:
        with patch("git_filter_repo_mcp.server.get_config") as mock_config:
            mock_config.return_value.server.auto_backup = True
            patched_adapter.rewrite_commit_messages.return_value = make_fr(dry_run=True)
            result = await _execute_tool("rewrite_commit_messages", {
                "repo_path": "/tmp/repo", "use_ai": False,
                "manual_mappings": {"old": "new"}, "dry_run": True,
            })
        assert "backup_branch" not in result
        patched_adapter.create_backup.assert_not_called()

    async def test_invalid_style_rejected_by_pydantic(self) -> None:
        result = await _execute_tool("rewrite_commit_messages", {
            "repo_path": "/tmp/repo", "use_ai": True, "style": "nonexistent",
        })
        assert result["success"] is False
        assert result["error_code"] == ErrorCode.INVALID_INPUT

    async def test_ai_rewrite_uses_bulk_collect_commit_files(
        self, patched_adapter: MagicMock,
    ) -> None:
        """Regression: the AI flow must call ``collect_commit_files`` once,
        not ``get_commit_files`` per commit."""
        mock_commits = [
            make_ci("aaa111", "msg1"),
            make_ci("bbb222", "msg2"),
        ]
        patched_adapter.get_commits.return_value = mock_commits
        patched_adapter.collect_commit_files.return_value = {
            "aaa111": ["file1.py"], "bbb222": ["file2.py"],
        }

        mock_engine = MagicMock()
        mock_engine.rewrite_batch = AsyncMock(return_value=[
            MagicMock(original="msg1", rewritten="feat: msg1", commit_hash="aaa111"),
            MagicMock(original="msg2", rewritten="feat: msg2", commit_hash="bbb222"),
        ])
        mock_engine.close = AsyncMock()

        with patch("git_filter_repo_mcp.server._create_ai_provider") as mock_create, \
             patch("git_filter_repo_mcp.server._check_ai_connection",
                   new_callable=AsyncMock, return_value=None), \
             patch("git_filter_repo_mcp.server.AICommitEngine", return_value=mock_engine):
            mock_create.return_value = MagicMock(close=AsyncMock())
            await _execute_tool("rewrite_commit_messages", {
                "repo_path": "/tmp/repo", "use_ai": True,
                "ai_provider": "ollama", "dry_run": True,
            })

        patched_adapter.collect_commit_files.assert_called_once_with(
            mock_commits, "HEAD", 2,
        )
        patched_adapter.get_commit_files.assert_not_called()


# =========================================================================
# rewrite_single_commit — combine all of its branches
# =========================================================================


class TestRewriteSingleCommit:
    """``rewrite_single_commit`` has many branches; one class covers them all."""

    @pytest.fixture
    def adapter_with_commit(self, patched_adapter: MagicMock) -> MagicMock:
        """Adapter pre-loaded with one commit, ready for happy-path tests."""
        patched_adapter.get_commits.return_value = [make_ci("abc123def456", "Original message")]
        patched_adapter._validate_commit_hash = MagicMock()
        return patched_adapter

    async def test_no_changes_returns_no_changes_error(
        self, adapter_with_commit: MagicMock,
    ) -> None:
        result = await _execute_tool("rewrite_single_commit", {
            "repo_path": "/tmp/repo", "commit_hash": "abc123", "dry_run": False,
        })
        assert result["success"] is False
        assert "No changes specified" in result["error"]
        assert result["error_code"] == ErrorCode.NO_CHANGES

    async def test_dry_run_returns_preview(
        self, adapter_with_commit: MagicMock,
    ) -> None:
        result = await _execute_tool("rewrite_single_commit", {
            "repo_path": "/tmp/repo", "commit_hash": "abc123",
            "new_message": "New message", "dry_run": True,
        })
        assert result["success"] and result["dry_run"]
        assert result["new_message"] == "New message"

    async def test_commit_not_found(self, patched_adapter: MagicMock) -> None:
        patched_adapter.get_commits.return_value = []
        patched_adapter._validate_commit_hash = MagicMock()
        result = await _execute_tool("rewrite_single_commit", {
            "repo_path": "/tmp/repo", "commit_hash": "nonexistent",
            "new_message": "New message",
        })
        assert result["success"] is False
        assert "not found" in result["error"]

    async def test_message_change_forwards_to_adapter(
        self, adapter_with_commit: MagicMock,
    ) -> None:
        adapter_with_commit.rewrite_single_commit.return_value = make_fr(
            message="Updated abc123de: message", commits_rewritten=1,
        )
        await _execute_tool("rewrite_single_commit", {
            "repo_path": "/tmp/repo", "commit_hash": "abc123",
            "new_message": "New message", "dry_run": False,
        })
        adapter_with_commit.rewrite_single_commit.assert_called_once_with(
            "abc123",
            new_message="New message",
            new_author_name=None,
            new_author_email=None,
        )

    @pytest.mark.parametrize(
        "extra,missing_field",
        [
            ({"new_author_name": "New Name"}, "new_author_email"),
            ({"new_author_email": "new@example.com"}, "new_author_name"),
        ],
        ids=["only_name", "only_email"],
    )
    async def test_partial_author_rejected(
        self, adapter_with_commit: MagicMock,
        extra: dict, missing_field: str,
    ) -> None:
        result = await _execute_tool("rewrite_single_commit", {
            "repo_path": "/tmp/repo", "commit_hash": "abc123",
            "dry_run": False, **extra,
        })
        assert result["success"] is False
        assert missing_field in result["error"]

    async def test_both_author_fields_accepted(
        self, adapter_with_commit: MagicMock,
    ) -> None:
        adapter_with_commit.rewrite_single_commit.return_value = make_fr(
            message="Updated abc123: author",
        )
        result = await _execute_tool("rewrite_single_commit", {
            "repo_path": "/tmp/repo", "commit_hash": "abc123",
            "new_author_name": "New Name", "new_author_email": "new@example.com",
            "dry_run": False,
        })
        assert result["success"] is True

    async def test_injection_hash_rejected_early(
        self, patched_adapter: MagicMock,
    ) -> None:
        patched_adapter._validate_commit_hash.side_effect = ValueError("Invalid commit hash")
        result = await _execute_tool("rewrite_single_commit", {
            "repo_path": "/tmp/repo", "commit_hash": '"; evil code',
            "new_message": "test", "dry_run": False,
        })
        assert result["success"] is False
        assert result["error_code"] == ErrorCode.INVALID_INPUT


# =========================================================================
# Cross-cutting: auto-backup, AI provider plumbing, error envelopes, main()
# =========================================================================


class TestAutoBackup:
    """Auto-backup must run before destructive ops, and never on dry_run."""

    async def test_backup_runs_before_destructive_op(
        self, patched_adapter: MagicMock,
    ) -> None:
        call_order: list[str] = []
        with patch("git_filter_repo_mcp.server.get_config") as mock_config:
            mock_config.return_value.server.auto_backup = True
            patched_adapter.create_backup.side_effect = lambda: call_order.append("backup") or "backup_test"
            patched_adapter.change_author.side_effect = (
                lambda *a, **kw: call_order.append("change_author") or make_fr(message="done")
            )
            await _execute_tool("change_author", {
                "repo_path": "/tmp/repo", "old_email": "a@b",
                "new_name": "N", "new_email": "n@b", "dry_run": False,
            })
        assert call_order == ["backup", "change_author"]

    async def test_dry_run_skips_backup(self, patched_adapter: MagicMock) -> None:
        with patch("git_filter_repo_mcp.server.get_config") as mock_config:
            mock_config.return_value.server.auto_backup = True
            patched_adapter.change_author.return_value = make_fr(dry_run=True)
            result = await _execute_tool("change_author", {
                "repo_path": "/tmp/repo", "old_email": "a@b",
                "new_name": "N", "new_email": "n@b", "dry_run": True,
            })
        assert "backup_branch" not in result
        patched_adapter.create_backup.assert_not_called()


class TestAIProviderPlumbing:
    """``_create_ai_provider`` reads config lazily and forwards correctly."""

    def test_create_ai_provider_reads_config(self) -> None:
        from git_filter_repo_mcp.server import _create_ai_provider
        with patch("git_filter_repo_mcp.server.get_config") as mock_gc, \
             patch("git_filter_repo_mcp.server.get_provider"):
            mock_gc.return_value.ai.model = "test-model"
            mock_gc.return_value.ai.ollama_base_url = "http://localhost:11434"
            _create_ai_provider({}, "ollama")
            mock_gc.assert_called()

    def test_openai_base_url_forwarded(self) -> None:
        from git_filter_repo_mcp.server import _create_ai_provider
        with patch("git_filter_repo_mcp.server.get_config") as mock_gc, \
             patch("git_filter_repo_mcp.server.get_provider") as mock_gp:
            mock_gc.return_value.ai.model = "gpt-4o"
            mock_gc.return_value.ai.openai_api_key = "sk-test"
            mock_gc.return_value.ai.openai_base_url = "https://custom.api.com/v1"
            _create_ai_provider({}, "openai")
            mock_gp.assert_called_once_with(
                "openai", model="gpt-4o", api_key="sk-test",
                base_url="https://custom.api.com/v1",
            )

    def test_rejects_none_provider(self) -> None:
        from git_filter_repo_mcp.server import _create_ai_provider
        with pytest.raises(ValueError, match="Invalid AI provider"):
            _create_ai_provider({}, "none")

    async def test_invalid_provider_at_pydantic_layer(
        self, patched_adapter: MagicMock,
    ) -> None:
        patched_adapter.get_commits.return_value = [make_ci("abc123")]
        patched_adapter.collect_commit_files.return_value = {"abc123": []}
        result = await _execute_tool("rewrite_commit_messages", {
            "repo_path": "/tmp/repo", "use_ai": True,
            "ai_provider": "nonexistent", "dry_run": True,
        })
        assert result["success"] is False
        # Could match either pydantic's Literal message or the legacy check.
        assert (
            "Invalid AI provider" in result["error"]
            or "nonexistent" in result["error"]
            or "ai_provider" in result["error"]
        )

    @pytest.mark.parametrize(
        "tool,extra_args",
        [
            ("rewrite_commit_messages", {"dry_run": True}),
            ("rewrite_single_commit",
             {"commit_hash": "abc123", "dry_run": False}),
        ],
    )
    async def test_provider_none_in_config_rejected(
        self, patched_adapter: MagicMock, tool: str, extra_args: dict,
    ) -> None:
        patched_adapter.get_commits.return_value = [make_ci("abc123")]
        patched_adapter._validate_commit_hash = MagicMock()
        with patch("git_filter_repo_mcp.server.get_config") as mock_config:
            mock_config.return_value.ai.provider = "none"
            result = await _execute_tool(tool, {
                "repo_path": "/tmp/repo", "use_ai": True, **extra_args,
            })
        assert result["success"] is False
        assert "none" in result["error"].lower()


class TestErrorEnvelope:
    """Adapter/subprocess errors must become structured handler responses."""

    async def test_timeout(self) -> None:
        with patch("git_filter_repo_mcp.server.GitFilterRepoAdapter",
                   side_effect=sp.TimeoutExpired("git", 30)):
            result = await _execute_tool("analyze_git_history", {"repo_path": "/tmp/repo"})
        assert result["success"] is False
        assert "timed out" in result["error"]
        assert result["error_code"] == ErrorCode.COMMAND_FAILED

    async def test_called_process_error(self) -> None:
        with patch("git_filter_repo_mcp.server.GitFilterRepoAdapter",
                   side_effect=sp.CalledProcessError(128, ["git", "log"], stderr="fatal")):
            result = await _execute_tool("analyze_git_history", {"repo_path": "/tmp/repo"})
        assert result["success"] is False
        assert result["error_code"] == ErrorCode.COMMAND_FAILED
        assert "Command failed" in result["error"]
        assert "Internal error" not in result["error"]


class TestServerLifecycle:
    """``main()`` and logging helpers."""

    def test_main_keyboard_interrupt_exits_cleanly(self) -> None:
        with patch(
            "git_filter_repo_mcp.server.asyncio.run",
            side_effect=KeyboardInterrupt,
        ):
            from git_filter_repo_mcp.server import main
            main()  # must not raise SystemExit

    def test_main_fatal_error_propagates(self) -> None:
        with patch(
            "git_filter_repo_mcp.server.asyncio.run",
            side_effect=RuntimeError("boom"),
        ):
            from git_filter_repo_mcp.server import main
            with pytest.raises(SystemExit) as excinfo:
                main()
        assert excinfo.value.code == 1

    def test_configure_logging_callable(self) -> None:
        from git_filter_repo_mcp.server import _configure_logging
        _configure_logging()  # must not raise

    def test_import_does_not_reconfigure_root_logger(self) -> None:
        """Importing the server module must NOT call ``logging.basicConfig``
        — that would clobber the host application's logger setup. The
        configuration must only run inside ``main()``."""
        import importlib
        import logging

        root = logging.getLogger()
        original_level = root.level

        # Force re-import to trigger any import-time side effects.
        import git_filter_repo_mcp.server as server_mod
        importlib.reload(server_mod)

        # The level must not have changed from import alone.
        assert root.level == original_level
