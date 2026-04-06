"""Tests for MCP server."""

import json
from unittest.mock import MagicMock, patch

import pytest

from git_filter_repo_mcp.adapter import CommitInfo, FilterResult
from git_filter_repo_mcp.server import (
    _execute_tool,
    call_tool,
    list_tools,
    result_to_dict,
)
from git_filter_repo_mcp.tools import ErrorCode


class TestResultToDict:
    """Test result_to_dict conversion."""

    def test_success_result(self):
        result = FilterResult(
            success=True,
            message="Operation completed",
            commits_processed=10,
            commits_rewritten=5,
            files_affected=["a.py", "b.py"],
            dry_run=False,
            error=None,
        )
        d = result_to_dict(result)
        assert d["success"] is True
        assert d["message"] == "Operation completed"
        assert d["commits_processed"] == 10
        assert d["commits_rewritten"] == 5
        assert d["files_affected"] == ["a.py", "b.py"]
        assert d["dry_run"] is False
        assert d["error"] is None

    def test_error_result(self):
        result = FilterResult(
            success=False,
            message="",
            commits_processed=0,
            commits_rewritten=0,
            files_affected=[],
            dry_run=False,
            error="Something went wrong",
        )
        d = result_to_dict(result)
        assert d["success"] is False
        assert d["error"] == "Something went wrong"

    def test_dry_run_result(self):
        result = FilterResult(
            success=True,
            message="Dry run completed",
            commits_processed=5,
            commits_rewritten=0,
            files_affected=[],
            dry_run=True,
            error=None,
        )
        d = result_to_dict(result)
        assert d["dry_run"] is True


class TestListTools:
    """Test list_tools handler."""

    @pytest.mark.asyncio
    async def test_list_tools_returns_tools(self):
        tools = await list_tools()
        assert len(tools) > 0

        # Check that essential tools are present
        tool_names = [t.name for t in tools]
        assert "analyze_git_history" in tool_names
        assert "rewrite_commit_messages" in tool_names
        assert "change_author" in tool_names
        assert "remove_files_from_history" in tool_names
        assert "create_backup" in tool_names

    @pytest.mark.asyncio
    async def test_tools_have_required_fields(self):
        tools = await list_tools()
        for tool in tools:
            assert tool.name is not None
            assert tool.description is not None
            assert tool.inputSchema is not None


class TestCallTool:
    """Test call_tool handler."""

    @pytest.mark.asyncio
    async def test_call_tool_returns_text_content(self):
        with patch("git_filter_repo_mcp.server._execute_tool") as mock_execute:
            mock_execute.return_value = {"success": True, "message": "Done"}

            result = await call_tool("analyze_git_history", {"repo_path": "/tmp/repo"})

            assert len(result) == 1
            assert result[0].type == "text"
            data = json.loads(result[0].text)
            assert data["success"] is True

    @pytest.mark.asyncio
    async def test_call_tool_handles_errors(self):
        with patch("git_filter_repo_mcp.server._execute_tool") as mock_execute:
            mock_execute.side_effect = ValueError("Test error")

            result = await call_tool("analyze_git_history", {"repo_path": "/tmp/repo"})

            assert len(result) == 1
            data = json.loads(result[0].text)
            assert data["success"] is False
            assert "Test error" in data["error"]


class TestExecuteTool:
    """Test _execute_tool function."""

    @pytest.mark.asyncio
    async def test_unknown_tool(self):
        result = await _execute_tool("unknown_tool", {})
        assert "error" in result
        assert "Unknown tool" in result["error"]
        assert result["error_code"] == ErrorCode.TOOL_NOT_FOUND

    @pytest.mark.asyncio
    async def test_analyze_git_history(self):
        with patch("git_filter_repo_mcp.server.GitFilterRepoAdapter") as MockAdapter:
            mock_adapter = MagicMock()
            mock_adapter.analyze_history.return_value = {
                "total_commits": 10,
                "authors": ["test@example.com"],
                "branches": ["main"],
            }
            MockAdapter.return_value = mock_adapter

            result = await _execute_tool(
                "analyze_git_history",
                {
                    "repo_path": "/tmp/repo",
                    "branch": "main",
                    "max_count": 50,
                },
            )

            assert result["success"] is True
            assert result["total_commits"] == 10
            mock_adapter.analyze_history.assert_called_once_with("main", 50)

    @pytest.mark.asyncio
    async def test_create_backup(self):
        with patch("git_filter_repo_mcp.server.GitFilterRepoAdapter") as MockAdapter:
            mock_adapter = MagicMock()
            mock_adapter.create_backup.return_value = "backup-20241209-123456"
            MockAdapter.return_value = mock_adapter

            result = await _execute_tool("create_backup", {"repo_path": "/tmp/repo"})

            assert result["success"] is True
            assert result["backup_branch"] == "backup-20241209-123456"
            mock_adapter.create_backup.assert_called_once()

    @pytest.mark.asyncio
    async def test_restore_backup(self):
        with patch("git_filter_repo_mcp.server.GitFilterRepoAdapter") as MockAdapter:
            mock_adapter = MagicMock()
            mock_adapter.restore_backup.return_value = FilterResult(
                success=True,
                message="Restored from backup",
                commits_processed=0,
                commits_rewritten=0,
                files_affected=[],
                dry_run=False,
                error=None,
            )
            MockAdapter.return_value = mock_adapter

            result = await _execute_tool(
                "restore_backup",
                {
                    "repo_path": "/tmp/repo",
                    "backup_branch": "backup-20241209-123456",
                },
            )

            assert result["success"] is True
            mock_adapter.restore_backup.assert_called_once_with("backup-20241209-123456")

    @pytest.mark.asyncio
    async def test_change_author_dry_run(self):
        with patch("git_filter_repo_mcp.server.GitFilterRepoAdapter") as MockAdapter:
            mock_adapter = MagicMock()
            mock_adapter.change_author.return_value = FilterResult(
                success=True,
                message="Would change 5 commits",
                commits_processed=5,
                commits_rewritten=0,
                files_affected=[],
                dry_run=True,
                error=None,
            )
            MockAdapter.return_value = mock_adapter

            result = await _execute_tool(
                "change_author",
                {
                    "repo_path": "/tmp/repo",
                    "old_email": "old@example.com",
                    "new_name": "New Name",
                    "new_email": "new@example.com",
                    "dry_run": True,
                },
            )

            assert result["success"] is True
            assert result["dry_run"] is True
            mock_adapter.change_author.assert_called_once_with(
                "old@example.com", "New Name", "new@example.com", True, False
            )

    @pytest.mark.asyncio
    async def test_remove_files_from_history(self):
        with patch("git_filter_repo_mcp.server.GitFilterRepoAdapter") as MockAdapter:
            mock_adapter = MagicMock()
            mock_adapter.remove_files.return_value = FilterResult(
                success=True,
                message="Removed files",
                commits_processed=10,
                commits_rewritten=3,
                files_affected=["secret.txt"],
                dry_run=False,
                error=None,
            )
            MockAdapter.return_value = mock_adapter

            result = await _execute_tool(
                "remove_files_from_history",
                {
                    "repo_path": "/tmp/repo",
                    "paths": ["secret.txt", "config.json"],
                    "dry_run": False,
                },
            )

            assert result["success"] is True
            assert result["files_affected"] == ["secret.txt"]

    @pytest.mark.asyncio
    async def test_get_commit_details(self):
        with patch("git_filter_repo_mcp.server.GitFilterRepoAdapter") as MockAdapter:
            from git_filter_repo_mcp.adapter import CommitInfo

            mock_adapter = MagicMock()
            mock_adapter.get_commits.return_value = [
                CommitInfo(
                    hash="abc123def456",
                    author_name="Test User",
                    author_email="test@example.com",
                    committer_name="Test User",
                    committer_email="test@example.com",
                    message="Test commit",
                    date="2024-12-09",
                )
            ]
            mock_adapter.get_commit_files.return_value = ["file1.py", "file2.py"]
            mock_adapter.get_commit_diff.return_value = "+ added line\n- removed line"
            MockAdapter.return_value = mock_adapter

            result = await _execute_tool(
                "get_commit_details",
                {
                    "repo_path": "/tmp/repo",
                    "commit_hash": "abc123",
                },
            )

            assert result["success"] is True
            assert result["commit"]["hash"] == "abc123def456"
            assert result["commit"]["author_name"] == "Test User"
            assert result["commit"]["files"] == ["file1.py", "file2.py"]

    @pytest.mark.asyncio
    async def test_get_commit_details_not_found(self):
        with patch("git_filter_repo_mcp.server.GitFilterRepoAdapter") as MockAdapter:
            mock_adapter = MagicMock()
            mock_adapter.get_commits.return_value = []
            MockAdapter.return_value = mock_adapter

            result = await _execute_tool(
                "get_commit_details",
                {
                    "repo_path": "/tmp/repo",
                    "commit_hash": "nonexistent",
                },
            )

            assert "error" in result
            assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_scan_secrets(self):
        with patch("git_filter_repo_mcp.server.GitFilterRepoAdapter") as MockAdapter:
            mock_adapter = MagicMock()
            mock_adapter.scan_secrets.return_value = {
                "findings": [],
                "files_scanned": 50,
                "commits_scanned": 10,
            }
            MockAdapter.return_value = mock_adapter

            result = await _execute_tool(
                "scan_secrets",
                {
                    "repo_path": "/tmp/repo",
                    "branch": "main",
                    "max_commits": 50,
                },
            )

            assert result["success"] is True
            assert result["findings"] == []
            mock_adapter.scan_secrets.assert_called_once_with("main", 50)

    @pytest.mark.asyncio
    async def test_list_all_files_in_history(self):
        with patch("git_filter_repo_mcp.server.GitFilterRepoAdapter") as MockAdapter:
            mock_adapter = MagicMock()
            mock_adapter.list_all_files_in_history.return_value = [
                "file1.py",
                "file2.py",
                "dir/file3.py",
            ]
            MockAdapter.return_value = mock_adapter

            result = await _execute_tool(
                "list_all_files_in_history",
                {
                    "repo_path": "/tmp/repo",
                },
            )

            assert result["success"] is True
            assert result["total_files"] == 3
            assert "file1.py" in result["files"]

    @pytest.mark.asyncio
    async def test_get_file_history(self):
        with patch("git_filter_repo_mcp.server.GitFilterRepoAdapter") as MockAdapter:
            mock_adapter = MagicMock()
            mock_adapter.get_file_history.return_value = [
                {"hash": "abc123", "message": "Add file"},
                {"hash": "def456", "message": "Update file"},
            ]
            MockAdapter.return_value = mock_adapter

            result = await _execute_tool(
                "get_file_history",
                {
                    "repo_path": "/tmp/repo",
                    "file_path": "src/main.py",
                },
            )

            assert result["success"] is True
            assert result["total_commits"] == 2
            mock_adapter.get_file_history.assert_called_once_with("src/main.py")


class TestRewriteCommitMessages:
    """Test rewrite_commit_messages tool."""

    @pytest.mark.asyncio
    async def test_manual_mappings_dry_run(self):
        with patch("git_filter_repo_mcp.server.GitFilterRepoAdapter") as MockAdapter:
            mock_adapter = MagicMock()
            mock_adapter.rewrite_commit_messages.return_value = FilterResult(
                success=True,
                message="Would rewrite 2 commits",
                commits_processed=5,
                commits_rewritten=0,
                files_affected=[],
                dry_run=True,
                error=None,
            )
            MockAdapter.return_value = mock_adapter

            result = await _execute_tool(
                "rewrite_commit_messages",
                {
                    "repo_path": "/tmp/repo",
                    "use_ai": False,
                    "manual_mappings": {
                        "old message 1": "new message 1",
                        "old message 2": "new message 2",
                    },
                    "dry_run": True,
                },
            )

            assert result["success"] is True
            assert result["dry_run"] is True

    @pytest.mark.asyncio
    async def test_no_ai_no_mappings_error(self):
        with patch("git_filter_repo_mcp.server.GitFilterRepoAdapter") as MockAdapter:
            MockAdapter.return_value = MagicMock()

            result = await _execute_tool(
                "rewrite_commit_messages",
                {
                    "repo_path": "/tmp/repo",
                    "use_ai": False,
                },
            )

            assert "error" in result
            assert "manual_mappings" in result["error"]
            assert result["error_code"] == ErrorCode.INVALID_INPUT


class TestValidationErrorHandling:
    """Test that Pydantic validation errors return useful messages."""

    @pytest.mark.asyncio
    async def test_missing_required_field(self):
        result = await _execute_tool("analyze_git_history", {})
        assert result["success"] is False
        assert "Invalid input" in result["error"]
        assert "repo_path" in result["error"]
        assert result["error_code"] == ErrorCode.INVALID_INPUT

    @pytest.mark.asyncio
    async def test_wrong_type(self):
        result = await _execute_tool(
            "analyze_git_history",
            {"repo_path": "/tmp/repo", "max_count": "not_a_number"},
        )
        assert result["success"] is False
        assert "Invalid input" in result["error"]
        assert result["error_code"] == ErrorCode.INVALID_INPUT

    @pytest.mark.asyncio
    async def test_missing_required_field_change_author(self):
        result = await _execute_tool(
            "change_author",
            {"repo_path": "/tmp/repo"},
        )
        assert result["success"] is False
        assert "Invalid input" in result["error"]
        assert result["error_code"] == ErrorCode.INVALID_INPUT


class TestRewriteSingleCommit:
    """Test rewrite_single_commit tool."""

    @pytest.mark.asyncio
    async def test_no_changes_returns_error(self):
        with patch("git_filter_repo_mcp.server.GitFilterRepoAdapter") as MockAdapter:
            mock_adapter = MagicMock()
            mock_adapter.get_commits.return_value = [
                CommitInfo(
                    hash="abc123def456",
                    author_name="Test",
                    author_email="test@example.com",
                    committer_name="Test",
                    committer_email="test@example.com",
                    message="Original message",
                    date="2024-12-09",
                )
            ]
            MockAdapter.return_value = mock_adapter

            result = await _execute_tool(
                "rewrite_single_commit",
                {
                    "repo_path": "/tmp/repo",
                    "commit_hash": "abc123",
                    "dry_run": False,
                },
            )

            assert result["success"] is False
            assert "No changes specified" in result["error"]
            assert result["error_code"] == ErrorCode.NO_CHANGES

    @pytest.mark.asyncio
    async def test_dry_run_shows_changes(self):
        with patch("git_filter_repo_mcp.server.GitFilterRepoAdapter") as MockAdapter:
            mock_adapter = MagicMock()
            mock_adapter.get_commits.return_value = [
                CommitInfo(
                    hash="abc123def456",
                    author_name="Test",
                    author_email="test@example.com",
                    committer_name="Test",
                    committer_email="test@example.com",
                    message="Original message",
                    date="2024-12-09",
                )
            ]
            MockAdapter.return_value = mock_adapter

            result = await _execute_tool(
                "rewrite_single_commit",
                {
                    "repo_path": "/tmp/repo",
                    "commit_hash": "abc123",
                    "new_message": "New message",
                    "dry_run": True,
                },
            )

            assert result["success"] is True
            assert result["dry_run"] is True
            assert result["new_message"] == "New message"

    @pytest.mark.asyncio
    async def test_commit_not_found(self):
        with patch("git_filter_repo_mcp.server.GitFilterRepoAdapter") as MockAdapter:
            mock_adapter = MagicMock()
            mock_adapter.get_commits.return_value = []
            MockAdapter.return_value = mock_adapter

            result = await _execute_tool(
                "rewrite_single_commit",
                {
                    "repo_path": "/tmp/repo",
                    "commit_hash": "nonexistent",
                    "new_message": "New message",
                },
            )

            assert result["success"] is False
            assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_message_change_calls_adapter(self):
        with patch("git_filter_repo_mcp.server.GitFilterRepoAdapter") as MockAdapter:
            mock_adapter = MagicMock()
            mock_adapter.get_commits.return_value = [
                CommitInfo(
                    hash="abc123def456",
                    author_name="Test",
                    author_email="test@example.com",
                    committer_name="Test",
                    committer_email="test@example.com",
                    message="Old message",
                    date="2024-12-09",
                )
            ]
            mock_adapter.rewrite_single_commit.return_value = FilterResult(
                success=True,
                message="Updated commit abc123de: message",
                commits_rewritten=1,
            )
            MockAdapter.return_value = mock_adapter

            result = await _execute_tool(
                "rewrite_single_commit",
                {
                    "repo_path": "/tmp/repo",
                    "commit_hash": "abc123",
                    "new_message": "New message",
                    "dry_run": False,
                },
            )

            assert result["success"] is True
            mock_adapter.rewrite_single_commit.assert_called_once_with(
                "abc123",
                new_message="New message",
                new_author_name=None,
                new_author_email=None,
            )


class TestFilterPaths:
    """Test filter_paths tool."""

    @pytest.mark.asyncio
    async def test_include_exclude_together_rejected(self):
        with patch("git_filter_repo_mcp.server.GitFilterRepoAdapter") as MockAdapter:
            mock_adapter = MagicMock()
            mock_adapter.filter_paths.return_value = FilterResult(
                success=False,
                message="Cannot use include_paths and exclude_paths together",
                error="git-filter-repo's --invert-paths is global",
            )
            MockAdapter.return_value = mock_adapter

            result = await _execute_tool(
                "filter_paths",
                {
                    "repo_path": "/tmp/repo",
                    "include_paths": ["src/"],
                    "exclude_paths": ["tests/"],
                    "dry_run": True,
                },
            )

            assert result["success"] is False

    @pytest.mark.asyncio
    async def test_replace_text_dry_run(self):
        with patch("git_filter_repo_mcp.server.GitFilterRepoAdapter") as MockAdapter:
            mock_adapter = MagicMock()
            mock_adapter.replace_text_in_history.return_value = FilterResult(
                success=True,
                message="Dry run: 3 files in history",
                files_affected=["a.py", "b.py", "c.py"],
                dry_run=True,
            )
            mock_adapter.create_backup.return_value = "backup_123"
            MockAdapter.return_value = mock_adapter

            result = await _execute_tool(
                "replace_text_in_history",
                {
                    "repo_path": "/tmp/repo",
                    "old_text": "old_value",
                    "new_text": "new_value",
                    "dry_run": True,
                },
            )

            assert result["success"] is True
            assert result["dry_run"] is True


class TestRemoveLargeFilesHandler:
    """Test remove_large_files handler."""

    @pytest.mark.asyncio
    async def test_dry_run(self):
        with patch("git_filter_repo_mcp.server.GitFilterRepoAdapter") as MockAdapter:
            mock_adapter = MagicMock()
            mock_adapter.remove_large_files.return_value = FilterResult(
                success=True,
                message="Found 2 files larger than 10.0 MB",
                commits_processed=5,
                commits_rewritten=0,
                files_affected=["big.bin", "huge.zip"],
                dry_run=True,
                error=None,
            )
            MockAdapter.return_value = mock_adapter

            result = await _execute_tool(
                "remove_large_files",
                {"repo_path": "/tmp/repo", "size_threshold_mb": 10.0, "dry_run": True},
            )

            assert result["success"] is True
            assert result["dry_run"] is True
            assert result["files_affected"] == ["big.bin", "huge.zip"]
            mock_adapter.remove_large_files.assert_called_once_with(10.0, True, False)


class TestSquashCommitsHandler:
    """Test squash_commits handler."""

    @pytest.mark.asyncio
    async def test_dry_run(self):
        with patch("git_filter_repo_mcp.server.GitFilterRepoAdapter") as MockAdapter:
            mock_adapter = MagicMock()
            mock_adapter.squash_commits.return_value = FilterResult(
                success=True,
                message="Would squash 3 commits",
                commits_processed=3,
                commits_rewritten=0,
                files_affected=[],
                dry_run=True,
                error=None,
            )
            MockAdapter.return_value = mock_adapter

            result = await _execute_tool(
                "squash_commits",
                {
                    "repo_path": "/tmp/repo",
                    "start_commit": "abc123",
                    "end_commit": "HEAD",
                    "dry_run": True,
                },
            )

            assert result["success"] is True
            assert result["dry_run"] is True
            assert result["commits_processed"] == 3

    @pytest.mark.asyncio
    async def test_with_auto_backup(self):
        with patch("git_filter_repo_mcp.server.GitFilterRepoAdapter") as MockAdapter, \
             patch("git_filter_repo_mcp.server.get_config") as mock_get_config:
            mock_get_config.return_value.server.auto_backup = True
            mock_adapter = MagicMock()
            mock_adapter.create_backup.return_value = "backup_squash"
            mock_adapter.squash_commits.return_value = FilterResult(
                success=True,
                message="Squashed 3 commits",
                commits_processed=3,
                commits_rewritten=1,
                files_affected=[],
                dry_run=False,
                error=None,
            )
            MockAdapter.return_value = mock_adapter

            result = await _execute_tool(
                "squash_commits",
                {
                    "repo_path": "/tmp/repo",
                    "start_commit": "abc123",
                    "dry_run": False,
                },
            )

            assert result["success"] is True
            assert result["backup_branch"] == "backup_squash"


class TestChangeDatesHandler:
    """Test change_commit_dates handler."""

    @pytest.mark.asyncio
    async def test_dry_run(self):
        with patch("git_filter_repo_mcp.server.GitFilterRepoAdapter") as MockAdapter:
            mock_adapter = MagicMock()
            mock_adapter.change_commit_dates.return_value = FilterResult(
                success=True,
                message="Preview: 5 commits would be changed",
                commits_processed=5,
                commits_rewritten=5,
                files_affected=[],
                dry_run=True,
                error=None,
            )
            MockAdapter.return_value = mock_adapter

            result = await _execute_tool(
                "change_commit_dates",
                {
                    "repo_path": "/tmp/repo",
                    "time_range": "evening",
                    "dry_run": True,
                },
            )

            assert result["success"] is True
            assert result["dry_run"] is True
            assert result["commits_rewritten"] == 5

    @pytest.mark.asyncio
    async def test_weekend_only(self):
        with patch("git_filter_repo_mcp.server.GitFilterRepoAdapter") as MockAdapter:
            mock_adapter = MagicMock()
            mock_adapter.change_commit_dates.return_value = FilterResult(
                success=True,
                message="Preview: weekend dates",
                commits_processed=3,
                commits_rewritten=3,
                files_affected=[],
                dry_run=True,
                error=None,
            )
            MockAdapter.return_value = mock_adapter

            result = await _execute_tool(
                "change_commit_dates",
                {
                    "repo_path": "/tmp/repo",
                    "time_range": "random",
                    "weekend_only": True,
                    "dry_run": True,
                },
            )

            assert result["success"] is True
            mock_adapter.change_commit_dates.assert_called_once_with(
                "random", True, True, None, dry_run=True, force=False,
            )


class TestMainExitCode:
    """Test that main() propagates fatal errors correctly."""

    def test_fatal_error_raises_system_exit(self):
        with patch("git_filter_repo_mcp.server.asyncio.run", side_effect=RuntimeError("boom")):
            with pytest.raises(SystemExit) as exc_info:
                from git_filter_repo_mcp.server import main
                main()
            assert exc_info.value.code == 1

    def test_keyboard_interrupt_exits_cleanly(self):
        with patch("git_filter_repo_mcp.server.asyncio.run", side_effect=KeyboardInterrupt):
            from git_filter_repo_mcp.server import main
            # Should not raise SystemExit
            main()


class TestAIRewriteUsesBulkFiles:
    """Test that AI rewrite mode uses collect_commit_files (bulk)."""

    @pytest.mark.asyncio
    async def test_ai_rewrite_calls_collect_commit_files(self):
        from unittest.mock import AsyncMock

        mock_commits = [
            CommitInfo("aaa111", "User", "u@e.com", "User", "u@e.com", "msg1", "2024-01-01"),
            CommitInfo("bbb222", "User", "u@e.com", "User", "u@e.com", "msg2", "2024-01-02"),
        ]

        with patch("git_filter_repo_mcp.server.GitFilterRepoAdapter") as MockAdapter, \
             patch("git_filter_repo_mcp.server._create_ai_provider") as mock_create, \
             patch("git_filter_repo_mcp.server._check_ai_connection", new_callable=AsyncMock, return_value=None):
            mock_adapter = MagicMock()
            mock_adapter.get_commits.return_value = mock_commits
            mock_adapter.collect_commit_files.return_value = {
                "aaa111": ["file1.py"],
                "bbb222": ["file2.py"],
            }
            MockAdapter.return_value = mock_adapter

            mock_provider = MagicMock()
            mock_provider.generate_message = AsyncMock(side_effect=[
                MagicMock(original="msg1", rewritten="feat: msg1", commit_hash="aaa111"),
                MagicMock(original="msg2", rewritten="feat: msg2", commit_hash="bbb222"),
            ])
            mock_provider.close = AsyncMock()

            mock_engine = MagicMock()
            mock_engine.rewrite_message = AsyncMock(side_effect=[
                MagicMock(original="msg1", rewritten="feat: msg1", commit_hash="aaa111"),
                MagicMock(original="msg2", rewritten="feat: msg2", commit_hash="bbb222"),
            ])
            mock_engine.close = AsyncMock()

            with patch("git_filter_repo_mcp.server.AICommitEngine", return_value=mock_engine):
                mock_create.return_value = mock_provider

                await _execute_tool(
                    "rewrite_commit_messages",
                    {
                        "repo_path": "/tmp/repo",
                        "use_ai": True,
                        "ai_provider": "ollama",
                        "dry_run": True,
                    },
                )

                # Verify bulk method was called, NOT per-commit get_commit_files
                mock_adapter.collect_commit_files.assert_called_once_with(
                    mock_commits, "HEAD", 2,
                )
                mock_adapter.get_commit_files.assert_not_called()


class TestPartialAuthorValidation:
    """Test that partial author info (only name or only email) is rejected."""

    @pytest.mark.asyncio
    async def test_only_name_rejected(self):
        with patch("git_filter_repo_mcp.server.GitFilterRepoAdapter") as MockAdapter:
            mock_adapter = MagicMock()
            mock_adapter.get_commits.return_value = [
                CommitInfo("abc123", "User", "u@e.com", "User", "u@e.com", "msg", "2024-01-01"),
            ]
            mock_adapter._validate_commit_hash = MagicMock()
            MockAdapter.return_value = mock_adapter

            result = await _execute_tool(
                "rewrite_single_commit",
                {
                    "repo_path": "/tmp/repo",
                    "commit_hash": "abc123",
                    "new_author_name": "New Name",
                    # new_author_email is missing
                    "dry_run": False,
                },
            )
            assert result["success"] is False
            assert "new_author_email" in result["error"]

    @pytest.mark.asyncio
    async def test_only_email_rejected(self):
        with patch("git_filter_repo_mcp.server.GitFilterRepoAdapter") as MockAdapter:
            mock_adapter = MagicMock()
            mock_adapter.get_commits.return_value = [
                CommitInfo("abc123", "User", "u@e.com", "User", "u@e.com", "msg", "2024-01-01"),
            ]
            mock_adapter._validate_commit_hash = MagicMock()
            MockAdapter.return_value = mock_adapter

            result = await _execute_tool(
                "rewrite_single_commit",
                {
                    "repo_path": "/tmp/repo",
                    "commit_hash": "abc123",
                    "new_author_email": "new@example.com",
                    # new_author_name is missing
                    "dry_run": False,
                },
            )
            assert result["success"] is False
            assert "new_author_name" in result["error"]

    @pytest.mark.asyncio
    async def test_both_provided_accepted(self):
        with patch("git_filter_repo_mcp.server.GitFilterRepoAdapter") as MockAdapter:
            mock_adapter = MagicMock()
            mock_adapter.get_commits.return_value = [
                CommitInfo("abc123", "User", "u@e.com", "User", "u@e.com", "msg", "2024-01-01"),
            ]
            mock_adapter._validate_commit_hash = MagicMock()
            mock_adapter.rewrite_single_commit.return_value = FilterResult(
                success=True, message="Updated abc123: author",
            )
            MockAdapter.return_value = mock_adapter

            result = await _execute_tool(
                "rewrite_single_commit",
                {
                    "repo_path": "/tmp/repo",
                    "commit_hash": "abc123",
                    "new_author_name": "New Name",
                    "new_author_email": "new@example.com",
                    "dry_run": False,
                },
            )
            assert result["success"] is True


class TestCommitHashValidationInHandler:
    """Test that rewrite_single_commit handler validates commit hash early."""

    @pytest.mark.asyncio
    async def test_invalid_hash_rejected_early(self):
        with patch("git_filter_repo_mcp.server.GitFilterRepoAdapter") as MockAdapter:
            mock_adapter = MagicMock()
            mock_adapter._validate_commit_hash.side_effect = ValueError("Invalid commit hash")
            MockAdapter.return_value = mock_adapter

            result = await _execute_tool(
                "rewrite_single_commit",
                {
                    "repo_path": "/tmp/repo",
                    "commit_hash": '"; evil code',
                    "new_message": "test",
                    "dry_run": False,
                },
            )
            assert result["success"] is False
            assert "INVALID_INPUT" in str(result.get("error_code", ""))


class TestLazyConfig:
    """Test that server uses get_config() lazily (not module-level snapshot)."""

    @pytest.mark.asyncio
    async def test_handler_reads_fresh_config(self):
        """Verify _create_ai_provider calls get_config() each time."""
        from git_filter_repo_mcp.server import _create_ai_provider
        with patch("git_filter_repo_mcp.server.get_config") as mock_gc, \
             patch("git_filter_repo_mcp.server.get_provider"):
            mock_gc.return_value.ai.model = "test-model"
            mock_gc.return_value.ai.ollama_base_url = "http://localhost:11434"
            _create_ai_provider({}, "ollama")
            mock_gc.assert_called()


class TestAIProviderValidation:
    """Test that invalid AI provider names are rejected early."""

    @pytest.mark.asyncio
    async def test_invalid_provider_rejected(self):
        with patch("git_filter_repo_mcp.server.GitFilterRepoAdapter") as MockAdapter:
            mock_adapter = MagicMock()
            mock_adapter.get_commits.return_value = [
                CommitInfo("abc123", "User", "u@e.com", "User", "u@e.com", "msg", "2024-01-01"),
            ]
            mock_adapter.collect_commit_files.return_value = {"abc123": []}
            MockAdapter.return_value = mock_adapter

            result = await _execute_tool(
                "rewrite_commit_messages",
                {
                    "repo_path": "/tmp/repo",
                    "use_ai": True,
                    "ai_provider": "nonexistent",
                    "dry_run": True,
                },
            )
            assert result["success"] is False
            assert "Invalid AI provider" in result["error"] or "nonexistent" in result["error"]

    def test_create_ai_provider_rejects_none(self):
        from git_filter_repo_mcp.server import _create_ai_provider
        with pytest.raises(ValueError, match="Invalid AI provider"):
            _create_ai_provider({}, "none")


class TestOpenAIBaseUrlForwarding:
    """Test that openai_base_url config is forwarded to the provider."""

    def test_openai_base_url_passed(self):
        from git_filter_repo_mcp.server import _create_ai_provider
        with patch("git_filter_repo_mcp.server.get_config") as mock_gc, \
             patch("git_filter_repo_mcp.server.get_provider") as mock_gp:
            mock_gc.return_value.ai.model = "gpt-4o"
            mock_gc.return_value.ai.openai_api_key = "sk-test"
            mock_gc.return_value.ai.openai_base_url = "https://custom.api.com/v1"
            _create_ai_provider({}, "openai")
            mock_gp.assert_called_once_with(
                "openai",
                model="gpt-4o",
                api_key="sk-test",
                base_url="https://custom.api.com/v1",
            )


class TestLoggingReconfigure:
    """Test that logging can be reconfigured."""

    def test_configure_logging_callable(self):
        from git_filter_repo_mcp.server import _configure_logging
        # Should not raise
        _configure_logging()


class TestToolsValidation:
    """Test Pydantic validation constraints on tool inputs."""

    def test_max_count_zero_rejected(self):
        from git_filter_repo_mcp.tools import AnalyzeHistoryInput
        with pytest.raises(Exception):
            AnalyzeHistoryInput(repo_path="/tmp", max_count=0)

    def test_max_count_negative_rejected(self):
        from git_filter_repo_mcp.tools import AnalyzeHistoryInput
        with pytest.raises(Exception):
            AnalyzeHistoryInput(repo_path="/tmp", max_count=-1)

    def test_max_count_over_limit_rejected(self):
        from git_filter_repo_mcp.tools import AnalyzeHistoryInput
        with pytest.raises(Exception):
            AnalyzeHistoryInput(repo_path="/tmp", max_count=20000)

    def test_size_threshold_zero_rejected(self):
        from git_filter_repo_mcp.tools import RemoveLargeFilesInput
        with pytest.raises(Exception):
            RemoveLargeFilesInput(repo_path="/tmp", size_threshold_mb=0.0)

    def test_size_threshold_negative_rejected(self):
        from git_filter_repo_mcp.tools import RemoveLargeFilesInput
        with pytest.raises(Exception):
            RemoveLargeFilesInput(repo_path="/tmp", size_threshold_mb=-5.0)

    def test_use_ai_defaults_false(self):
        from git_filter_repo_mcp.tools import RewriteCommitMessagesInput
        params = RewriteCommitMessagesInput(repo_path="/tmp")
        assert params.use_ai is False

    def test_ai_provider_defaults_none(self):
        from git_filter_repo_mcp.tools import RewriteCommitMessagesInput
        params = RewriteCommitMessagesInput(repo_path="/tmp")
        assert params.ai_provider is None


class TestGetCommitDetailsInjection:
    """Test that get_commit_details rejects dash-prefixed hashes."""

    @pytest.mark.asyncio
    async def test_dash_hash_rejected(self):
        result = await _execute_tool(
            "get_commit_details",
            {"repo_path": "/tmp/repo", "commit_hash": "--exec=evil"},
        )
        assert result["success"] is False
        assert result["error_code"] == ErrorCode.INVALID_INPUT

    @pytest.mark.asyncio
    async def test_normal_hash_accepted(self):
        with patch("git_filter_repo_mcp.server.GitFilterRepoAdapter") as MockAdapter:
            mock_adapter = MagicMock()
            mock_adapter.get_commits.return_value = [
                CommitInfo("abc123", "U", "u@e", "U", "u@e", "msg", "2024-01-01")
            ]
            mock_adapter.get_commit_files.return_value = []
            mock_adapter.get_commit_diff.return_value = ""
            MockAdapter.return_value = mock_adapter

            result = await _execute_tool(
                "get_commit_details",
                {"repo_path": "/tmp/repo", "commit_hash": "abc123"},
            )
            assert result["success"] is True


class TestBackupBeforeDestructiveOps:
    """Test that backup is created BEFORE destructive operations."""

    @pytest.mark.asyncio
    async def test_change_author_backup_before_operation(self):
        """Verify create_backup is called before change_author."""
        call_order = []

        with patch("git_filter_repo_mcp.server.GitFilterRepoAdapter") as MockAdapter, \
             patch("git_filter_repo_mcp.server.get_config") as mock_config:
            mock_config.return_value.server.auto_backup = True
            mock_adapter = MagicMock()

            def track_backup():
                call_order.append("backup")
                return "backup_test"
            mock_adapter.create_backup.side_effect = track_backup

            def track_change(*a, **kw):
                call_order.append("change_author")
                return FilterResult(success=True, message="done")
            mock_adapter.change_author.side_effect = track_change
            MockAdapter.return_value = mock_adapter

            await _execute_tool("change_author", {
                "repo_path": "/tmp/repo", "old_email": "a@b",
                "new_name": "N", "new_email": "n@b", "dry_run": False,
            })

            assert call_order == ["backup", "change_author"]

    @pytest.mark.asyncio
    async def test_no_backup_on_dry_run(self):
        with patch("git_filter_repo_mcp.server.GitFilterRepoAdapter") as MockAdapter, \
             patch("git_filter_repo_mcp.server.get_config") as mock_config:
            mock_config.return_value.server.auto_backup = True
            mock_adapter = MagicMock()
            mock_adapter.change_author.return_value = FilterResult(success=True, message="dry", dry_run=True)
            MockAdapter.return_value = mock_adapter

            result = await _execute_tool("change_author", {
                "repo_path": "/tmp/repo", "old_email": "a@b",
                "new_name": "N", "new_email": "n@b", "dry_run": True,
            })

            mock_adapter.create_backup.assert_not_called()
            assert "backup_branch" not in result


class TestTimeoutHandling:
    """Test that subprocess timeouts return structured errors, not crashes."""

    @pytest.mark.asyncio
    async def test_timeout_returns_error(self):
        import subprocess as sp
        with patch("git_filter_repo_mcp.server.GitFilterRepoAdapter") as MockAdapter:
            MockAdapter.side_effect = sp.TimeoutExpired("git", 30)

            result = await _execute_tool("analyze_git_history", {"repo_path": "/tmp/repo"})
            assert result["success"] is False
            assert "timed out" in result["error"]
            assert result["error_code"] == ErrorCode.COMMAND_FAILED


class TestSquashInjection:
    """Test that squash_commits rejects dash-prefixed refs."""

    @pytest.mark.asyncio
    async def test_dash_start_commit_rejected(self):
        with patch("git_filter_repo_mcp.server.GitFilterRepoAdapter") as MockAdapter:
            mock_adapter = MagicMock()
            mock_adapter.squash_commits.side_effect = ValueError("Invalid ref")
            MockAdapter.return_value = mock_adapter
            result = await _execute_tool("squash_commits", {
                "repo_path": "/tmp/repo", "start_commit": "--exec=evil", "dry_run": True,
            })
            assert result["success"] is False


class TestReplaceTextValidation:
    """Test replace_text input validation."""

    @pytest.mark.asyncio
    async def test_empty_old_text_rejected(self):
        with patch("git_filter_repo_mcp.server.GitFilterRepoAdapter") as MockAdapter:
            mock_adapter = MagicMock()
            mock_adapter.replace_text_in_history.return_value = FilterResult(
                success=False, message="old_text must not be empty",
            )
            MockAdapter.return_value = mock_adapter

            result = await _execute_tool("replace_text_in_history", {
                "repo_path": "/tmp/repo", "old_text": "", "new_text": "x", "dry_run": True,
            })
            assert result["success"] is False


class TestManualMappingsRealExecution:
    """Test rewrite_commit_messages with manual_mappings, dry_run=False."""

    @pytest.mark.asyncio
    async def test_manual_mappings_real_calls_adapter(self):
        with patch("git_filter_repo_mcp.server.GitFilterRepoAdapter") as MockAdapter, \
             patch("git_filter_repo_mcp.server.get_config") as mock_config:
            mock_config.return_value.server.auto_backup = True
            mock_adapter = MagicMock()
            mock_adapter.create_backup.return_value = "backup_test"
            mock_adapter.rewrite_commit_messages.return_value = FilterResult(
                success=True, message="Rewrote 1", commits_processed=3, commits_rewritten=1,
            )
            MockAdapter.return_value = mock_adapter

            result = await _execute_tool("rewrite_commit_messages", {
                "repo_path": "/tmp/repo", "use_ai": False,
                "manual_mappings": {"old": "new"}, "dry_run": False,
            })

            assert result["success"] is True
            assert result["backup_branch"] == "backup_test"
            mock_adapter.rewrite_commit_messages.assert_called_once()
            mock_adapter.create_backup.assert_called_once()

    @pytest.mark.asyncio
    async def test_manual_mappings_no_backup_on_dry_run(self):
        with patch("git_filter_repo_mcp.server.GitFilterRepoAdapter") as MockAdapter, \
             patch("git_filter_repo_mcp.server.get_config") as mock_config:
            mock_config.return_value.server.auto_backup = True
            mock_adapter = MagicMock()
            mock_adapter.rewrite_commit_messages.return_value = FilterResult(
                success=True, message="Dry run", dry_run=True,
            )
            MockAdapter.return_value = mock_adapter

            result = await _execute_tool("rewrite_commit_messages", {
                "repo_path": "/tmp/repo", "use_ai": False,
                "manual_mappings": {"old": "new"}, "dry_run": True,
            })

            assert result["success"] is True
            assert "backup_branch" not in result
            mock_adapter.create_backup.assert_not_called()


class TestScanSecretsEmptyRepo:
    """Test scan_secrets on various repo states."""

    @pytest.mark.asyncio
    async def test_empty_repo_scan(self):
        with patch("git_filter_repo_mcp.server.GitFilterRepoAdapter") as MockAdapter:
            mock_adapter = MagicMock()
            mock_adapter.scan_secrets.return_value = {
                "commits_scanned": 0, "secrets_found": 0,
                "sensitive_files": 0, "findings": [],
                "sensitive_file_list": [], "files_scanned": 0,
            }
            MockAdapter.return_value = mock_adapter

            result = await _execute_tool("scan_secrets", {"repo_path": "/tmp/repo"})
            assert result["success"] is True
            assert result["secrets_found"] == 0
