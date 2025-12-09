"""Tests for MCP server."""

import json
from unittest.mock import MagicMock, patch

import pytest

from git_filter_repo_mcp.adapter import FilterResult
from git_filter_repo_mcp.server import (
    _execute_tool,
    call_tool,
    list_tools,
    result_to_dict,
)


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
            mock_adapter.analyze_history.assert_called_once_with(
                branch="main",
                max_count=50,
            )

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
                old_email="old@example.com",
                new_name="New Name",
                new_email="new@example.com",
                dry_run=True,
                force=False,
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
            mock_adapter.scan_secrets.assert_called_once_with(
                branch="main",
                max_commits=50,
            )

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
