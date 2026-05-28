"""Unit tests for MCP tool definitions and the ErrorCode enum."""

import json

import pytest

from git_filter_repo_mcp.tools import (
    DESTRUCTIVE_TOOL_NAMES,
    TOOL_DEFINITIONS,
    TOOL_NAMES,
    TOOL_SPECS,
    ErrorCode,
)


# Canonical list of tool names the server must expose. Adding/removing a
# tool requires updating this set, which forces a deliberate review.
EXPECTED_TOOLS: frozenset[str] = frozenset({
    "analyze_git_history",
    "validate_repo_safety",
    "find_large_files",
    "list_backups",
    "resolve_commit",
    "rewrite_commit_messages",
    "change_author",
    "remove_files_from_history",
    "remove_large_files",
    "filter_paths",
    "create_backup",
    "restore_backup",
    "get_commit_details",
    "rewrite_single_commit",
    "scan_secrets",
    "squash_commits",
    "replace_text_in_history",
    "get_file_history",
    "list_all_files_in_history",
    "change_commit_dates",
})

# Tools that mutate history and therefore MUST expose ``dry_run``.
DESTRUCTIVE_TOOLS = DESTRUCTIVE_TOOL_NAMES


class TestToolDefinitions:
    """Structural invariants of ``TOOL_DEFINITIONS``."""

    def test_exposes_exact_expected_set(self) -> None:
        names = {tool["name"] for tool in TOOL_DEFINITIONS}
        assert names == EXPECTED_TOOLS, (
            f"missing={EXPECTED_TOOLS - names}, extra={names - EXPECTED_TOOLS}"
        )
        assert TOOL_NAMES == EXPECTED_TOOLS

    def test_tool_specs_drive_definitions(self) -> None:
        assert [spec.name for spec in TOOL_SPECS] == [
            tool["name"] for tool in TOOL_DEFINITIONS
        ]

    def test_names_are_unique(self) -> None:
        names = [tool["name"] for tool in TOOL_DEFINITIONS]
        assert len(names) == len(set(names))

    @pytest.mark.parametrize("tool", TOOL_DEFINITIONS, ids=lambda t: t["name"])
    def test_required_top_level_fields(self, tool: dict) -> None:
        assert {"name", "description", "inputSchema"} <= tool.keys()
        assert isinstance(tool["description"], str) and tool["description"].strip()

    @pytest.mark.parametrize("tool", TOOL_DEFINITIONS, ids=lambda t: t["name"])
    def test_input_schema_shape(self, tool: dict) -> None:
        schema = tool["inputSchema"]
        assert isinstance(schema, dict)
        assert "type" in schema or "properties" in schema

    @pytest.mark.parametrize(
        "tool",
        [t for t in TOOL_DEFINITIONS if t["name"] in DESTRUCTIVE_TOOLS],
        ids=lambda t: t["name"],
    )
    def test_destructive_tool_has_dry_run(self, tool: dict) -> None:
        properties = tool["inputSchema"].get("properties", {})
        assert "dry_run" in properties, (
            f"{tool['name']} is destructive but missing dry_run"
        )


class TestErrorCode:
    """``ErrorCode`` string enum used in tool responses."""

    def test_full_set_defined(self) -> None:
        assert {code.value for code in ErrorCode} == {
            "INVALID_INPUT", "REPO_NOT_FOUND", "TOOL_NOT_FOUND",
            "COMMAND_FAILED", "AI_CONNECTION_FAILED", "NO_CHANGES", "INTERNAL_ERROR",
        }

    def test_is_str_enum(self) -> None:
        assert isinstance(ErrorCode.INVALID_INPUT, str)
        assert ErrorCode.INVALID_INPUT == "INVALID_INPUT"

    def test_json_serialisable(self) -> None:
        serialised = json.dumps({"error_code": ErrorCode.REPO_NOT_FOUND})
        assert '"REPO_NOT_FOUND"' in serialised
