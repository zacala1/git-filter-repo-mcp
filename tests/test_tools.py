"""Tests for MCP tool definitions."""

from git_filter_repo_mcp.tools import TOOL_DEFINITIONS


class TestToolDefinitions:
    """Test tool definitions."""

    def test_tool_count(self):
        # We should have 15 tools
        assert len(TOOL_DEFINITIONS) >= 15

    def test_all_tools_have_required_fields(self):
        for tool in TOOL_DEFINITIONS:
            assert "name" in tool, f"Tool missing 'name': {tool}"
            assert "description" in tool, f"Tool {tool.get('name')} missing 'description'"
            assert "inputSchema" in tool, f"Tool {tool.get('name')} missing 'inputSchema'"

    def test_tool_names_are_unique(self):
        names = [tool["name"] for tool in TOOL_DEFINITIONS]
        assert len(names) == len(set(names)), "Duplicate tool names found"

    def test_expected_tools_exist(self):
        tool_names = {tool["name"] for tool in TOOL_DEFINITIONS}
        expected_tools = {
            "analyze_git_history",
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
        }
        for expected in expected_tools:
            assert expected in tool_names, f"Missing expected tool: {expected}"

    def test_input_schemas_are_valid(self):
        for tool in TOOL_DEFINITIONS:
            schema = tool["inputSchema"]
            assert isinstance(schema, dict), f"Invalid schema for {tool['name']}"
            # JSON Schema should have 'type' or 'properties'
            assert "type" in schema or "properties" in schema, (
                f"Schema for {tool['name']} missing type/properties"
            )

    def test_dangerous_tools_have_dry_run(self):
        """Tools that modify history should have dry_run parameter."""
        dangerous_tools = [
            "rewrite_commit_messages",
            "change_author",
            "remove_files_from_history",
            "remove_large_files",
            "filter_paths",
            "squash_commits",
            "replace_text_in_history",
            "rewrite_single_commit",
        ]
        for tool in TOOL_DEFINITIONS:
            if tool["name"] in dangerous_tools:
                schema = tool["inputSchema"]
                properties = schema.get("properties", {})
                assert "dry_run" in properties, (
                    f"Dangerous tool {tool['name']} missing dry_run parameter"
                )
