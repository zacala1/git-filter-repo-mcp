"""MCP Server for git-filter-repo operations."""

import asyncio
import json
import logging
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from .adapter import FilterResult, GitFilterRepoAdapter
from .ai_engine import AICommitEngine, MessageStyle, get_provider
from .config import get_config
from .tools import TOOL_DEFINITIONS

# Load config and configure logging
config = get_config()
logging.basicConfig(level=getattr(logging, config.server.log_level, logging.INFO))
logger = logging.getLogger(__name__)

# Create MCP server
server = Server("git-filter-repo-mcp")


def result_to_dict(result: FilterResult) -> dict:
    """Convert FilterResult to dictionary for JSON response."""
    return {
        "success": result.success,
        "message": result.message,
        "commits_processed": result.commits_processed,
        "commits_rewritten": result.commits_rewritten,
        "files_affected": result.files_affected,
        "dry_run": result.dry_run,
        "error": result.error,
    }


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name=tool["name"],
            description=tool["description"],
            inputSchema=tool["inputSchema"],
        )
        for tool in TOOL_DEFINITIONS
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""
    try:
        result = await _execute_tool(name, arguments)
        return [TextContent(type="text", text=json.dumps(result, indent=2, ensure_ascii=False))]
    except Exception as e:
        logger.exception(f"Error executing tool {name}")
        error_result = {"error": str(e), "success": False}
        return [TextContent(type="text", text=json.dumps(error_result, indent=2))]


async def _execute_tool(name: str, args: dict[str, Any]) -> dict:
    """Execute a tool and return the result."""
    logger.info(f"Executing tool: {name}")

    # Apply config defaults
    default_dry_run = config.server.default_dry_run

    if name == "analyze_git_history":
        adapter = GitFilterRepoAdapter(args["repo_path"])
        result = adapter.analyze_history(
            branch=args.get("branch", "HEAD"),
            max_count=args.get("max_count", 100),
        )
        return {"success": True, **result}

    elif name == "rewrite_commit_messages":
        adapter = GitFilterRepoAdapter(args["repo_path"])
        dry_run = args.get("dry_run", default_dry_run)
        use_ai = args.get("use_ai", True)
        manual_mappings = args.get("manual_mappings")

        if use_ai:
            # Use AI engine with config defaults
            style = MessageStyle(args.get("style", "conventional"))
            provider = get_provider(
                args.get("ai_provider", config.ai.provider),
                model=args.get("ai_model", config.ai.model),
                api_key=config.ai.openai_api_key
                if args.get("ai_provider", config.ai.provider) == "openai"
                else config.ai.anthropic_api_key,
                base_url=config.ai.ollama_base_url,
            )
            engine = AICommitEngine(provider, style)

            try:
                # Create async callback
                async def ai_callback(message: str, commit_hash: str) -> str:
                    files = adapter.get_commit_files(commit_hash)
                    result = await engine.rewrite_message(message, commit_hash, files)
                    return result.rewritten

                # Get commits and rewrite
                commits = adapter.get_commits(args.get("branch", "HEAD"))
                rewrites = []

                for commit in commits:
                    new_message = await ai_callback(commit.message, commit.hash)
                    if new_message != commit.message:
                        rewrites.append(
                            {
                                "hash": commit.hash[:8],
                                "original": commit.message,
                                "new": new_message,
                            }
                        )

                if dry_run:
                    return {
                        "success": True,
                        "dry_run": True,
                        "message": f"Would rewrite {len(rewrites)} commits",
                        "commits_to_rewrite": rewrites[:20],  # Preview first 20
                        "total_rewrites": len(rewrites),
                    }

                # Actually rewrite
                def sync_callback(msg: str, hash: str) -> str:
                    for r in rewrites:
                        if msg == r["original"]:
                            return r["new"]
                    return msg

                result = adapter.rewrite_commit_messages(
                    sync_callback,
                    branch=args.get("branch", "HEAD"),
                    dry_run=False,
                    force=True,
                )
                return result_to_dict(result)
            finally:
                await engine.close()

        elif manual_mappings:
            # Use manual mappings
            def callback(msg: str, _: str) -> str:
                return manual_mappings.get(msg, msg)

            result = adapter.rewrite_commit_messages(
                callback,
                branch=args.get("branch", "HEAD"),
                dry_run=dry_run,
                force=not dry_run,
            )
            return result_to_dict(result)

        else:
            return {"error": "Either use_ai or manual_mappings must be provided"}

    elif name == "change_author":
        adapter = GitFilterRepoAdapter(args["repo_path"])
        dry_run = args.get("dry_run", default_dry_run)
        result = adapter.change_author(
            old_email=args["old_email"],
            new_name=args["new_name"],
            new_email=args["new_email"],
            dry_run=dry_run,
            force=not dry_run,
        )
        return result_to_dict(result)

    elif name == "remove_files_from_history":
        adapter = GitFilterRepoAdapter(args["repo_path"])
        dry_run = args.get("dry_run", default_dry_run)
        result = adapter.remove_files(
            paths=args["paths"],
            dry_run=dry_run,
            force=not dry_run,
        )
        return result_to_dict(result)

    elif name == "remove_large_files":
        adapter = GitFilterRepoAdapter(args["repo_path"])
        dry_run = args.get("dry_run", default_dry_run)
        result = adapter.remove_large_files(
            size_threshold_mb=args.get("size_threshold_mb", 10.0),
            dry_run=dry_run,
            force=not dry_run,
        )
        return result_to_dict(result)

    elif name == "filter_paths":
        adapter = GitFilterRepoAdapter(args["repo_path"])
        dry_run = args.get("dry_run", default_dry_run)
        result = adapter.filter_paths(
            include_paths=args.get("include_paths"),
            exclude_paths=args.get("exclude_paths"),
            dry_run=dry_run,
            force=not dry_run,
        )
        return result_to_dict(result)

    elif name == "create_backup":
        adapter = GitFilterRepoAdapter(args["repo_path"])
        backup_branch = adapter.create_backup()
        return {
            "success": True,
            "backup_branch": backup_branch,
            "message": f"Created backup branch: {backup_branch}",
        }

    elif name == "restore_backup":
        adapter = GitFilterRepoAdapter(args["repo_path"])
        result = adapter.restore_backup(args["backup_branch"])
        return result_to_dict(result)

    elif name == "get_commit_details":
        adapter = GitFilterRepoAdapter(args["repo_path"])
        commits = adapter.get_commits(args["commit_hash"], max_count=1)
        if not commits:
            return {"error": f"Commit not found: {args['commit_hash']}"}

        commit = commits[0]
        files = adapter.get_commit_files(commit.hash)
        diff = adapter.get_commit_diff(commit.hash)

        return {
            "success": True,
            "commit": {
                "hash": commit.hash,
                "author_name": commit.author_name,
                "author_email": commit.author_email,
                "committer_name": commit.committer_name,
                "committer_email": commit.committer_email,
                "message": commit.message,
                "date": commit.date,
                "files": files,
            },
            "diff_summary": diff[:2000] if diff else None,
        }

    elif name == "rewrite_single_commit":
        adapter = GitFilterRepoAdapter(args["repo_path"])
        commit_hash = args["commit_hash"]
        dry_run = args.get("dry_run", default_dry_run)

        # Get commit info
        commits = adapter.get_commits(commit_hash, max_count=1)
        if not commits:
            return {"error": f"Commit not found: {commit_hash}"}

        commit = commits[0]
        new_message = args.get("new_message")

        # Generate message with AI if requested (using config defaults)
        if not new_message and args.get("use_ai"):
            provider = get_provider(
                args.get("ai_provider", config.ai.provider),
                model=args.get("ai_model", config.ai.model),
                api_key=config.ai.openai_api_key
                if args.get("ai_provider", config.ai.provider) == "openai"
                else config.ai.anthropic_api_key,
                base_url=config.ai.ollama_base_url,
            )
            engine = AICommitEngine(provider, MessageStyle.CONVENTIONAL)
            try:
                files = adapter.get_commit_files(commit.hash)
                result = await engine.rewrite_message(commit.message, commit.hash, files)
                new_message = result.rewritten
            finally:
                await engine.close()

        if dry_run:
            return {
                "success": True,
                "dry_run": True,
                "commit_hash": commit_hash,
                "original_message": commit.message,
                "new_message": new_message or commit.message,
                "new_author_name": args.get("new_author_name"),
                "new_author_email": args.get("new_author_email"),
            }

        # Apply changes
        changes_made = []

        if new_message and new_message != commit.message:

            def msg_callback(msg: str, h: str) -> str:
                if h.startswith(commit_hash) or commit_hash.startswith(h):
                    return new_message
                return msg

            result = adapter.rewrite_commit_messages(
                msg_callback,
                dry_run=False,
                force=True,
            )
            if result.success:
                changes_made.append("message")

        if args.get("new_author_email") and args.get("new_author_name"):
            result = adapter.change_author(
                old_email=commit.author_email,
                new_name=args["new_author_name"],
                new_email=args["new_author_email"],
                dry_run=False,
                force=True,
            )
            if result.success:
                changes_made.append("author")

        return {
            "success": True,
            "changes_made": changes_made,
            "message": f"Updated commit {commit_hash[:8]}: {', '.join(changes_made)}",
        }

    elif name == "scan_secrets":
        adapter = GitFilterRepoAdapter(args["repo_path"])
        result = adapter.scan_secrets(
            branch=args.get("branch", "HEAD"),
            max_commits=args.get("max_commits", 100),
        )
        return {"success": True, **result}

    elif name == "squash_commits":
        adapter = GitFilterRepoAdapter(args["repo_path"])
        dry_run = args.get("dry_run", default_dry_run)

        # Auto backup if enabled
        backup_branch = None
        if config.server.auto_backup and not dry_run:
            backup_branch = adapter.create_backup()

        result = adapter.squash_commits(
            start_commit=args["start_commit"],
            end_commit=args.get("end_commit", "HEAD"),
            new_message=args.get("new_message"),
            dry_run=dry_run,
        )

        response = result_to_dict(result)
        if backup_branch:
            response["backup_branch"] = backup_branch
        return response

    elif name == "replace_text_in_history":
        adapter = GitFilterRepoAdapter(args["repo_path"])
        dry_run = args.get("dry_run", default_dry_run)

        # Auto backup if enabled
        backup_branch = None
        if config.server.auto_backup and not dry_run:
            backup_branch = adapter.create_backup()

        result = adapter.replace_text_in_history(
            old_text=args["old_text"],
            new_text=args["new_text"],
            file_pattern=args.get("file_pattern"),
            dry_run=dry_run,
            force=not dry_run,
        )

        response = result_to_dict(result)
        if backup_branch:
            response["backup_branch"] = backup_branch
        return response

    elif name == "get_file_history":
        adapter = GitFilterRepoAdapter(args["repo_path"])
        history = adapter.get_file_history(args["file_path"])
        return {
            "success": True,
            "file_path": args["file_path"],
            "commits": history,
            "total_commits": len(history),
        }

    elif name == "list_all_files_in_history":
        adapter = GitFilterRepoAdapter(args["repo_path"])
        files = adapter.list_all_files_in_history()
        return {
            "success": True,
            "files": files[:500],  # Limit output
            "total_files": len(files),
            "truncated": len(files) > 500,
        }

    else:
        return {"error": f"Unknown tool: {name}"}


async def run_server():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main():
    """Main entry point."""
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
