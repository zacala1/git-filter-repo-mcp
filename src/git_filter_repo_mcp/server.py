"""MCP Server for git-filter-repo operations."""

import asyncio
import json
import logging
from functools import wraps
from typing import Any, Callable

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool
from pydantic import ValidationError

from .adapter import FilterResult, GitFilterRepoAdapter
from .ai_engine import AICommitEngine, AIConnectionError, MessageStyle, get_provider
from .config import get_config
from .tools import (
    TOOL_DEFINITIONS,
    AnalyzeHistoryInput,
    ChangeAuthorInput,
    ChangeCommitDatesInput,
    CreateBackupInput,
    FilterPathsInput,
    GetCommitDetailsInput,
    GetFileHistoryInput,
    ListAllFilesInput,
    RemoveFilesInput,
    RemoveLargeFilesInput,
    ReplaceTextInput,
    RestoreBackupInput,
    RewriteCommitMessagesInput,
    RewriteSingleCommitInput,
    ScanSecretsInput,
    SquashCommitsInput,
)

# Config and logging
config = get_config()
logging.basicConfig(level=getattr(logging, config.server.log_level, logging.INFO))
logger = logging.getLogger(__name__)

server = Server("git-filter-repo-mcp")

# Tool handler registry
_HANDLERS: dict[str, Callable] = {}


def tool_handler(name: str):
    """Register an async function as the handler for a named tool."""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(args: dict[str, Any]) -> dict:
            try:
                return await func(args)
            except ValidationError as e:
                errors = e.errors()
                details = "; ".join(
                    f"{'.'.join(str(l) for l in err['loc'])}: {err['msg']}"
                    for err in errors
                )
                return {"success": False, "error": f"Invalid input: {details}"}
            except (ValueError, RuntimeError) as e:
                return {"success": False, "error": str(e)}
            except Exception as e:
                logger.exception(f"{name} failed")
                return {"success": False, "error": f"Internal error: {type(e).__name__}"}
        _HANDLERS[name] = wrapper
        return wrapper
    return decorator


def result_to_dict(result: FilterResult) -> dict:
    """FilterResult -> dict"""
    return {
        "success": result.success,
        "message": result.message,
        "commits_processed": result.commits_processed,
        "commits_rewritten": result.commits_rewritten,
        "files_affected": result.files_affected,
        "dry_run": result.dry_run,
        "error": result.error,
    }


def _create_ai_provider(args: dict, provider_name: str):
    """Create an AI provider from args and config."""
    return get_provider(
        provider_name,
        model=args.get("ai_model", config.ai.model),
        api_key=config.ai.openai_api_key
        if provider_name == "openai"
        else config.ai.anthropic_api_key,
        base_url=config.ai.ollama_base_url,
    )


async def _check_ai_connection(provider, provider_name: str) -> dict | None:
    """Check AI connection, return error dict if failed, None if ok."""
    if hasattr(provider, "check_connection"):
        connected, status = await provider.check_connection()
        if not connected:
            return {
                "success": False,
                "error": f"AI ({provider_name}) connection failed: {status}",
                "ai_provider": provider_name,
            }
    return None


# --- Tool Handlers ---


@tool_handler("analyze_git_history")
async def _handle_analyze(args: dict) -> dict:
    params = AnalyzeHistoryInput(**args)

    def _run():
        adapter = GitFilterRepoAdapter(params.repo_path)
        return {"success": True, **adapter.analyze_history(params.branch, params.max_count)}

    return await asyncio.to_thread(_run)


@tool_handler("rewrite_commit_messages")
async def _handle_rewrite_messages(args: dict) -> dict:
    params = RewriteCommitMessagesInput(**args)
    dry_run = params.dry_run
    adapter = await asyncio.to_thread(GitFilterRepoAdapter, params.repo_path)

    if params.use_ai:
        ai_provider_name = params.ai_provider or config.ai.provider
        provider = _create_ai_provider(args, ai_provider_name)
        engine = AICommitEngine(provider, MessageStyle(params.style))

        try:
            if err := await _check_ai_connection(provider, ai_provider_name):
                return err

            commits = await asyncio.to_thread(adapter.get_commits, params.branch)
            rewrites = []

            for commit in commits:
                files = await asyncio.to_thread(adapter.get_commit_files, commit.hash)
                result = await engine.rewrite_message(commit.message, commit.hash, files)
                if result.rewritten != commit.message:
                    rewrites.append({
                        "hash": commit.hash,
                        "hash_short": commit.hash[:8],
                        "original": commit.message,
                        "new": result.rewritten,
                    })

            if dry_run:
                return {
                    "success": True,
                    "dry_run": True,
                    "message": f"Would rewrite {len(rewrites)} commits",
                    "commits_to_rewrite": [
                        {"hash": r["hash_short"], "original": r["original"], "new": r["new"]}
                        for r in rewrites[:20]
                    ],
                    "total_rewrites": len(rewrites),
                    "ai_provider": ai_provider_name,
                }

            rewrite_by_hash = {r["hash"]: r["new"] for r in rewrites}

            def sync_callback(msg: str, commit_hash: str) -> str:
                return rewrite_by_hash.get(commit_hash, msg)

            result = await asyncio.to_thread(
                adapter.rewrite_commit_messages,
                sync_callback, branch=params.branch, dry_run=False, force=True,
            )
            return result_to_dict(result)
        except AIConnectionError as e:
            return {"success": False, "error": str(e), "ai_provider": ai_provider_name}
        finally:
            await engine.close()

    elif params.manual_mappings:
        def callback(msg: str, _: str) -> str:
            return params.manual_mappings.get(msg, msg)

        result = await asyncio.to_thread(
            adapter.rewrite_commit_messages,
            callback, branch=params.branch, dry_run=dry_run, force=not dry_run,
        )
        return result_to_dict(result)

    else:
        return {"success": False, "error": "Either use_ai or manual_mappings must be provided"}


@tool_handler("change_author")
async def _handle_change_author(args: dict) -> dict:
    params = ChangeAuthorInput(**args)

    def _run():
        adapter = GitFilterRepoAdapter(params.repo_path)
        return result_to_dict(
            adapter.change_author(params.old_email, params.new_name, params.new_email, params.dry_run, not params.dry_run)
        )

    return await asyncio.to_thread(_run)


@tool_handler("remove_files_from_history")
async def _handle_remove_files(args: dict) -> dict:
    params = RemoveFilesInput(**args)

    def _run():
        adapter = GitFilterRepoAdapter(params.repo_path)
        return result_to_dict(adapter.remove_files(params.paths, params.dry_run, not params.dry_run))

    return await asyncio.to_thread(_run)


@tool_handler("remove_large_files")
async def _handle_remove_large_files(args: dict) -> dict:
    params = RemoveLargeFilesInput(**args)

    def _run():
        adapter = GitFilterRepoAdapter(params.repo_path)
        return result_to_dict(adapter.remove_large_files(params.size_threshold_mb, params.dry_run, not params.dry_run))

    return await asyncio.to_thread(_run)


@tool_handler("filter_paths")
async def _handle_filter_paths(args: dict) -> dict:
    params = FilterPathsInput(**args)

    def _run():
        adapter = GitFilterRepoAdapter(params.repo_path)
        return result_to_dict(
            adapter.filter_paths(params.include_paths, params.exclude_paths, params.dry_run, not params.dry_run)
        )

    return await asyncio.to_thread(_run)


@tool_handler("create_backup")
async def _handle_create_backup(args: dict) -> dict:
    params = CreateBackupInput(**args)

    def _run():
        backup = GitFilterRepoAdapter(params.repo_path).create_backup()
        return {"success": True, "backup_branch": backup, "message": f"Backup: {backup}"}

    return await asyncio.to_thread(_run)


@tool_handler("restore_backup")
async def _handle_restore_backup(args: dict) -> dict:
    params = RestoreBackupInput(**args)

    def _run():
        return result_to_dict(GitFilterRepoAdapter(params.repo_path).restore_backup(params.backup_branch))

    return await asyncio.to_thread(_run)


@tool_handler("get_commit_details")
async def _handle_get_commit_details(args: dict) -> dict:
    params = GetCommitDetailsInput(**args)

    def _run():
        adapter = GitFilterRepoAdapter(params.repo_path)
        commits = adapter.get_commits(params.commit_hash, max_count=1)
        if not commits:
            return {"success": False, "error": f"Commit not found: {params.commit_hash}"}
        c = commits[0]
        return {
            "success": True,
            "commit": {
                "hash": c.hash,
                "author_name": c.author_name,
                "author_email": c.author_email,
                "committer_name": c.committer_name,
                "committer_email": c.committer_email,
                "message": c.message,
                "date": c.date,
                "files": adapter.get_commit_files(c.hash),
            },
            "diff_summary": (adapter.get_commit_diff(c.hash) or "")[:2000] or None,
        }

    return await asyncio.to_thread(_run)


@tool_handler("rewrite_single_commit")
async def _handle_rewrite_single_commit(args: dict) -> dict:
    params = RewriteSingleCommitInput(**args)
    adapter = await asyncio.to_thread(GitFilterRepoAdapter, params.repo_path)
    commit_hash = params.commit_hash
    commits = await asyncio.to_thread(adapter.get_commits, commit_hash, 1)
    if not commits:
        return {"success": False, "error": f"Commit not found: {commit_hash}"}

    commit = commits[0]
    new_message = params.new_message

    if not new_message and params.use_ai:
        ai_provider_name = params.ai_provider or config.ai.provider
        provider = _create_ai_provider(args, ai_provider_name)
        engine = AICommitEngine(provider, MessageStyle.CONVENTIONAL)
        try:
            if err := await _check_ai_connection(provider, ai_provider_name):
                return err
            files = await asyncio.to_thread(adapter.get_commit_files, commit.hash)
            result = await engine.rewrite_message(commit.message, commit.hash, files)
            new_message = result.rewritten
        except AIConnectionError as e:
            return {"success": False, "error": str(e), "ai_provider": ai_provider_name}
        finally:
            await engine.close()

    if params.dry_run:
        return {
            "success": True,
            "dry_run": True,
            "commit_hash": commit_hash,
            "original_message": commit.message,
            "new_message": new_message or commit.message,
            "new_author_name": params.new_author_name,
            "new_author_email": params.new_author_email,
        }

    changes_made = []

    if new_message and new_message != commit.message:
        def msg_callback(msg: str, h: str) -> str:
            if h.startswith(commit_hash) or commit_hash.startswith(h):
                return new_message
            return msg

        result = await asyncio.to_thread(
            adapter.rewrite_commit_messages, msg_callback, dry_run=False, force=True,
        )
        if result.success:
            changes_made.append("message")

    if params.new_author_email and params.new_author_name:
        result = await asyncio.to_thread(
            adapter.change_author,
            old_email=commit.author_email,
            new_name=params.new_author_name,
            new_email=params.new_author_email,
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


@tool_handler("scan_secrets")
async def _handle_scan_secrets(args: dict) -> dict:
    params = ScanSecretsInput(**args)

    def _run():
        return {"success": True, **GitFilterRepoAdapter(params.repo_path).scan_secrets(params.branch, params.max_commits)}

    return await asyncio.to_thread(_run)


@tool_handler("squash_commits")
async def _handle_squash_commits(args: dict) -> dict:
    params = SquashCommitsInput(**args)

    def _run():
        adapter = GitFilterRepoAdapter(params.repo_path)
        backup = adapter.create_backup() if config.server.auto_backup and not params.dry_run else None
        result = adapter.squash_commits(params.start_commit, params.end_commit, params.new_message, params.dry_run)
        response = result_to_dict(result)
        if backup:
            response["backup_branch"] = backup
        return response

    return await asyncio.to_thread(_run)


@tool_handler("replace_text_in_history")
async def _handle_replace_text(args: dict) -> dict:
    params = ReplaceTextInput(**args)

    def _run():
        adapter = GitFilterRepoAdapter(params.repo_path)
        backup = adapter.create_backup() if config.server.auto_backup and not params.dry_run else None
        result = adapter.replace_text_in_history(
            params.old_text, params.new_text, params.file_pattern, params.dry_run, not params.dry_run,
        )
        response = result_to_dict(result)
        if backup:
            response["backup_branch"] = backup
        return response

    return await asyncio.to_thread(_run)


@tool_handler("get_file_history")
async def _handle_get_file_history(args: dict) -> dict:
    params = GetFileHistoryInput(**args)

    def _run():
        history = GitFilterRepoAdapter(params.repo_path).get_file_history(params.file_path)
        return {"success": True, "file_path": params.file_path, "commits": history, "total_commits": len(history)}

    return await asyncio.to_thread(_run)


@tool_handler("list_all_files_in_history")
async def _handle_list_all_files(args: dict) -> dict:
    params = ListAllFilesInput(**args)

    def _run():
        files = GitFilterRepoAdapter(params.repo_path).list_all_files_in_history()
        return {"success": True, "files": files[:500], "total_files": len(files), "truncated": len(files) > 500}

    return await asyncio.to_thread(_run)


@tool_handler("change_commit_dates")
async def _handle_change_dates(args: dict) -> dict:
    params = ChangeCommitDatesInput(**args)

    def _run():
        adapter = GitFilterRepoAdapter(params.repo_path)
        backup = adapter.create_backup() if config.server.auto_backup and not params.dry_run else None
        result = adapter.change_commit_dates(
            params.time_range, params.weekend_only, params.preserve_order,
            params.start_date, dry_run=params.dry_run, force=not params.dry_run,
        )
        response = result_to_dict(result)
        if backup:
            response["backup_branch"] = backup
        return response

    return await asyncio.to_thread(_run)


# --- MCP Protocol ---


@server.list_tools()
async def list_tools() -> list[Tool]:
    """Return tool list."""
    return [
        Tool(name=tool["name"], description=tool["description"], inputSchema=tool["inputSchema"])
        for tool in TOOL_DEFINITIONS
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool call."""
    try:
        result = await _execute_tool(name, arguments)
        return [TextContent(type="text", text=json.dumps(result, indent=2, ensure_ascii=False))]
    except Exception as e:
        logger.exception(f"{name} failed")
        return [TextContent(type="text", text=json.dumps({"error": str(e), "success": False}, indent=2))]


async def _execute_tool(name: str, args: dict[str, Any]) -> dict:
    """Dispatch tool execution to registered handler."""
    logger.info(f"tool: {name}")
    handler = _HANDLERS.get(name)
    if handler is None:
        return {"success": False, "error": f"Unknown tool: {name}"}
    return await handler(args)


async def run_server():
    """Run MCP server."""
    logger.info("server starting")
    try:
        async with stdio_server() as (read_stream, write_stream):
            logger.info("ready")
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )
    except Exception as e:
        logger.exception("server error")
        raise


def main():
    """Entry point."""
    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        logger.info("stopped")
    except Exception as e:
        logger.exception("fatal")


if __name__ == "__main__":
    main()
