"""MCP Server for git-filter-repo operations."""

import asyncio
import json
import logging
import subprocess
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
    DESTRUCTIVE_TOOL_NAMES,
    AnalyzeHistoryInput,
    ErrorCode,
    ChangeAuthorInput,
    ChangeCommitDatesInput,
    CreateBackupInput,
    FindLargeFilesInput,
    FilterPathsInput,
    GetCommitDetailsInput,
    GetFileHistoryInput,
    ListAllFilesInput,
    ListBackupsInput,
    RemoveFilesInput,
    RemoveLargeFilesInput,
    ReplaceTextInput,
    RestoreBackupInput,
    ResolveCommitInput,
    RewriteCommitMessagesInput,
    RewriteSingleCommitInput,
    ScanSecretsInput,
    SquashCommitsInput,
    ValidateRepoSafetyInput,
)

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Configure logging level from config.

    Idempotent. NOT called at import time — callers (only ``main()`` /
    ``run_server()``) trigger it so that importing the module as a library
    does not reconfigure the host application's root logger.
    """
    level = getattr(logging, get_config().server.log_level, logging.INFO)
    logging.basicConfig(level=level)
    logging.getLogger().setLevel(level)


server = Server("git-filter-repo-mcp")

_HANDLERS: dict[str, Callable] = {}


def _apply_default_dry_run(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Fill omitted dry_run from config for destructive tools."""
    if name in DESTRUCTIVE_TOOL_NAMES and "dry_run" not in args:
        return {**args, "dry_run": get_config().server.default_dry_run}
    return args


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
                    f"{'.'.join(str(part) for part in err['loc'])}: {err['msg']}"
                    for err in errors
                )
                return {"success": False, "error": f"Invalid input: {details}", "error_code": ErrorCode.INVALID_INPUT}
            except ValueError as e:
                msg = str(e)
                if "Not a git repository" in msg or "does not exist" in msg:
                    code = ErrorCode.REPO_NOT_FOUND
                elif (
                    "must " in msg
                    or " required" in msg
                    or msg.startswith("Invalid ")
                    or msg.startswith("Unknown provider")
                ):
                    code = ErrorCode.INVALID_INPUT
                else:
                    code = ErrorCode.COMMAND_FAILED
                return {"success": False, "error": msg, "error_code": code}
            except RuntimeError as e:
                return {"success": False, "error": str(e), "error_code": ErrorCode.COMMAND_FAILED}
            except subprocess.CalledProcessError as e:
                cmd = e.cmd[0] if isinstance(e.cmd, list) else str(e.cmd)
                logger.warning("%s: %s failed (rc=%d)", name, cmd, e.returncode)
                return {"success": False, "error": f"Command failed: {cmd} (exit code {e.returncode})", "error_code": ErrorCode.COMMAND_FAILED}
            except subprocess.TimeoutExpired as e:
                logger.warning("%s timed out: %s", name, e)
                return {"success": False, "error": f"Operation timed out ({e.timeout}s)", "error_code": ErrorCode.COMMAND_FAILED}
            except Exception as e:
                logger.exception("%s failed", name)
                return {"success": False, "error": f"Internal error: {type(e).__name__}", "error_code": ErrorCode.INTERNAL_ERROR}
        _HANDLERS[name] = wrapper
        return wrapper
    return decorator


def result_to_dict(result: FilterResult) -> dict:
    """Convert a FilterResult dataclass to a JSON-serializable dict."""
    return {
        "success": result.success,
        "message": result.message,
        "commits_processed": result.commits_processed,
        "commits_rewritten": result.commits_rewritten,
        "files_affected": result.files_affected,
        "dry_run": result.dry_run,
        "error": result.error,
    }


def _maybe_backup(adapter: GitFilterRepoAdapter, dry_run: bool) -> str | None:
    """Create a backup branch before a destructive operation, if configured."""
    if get_config().server.auto_backup and not dry_run:
        return adapter.create_backup()
    return None


def _run_destructive(
    repo_path: str,
    dry_run: bool,
    action: Callable[[GitFilterRepoAdapter], FilterResult],
) -> dict:
    """Common pipeline for destructive tools: create adapter, optionally back
    up, run the action, and surface the backup branch on the response.

    The action receives the constructed adapter and returns a ``FilterResult``.
    Centralising this collapses ~6 lines of identical boilerplate across every
    destructive handler.
    """
    adapter = GitFilterRepoAdapter(repo_path)
    backup = _maybe_backup(adapter, dry_run)
    result = action(adapter)
    response = result_to_dict(result)
    if backup:
        response["backup_branch"] = backup
    return response


_VALID_AI_PROVIDERS = {"ollama", "openai", "anthropic"}
_CONFIG_DEFAULT_AI_MODEL = "llama3.2"
_PROVIDER_DEFAULT_MODELS = {
    "ollama": "llama3.2",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-20250514",
}


def _resolve_ai_model(args: dict, provider_name: str, configured_model: str) -> str:
    """Choose a provider-appropriate model unless the user configured one."""
    if model := args.get("ai_model"):
        return model

    if provider_name == "ollama" or configured_model != _CONFIG_DEFAULT_AI_MODEL:
        return configured_model
    return _PROVIDER_DEFAULT_MODELS[provider_name]


def _create_ai_provider(args: dict, provider_name: str):
    """Create an AI provider instance from tool args and global config.

    Raises ValueError if provider_name is not in _VALID_AI_PROVIDERS.
    """
    if provider_name not in _VALID_AI_PROVIDERS:
        raise ValueError(
            f"Invalid AI provider: {provider_name!r}. Must be one of: {', '.join(sorted(_VALID_AI_PROVIDERS))}"
        )
    cfg = get_config()
    kwargs: dict = {"model": _resolve_ai_model(args, provider_name, cfg.ai.model)}
    if provider_name == "ollama":
        kwargs["base_url"] = cfg.ai.ollama_base_url
    elif provider_name == "openai":
        kwargs["api_key"] = cfg.ai.openai_api_key
        kwargs["base_url"] = cfg.ai.openai_base_url
    elif provider_name == "anthropic":
        kwargs["api_key"] = cfg.ai.anthropic_api_key
    return get_provider(provider_name, **kwargs)


def _ai_provider_config_error(provider_name: str, error: Exception) -> dict:
    """Return a consistent envelope for provider selection/config failures."""
    return {
        "success": False,
        "error": f"AI provider '{provider_name}' is not configured: {error}",
        "error_code": ErrorCode.INVALID_INPUT,
        "ai_provider": provider_name,
    }


async def _check_ai_connection(provider, provider_name: str) -> dict | None:
    """Check AI connection, return error dict if failed, None if ok."""
    connected, status = await provider.check_connection()
    if not connected:
        return {
            "success": False,
            "error": f"AI ({provider_name}) connection failed: {status}",
            "error_code": ErrorCode.AI_CONNECTION_FAILED,
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


@tool_handler("validate_repo_safety")
async def _handle_validate_repo_safety(args: dict) -> dict:
    params = ValidateRepoSafetyInput(**args)

    def _run():
        return {
            "success": True,
            **GitFilterRepoAdapter(params.repo_path).validate_repo_safety(),
        }

    return await asyncio.to_thread(_run)


@tool_handler("find_large_files")
async def _handle_find_large_files(args: dict) -> dict:
    params = FindLargeFilesInput(**args)

    def _run():
        return {
            "success": True,
            **GitFilterRepoAdapter(params.repo_path).find_large_files(
                params.size_threshold_mb,
                params.limit,
            ),
        }

    return await asyncio.to_thread(_run)


@tool_handler("list_backups")
async def _handle_list_backups(args: dict) -> dict:
    params = ListBackupsInput(**args)

    def _run():
        return {
            "success": True,
            **GitFilterRepoAdapter(params.repo_path).list_backups(params.limit),
        }

    return await asyncio.to_thread(_run)


@tool_handler("resolve_commit")
async def _handle_resolve_commit(args: dict) -> dict:
    params = ResolveCommitInput(**args)

    def _run():
        resolved = GitFilterRepoAdapter(params.repo_path).resolve_commit_ref(params.commit_ref)
        if resolved is None:
            return {
                "success": False,
                "error": f"Commit ref not found: {params.commit_ref}",
                "error_code": ErrorCode.INVALID_INPUT,
            }
        return {"success": True, "commit": resolved}

    return await asyncio.to_thread(_run)


@tool_handler("rewrite_commit_messages")
async def _handle_rewrite_messages(args: dict) -> dict:
    params = RewriteCommitMessagesInput(**args)
    dry_run = params.dry_run

    # Reject ambiguous input early — silently ignoring manual_mappings while
    # using AI would surprise users debugging unexpected rewrites.
    if params.use_ai and params.manual_mappings:
        return {
            "success": False,
            "error": "Cannot use use_ai=true together with manual_mappings. "
                     "Provide one or the other.",
            "error_code": ErrorCode.INVALID_INPUT,
        }

    adapter = await asyncio.to_thread(GitFilterRepoAdapter, params.repo_path)

    if params.use_ai:
        ai_provider_name = params.ai_provider or get_config().ai.provider
        if ai_provider_name == "none":
            return {"success": False, "error": "AI provider is set to 'none'. Configure a provider or pass ai_provider explicitly.", "error_code": ErrorCode.INVALID_INPUT}
        try:
            provider = _create_ai_provider(args, ai_provider_name)
        except ValueError as e:
            return _ai_provider_config_error(ai_provider_name, e)
        engine = AICommitEngine(provider, MessageStyle(params.style))

        try:
            if err := await _check_ai_connection(provider, ai_provider_name):
                return err

            commits = await asyncio.to_thread(adapter.get_commits, params.branch)
            commit_files = await asyncio.to_thread(
                adapter.collect_commit_files, commits, params.branch, len(commits),
            )

            batch_input = [
                (c.hash, c.message, commit_files.get(c.hash, []))
                for c in commits
            ]
            results = await engine.rewrite_batch(batch_input)
            ai_failures = [
                r for r in results
                if r.reasoning and r.reasoning.startswith("AI call failed")
            ]
            if results and len(ai_failures) == len(results):
                return {
                    "success": False,
                    "error": "AI failed to rewrite all commit messages",
                    "error_code": ErrorCode.AI_CONNECTION_FAILED,
                    "ai_provider": ai_provider_name,
                    "ai_failures": len(ai_failures),
                }

            rewrites = []
            for commit, result in zip(commits, results):
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
                    "ai_failures": len(ai_failures),
                }

            if not rewrites:
                return {
                    "success": True,
                    "message": "No commits need rewriting",
                    "ai_provider": ai_provider_name,
                    "ai_failures": len(ai_failures),
                }

            backup = _maybe_backup(adapter, dry_run)
            rewrite_by_hash = {r["hash"]: r["new"] for r in rewrites}

            def sync_callback(msg: str, commit_hash: str) -> str:
                return rewrite_by_hash.get(commit_hash, msg)

            result = await asyncio.to_thread(
                adapter.rewrite_commit_messages,
                sync_callback, branch=params.branch, dry_run=False, force=True,
            )
            response = result_to_dict(result)
            response["ai_failures"] = len(ai_failures)
            if backup:
                response["backup_branch"] = backup
            return response
        except AIConnectionError as e:
            return {"success": False, "error": str(e), "error_code": ErrorCode.AI_CONNECTION_FAILED, "ai_provider": ai_provider_name}
        finally:
            # ``close()`` failures (e.g. httpx teardown) must NOT mask the
            # handler's actual result/exception. Log and swallow.
            try:
                await engine.close()
            except Exception:
                logger.warning("AI engine close failed", exc_info=True)

    elif params.manual_mappings:
        mappings = params.manual_mappings

        def callback(msg: str, _: str) -> str:
            return mappings.get(msg, msg)

        backup = _maybe_backup(adapter, dry_run)
        result = await asyncio.to_thread(
            adapter.rewrite_commit_messages,
            callback, branch=params.branch, dry_run=dry_run, force=not dry_run,
        )
        response = result_to_dict(result)
        if backup:
            response["backup_branch"] = backup
        return response

    else:
        return {"success": False, "error": "Either use_ai or manual_mappings must be provided", "error_code": ErrorCode.INVALID_INPUT}


@tool_handler("change_author")
async def _handle_change_author(args: dict) -> dict:
    params = ChangeAuthorInput(**args)
    return await asyncio.to_thread(
        _run_destructive,
        params.repo_path, params.dry_run,
        lambda a: a.change_author(
            params.old_email, params.new_name, params.new_email,
            params.dry_run, not params.dry_run,
        ),
    )


@tool_handler("remove_files_from_history")
async def _handle_remove_files(args: dict) -> dict:
    params = RemoveFilesInput(**args)
    return await asyncio.to_thread(
        _run_destructive,
        params.repo_path, params.dry_run,
        lambda a: a.remove_files(params.paths, params.dry_run, not params.dry_run),
    )


@tool_handler("remove_large_files")
async def _handle_remove_large_files(args: dict) -> dict:
    params = RemoveLargeFilesInput(**args)
    return await asyncio.to_thread(
        _run_destructive,
        params.repo_path, params.dry_run,
        lambda a: a.remove_large_files(
            params.size_threshold_mb, params.dry_run, not params.dry_run,
        ),
    )


@tool_handler("filter_paths")
async def _handle_filter_paths(args: dict) -> dict:
    params = FilterPathsInput(**args)
    return await asyncio.to_thread(
        _run_destructive,
        params.repo_path, params.dry_run,
        lambda a: a.filter_paths(
            params.include_paths, params.exclude_paths,
            params.dry_run, not params.dry_run,
        ),
    )


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
    if params.commit_hash.startswith("-"):
        return {"success": False, "error": "Invalid commit hash", "error_code": ErrorCode.INVALID_INPUT}

    def _run():
        adapter = GitFilterRepoAdapter(params.repo_path)
        commits = adapter.get_commits(params.commit_hash, max_count=1)
        if not commits:
            return {"success": False, "error": f"Commit not found: {params.commit_hash}", "error_code": ErrorCode.INVALID_INPUT}
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
    try:
        adapter._validate_commit_hash(commit_hash)
    except ValueError as e:
        return {"success": False, "error": str(e), "error_code": ErrorCode.INVALID_INPUT}
    commits = await asyncio.to_thread(adapter.get_commits, commit_hash, 1)
    if not commits:
        return {"success": False, "error": f"Commit not found: {commit_hash}", "error_code": ErrorCode.INVALID_INPUT}

    commit = commits[0]
    new_message = params.new_message

    if not new_message and params.use_ai:
        ai_provider_name = params.ai_provider or get_config().ai.provider
        if ai_provider_name == "none":
            return {"success": False, "error": "AI provider is set to 'none'. Configure a provider or pass ai_provider explicitly.", "error_code": ErrorCode.INVALID_INPUT}
        try:
            provider = _create_ai_provider(args, ai_provider_name)
        except ValueError as e:
            return _ai_provider_config_error(ai_provider_name, e)
        engine = AICommitEngine(provider, MessageStyle.CONVENTIONAL)
        try:
            if err := await _check_ai_connection(provider, ai_provider_name):
                return err
            files = await asyncio.to_thread(adapter.get_commit_files, commit.hash)
            result = await engine.rewrite_message(commit.message, commit.hash, files)
            new_message = result.rewritten
        except AIConnectionError as e:
            return {"success": False, "error": str(e), "error_code": ErrorCode.AI_CONNECTION_FAILED, "ai_provider": ai_provider_name}
        finally:
            # ``close()`` failures (e.g. httpx teardown) must NOT mask the
            # handler's actual result/exception. Log and swallow.
            try:
                await engine.close()
            except Exception:
                logger.warning("AI engine close failed", exc_info=True)

    has_message_change = new_message and new_message != commit.message
    has_partial_author = bool(params.new_author_email) != bool(params.new_author_name)
    has_author_change = params.new_author_email and params.new_author_name

    if has_partial_author:
        missing = "new_author_name" if params.new_author_email else "new_author_email"
        return {
            "success": False,
            "error": f"Both new_author_name and new_author_email are required (missing: {missing})",
            "error_code": ErrorCode.INVALID_INPUT,
        }

    if not has_message_change and not has_author_change and not params.dry_run:
        return {
            "success": False,
            "error": "No changes specified. Provide new_message, use_ai=true, or new author info.",
            "error_code": ErrorCode.NO_CHANGES,
        }

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

    backup = _maybe_backup(adapter, params.dry_run)
    result = await asyncio.to_thread(
        adapter.rewrite_single_commit,
        commit_hash,
        new_message=new_message if has_message_change else None,
        new_author_name=params.new_author_name if has_author_change else None,
        new_author_email=params.new_author_email if has_author_change else None,
    )
    response = result_to_dict(result)
    if backup:
        response["backup_branch"] = backup
    return response


@tool_handler("scan_secrets")
async def _handle_scan_secrets(args: dict) -> dict:
    params = ScanSecretsInput(**args)

    def _run():
        return {"success": True, **GitFilterRepoAdapter(params.repo_path).scan_secrets(params.branch, params.max_commits)}

    return await asyncio.to_thread(_run)


@tool_handler("squash_commits")
async def _handle_squash_commits(args: dict) -> dict:
    params = SquashCommitsInput(**args)
    return await asyncio.to_thread(
        _run_destructive,
        params.repo_path, params.dry_run,
        lambda a: a.squash_commits(
            params.start_commit, params.end_commit, params.new_message, params.dry_run,
        ),
    )


@tool_handler("replace_text_in_history")
async def _handle_replace_text(args: dict) -> dict:
    params = ReplaceTextInput(**args)
    return await asyncio.to_thread(
        _run_destructive,
        params.repo_path, params.dry_run,
        lambda a: a.replace_text_in_history(
            params.old_text, params.new_text, params.file_pattern,
            params.dry_run, not params.dry_run,
        ),
    )


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
    return await asyncio.to_thread(
        _run_destructive,
        params.repo_path, params.dry_run,
        lambda a: a.change_commit_dates(
            params.time_range, params.weekend_only, params.preserve_order,
            params.start_date, dry_run=params.dry_run, force=not params.dry_run,
        ),
    )


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
        logger.exception("%s failed", name)
        return [TextContent(type="text", text=json.dumps({"error": str(e), "success": False}, indent=2))]


async def _execute_tool(name: str, args: dict[str, Any] | None) -> dict:
    """Dispatch tool execution to registered handler."""
    logger.info("tool: %s", name)
    handler = _HANDLERS.get(name)
    if handler is None:
        return {"success": False, "error": f"Unknown tool: {name}", "error_code": ErrorCode.TOOL_NOT_FOUND}
    if args is None:
        args = {}
    elif not isinstance(args, dict):
        return {
            "success": False,
            "error": "Tool arguments must be a JSON object",
            "error_code": ErrorCode.INVALID_INPUT,
        }
    args = _apply_default_dry_run(name, args)
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
    except Exception:
        logger.exception("server error")
        raise


def main():
    """Entry point."""
    _configure_logging()
    server_coro = run_server()
    try:
        asyncio.run(server_coro)
    except KeyboardInterrupt:
        logger.info("stopped")
    except Exception:
        logger.exception("fatal")
        raise SystemExit(1)
    finally:
        server_coro.close()


if __name__ == "__main__":
    main()
