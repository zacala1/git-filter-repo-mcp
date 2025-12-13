"""MCP Server for git-filter-repo operations."""

import asyncio
import json
import logging
from functools import wraps
from typing import Any, Callable

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from .adapter import FilterResult, GitFilterRepoAdapter
from .ai_engine import AICommitEngine, AIConnectionError, MessageStyle, get_provider
from .config import get_config
from .tools import TOOL_DEFINITIONS

# Config and logging
config = get_config()
logging.basicConfig(level=getattr(logging, config.server.log_level, logging.INFO))
logger = logging.getLogger(__name__)

server = Server("git-filter-repo-mcp")


def result_to_dict(result: FilterResult) -> dict:
    """FilterResult -> dict"""
    return {
        "success": result.success, "message": result.message,
        "commits_processed": result.commits_processed, "commits_rewritten": result.commits_rewritten,
        "files_affected": result.files_affected, "dry_run": result.dry_run, "error": result.error,
    }


def create_adapter(repo_path: str) -> GitFilterRepoAdapter:
    """Create adapter with proper error handling."""
    return GitFilterRepoAdapter(repo_path)


def handle_errors(tool_name: str):
    """Decorator for consistent error handling in tool handlers."""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except (ValueError, RuntimeError) as e:
                return {"success": False, "error": str(e)}
            except Exception as e:
                logger.exception(f"{tool_name} failed")
                return {"success": False, "error": str(e)}
        return wrapper
    return decorator


@server.list_tools()
async def list_tools() -> list[Tool]:
    """Return tool list."""
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
    """Handle tool call."""
    try:
        result = await _execute_tool(name, arguments)
        return [TextContent(type="text", text=json.dumps(result, indent=2, ensure_ascii=False))]
    except Exception as e:
        logger.exception(f"{name} failed")
        return [TextContent(type="text", text=json.dumps({"error": str(e), "success": False}, indent=2))]


async def _execute_tool(name: str, args: dict[str, Any]) -> dict:
    """Execute tool."""
    logger.info(f"tool: {name}")
    dry_run = args.get("dry_run", config.server.default_dry_run)

    if name == "analyze_git_history":
        try:
            adapter = create_adapter(args["repo_path"])
            return {"success": True, **adapter.analyze_history(args.get("branch", "HEAD"), args.get("max_count", 100))}
        except (ValueError, RuntimeError) as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.exception("analyze_git_history failed")
            return {"success": False, "error": str(e)}

    elif name == "rewrite_commit_messages":
        try:
            adapter = create_adapter(args["repo_path"])
        except (ValueError, RuntimeError) as e:
            return {"success": False, "error": str(e)}
        use_ai = args.get("use_ai", True)
        manual_mappings = args.get("manual_mappings")

        if use_ai:
            style = MessageStyle(args.get("style", "conventional"))
            ai_provider_name = args.get("ai_provider", config.ai.provider)
            provider = get_provider(
                ai_provider_name,
                model=args.get("ai_model", config.ai.model),
                api_key=config.ai.openai_api_key
                if ai_provider_name == "openai"
                else config.ai.anthropic_api_key,
                base_url=config.ai.ollama_base_url,
            )
            engine = AICommitEngine(provider, style)

            try:
                if hasattr(provider, "check_connection"):
                    connected, status = await provider.check_connection()
                    if not connected:
                        return {
                            "success": False,
                            "error": f"AI ({ai_provider_name}) connection failed: {status}",
                            "ai_provider": ai_provider_name,
                        }

                async def ai_callback(message: str, commit_hash: str) -> str:
                    files = adapter.get_commit_files(commit_hash)
                    result = await engine.rewrite_message(message, commit_hash, files)
                    return result.rewritten

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
                        "commits_to_rewrite": rewrites[:20],
                        "total_rewrites": len(rewrites),
                        "ai_provider": ai_provider_name,
                    }

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
            except AIConnectionError as e:
                return {
                    "success": False,
                    "error": str(e),
                    "ai_provider": ai_provider_name,
                }
            finally:
                await engine.close()

        elif manual_mappings:
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
        try:
            adapter = create_adapter(args["repo_path"])
            return result_to_dict(adapter.change_author(args["old_email"], args["new_name"], args["new_email"], dry_run, not dry_run))
        except (ValueError, RuntimeError) as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.exception("change_author failed")
            return {"success": False, "error": str(e)}

    elif name == "remove_files_from_history":
        try:
            adapter = create_adapter(args["repo_path"])
            return result_to_dict(adapter.remove_files(args["paths"], dry_run, not dry_run))
        except (ValueError, RuntimeError) as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.exception("remove_files_from_history failed")
            return {"success": False, "error": str(e)}

    elif name == "remove_large_files":
        try:
            adapter = create_adapter(args["repo_path"])
            return result_to_dict(adapter.remove_large_files(args.get("size_threshold_mb", 10.0), dry_run, not dry_run))
        except (ValueError, RuntimeError) as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.exception("remove_large_files failed")
            return {"success": False, "error": str(e)}

    elif name == "filter_paths":
        try:
            adapter = create_adapter(args["repo_path"])
            return result_to_dict(adapter.filter_paths(args.get("include_paths"), args.get("exclude_paths"), dry_run, not dry_run))
        except (ValueError, RuntimeError) as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.exception("filter_paths failed")
            return {"success": False, "error": str(e)}

    elif name == "create_backup":
        try:
            backup = create_adapter(args["repo_path"]).create_backup()
            return {"success": True, "backup_branch": backup, "message": f"Backup: {backup}"}
        except (ValueError, RuntimeError) as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.exception("create_backup failed")
            return {"success": False, "error": str(e)}

    elif name == "restore_backup":
        try:
            return result_to_dict(create_adapter(args["repo_path"]).restore_backup(args["backup_branch"]))
        except (ValueError, RuntimeError) as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.exception("restore_backup failed")
            return {"success": False, "error": str(e)}

    elif name == "get_commit_details":
        try:
            adapter = create_adapter(args["repo_path"])
            commits = adapter.get_commits(args["commit_hash"], max_count=1)
            if not commits:
                return {"success": False, "error": f"Commit not found: {args['commit_hash']}"}
            c = commits[0]
            return {
                "success": True,
                "commit": {
                    "hash": c.hash, "author_name": c.author_name, "author_email": c.author_email,
                    "committer_name": c.committer_name, "committer_email": c.committer_email,
                    "message": c.message, "date": c.date, "files": adapter.get_commit_files(c.hash),
                },
                "diff_summary": (adapter.get_commit_diff(c.hash) or "")[:2000] or None,
            }
        except (ValueError, RuntimeError) as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.exception("get_commit_details failed")
            return {"success": False, "error": str(e)}

    elif name == "rewrite_single_commit":
        try:
            adapter = create_adapter(args["repo_path"])
            commits = adapter.get_commits(args["commit_hash"], max_count=1)
            if not commits:
                return {"success": False, "error": f"Commit not found: {args['commit_hash']}"}

            commit = commits[0]
            new_message = args.get("new_message")

            if not new_message and args.get("use_ai"):
                ai_provider_name = args.get("ai_provider", config.ai.provider)
                provider = get_provider(
                    ai_provider_name,
                    model=args.get("ai_model", config.ai.model),
                    api_key=config.ai.openai_api_key
                    if ai_provider_name == "openai"
                    else config.ai.anthropic_api_key,
                    base_url=config.ai.ollama_base_url,
                )
                engine = AICommitEngine(provider, MessageStyle.CONVENTIONAL)
                try:
                    if hasattr(provider, "check_connection"):
                        connected, status = await provider.check_connection()
                        if not connected:
                            return {
                                "success": False,
                                "error": f"AI ({ai_provider_name}) connection failed: {status}",
                                "ai_provider": ai_provider_name,
                            }

                    files = adapter.get_commit_files(commit.hash)
                    result = await engine.rewrite_message(commit.message, commit.hash, files)
                    new_message = result.rewritten
                except AIConnectionError as e:
                    return {
                        "success": False,
                        "error": str(e),
                        "ai_provider": ai_provider_name,
                    }
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
        except (ValueError, RuntimeError) as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.exception("rewrite_single_commit failed")
            return {"success": False, "error": str(e)}

    elif name == "scan_secrets":
        try:
            return {"success": True, **create_adapter(args["repo_path"]).scan_secrets(args.get("branch", "HEAD"), args.get("max_commits", 100))}
        except (ValueError, RuntimeError) as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.exception("scan_secrets failed")
            return {"success": False, "error": str(e)}

    elif name == "squash_commits":
        try:
            adapter = create_adapter(args["repo_path"])
            backup = adapter.create_backup() if config.server.auto_backup and not dry_run else None
            result = adapter.squash_commits(args["start_commit"], args.get("end_commit", "HEAD"), args.get("new_message"), dry_run)
            response = result_to_dict(result)
            if backup:
                response["backup_branch"] = backup
            return response
        except (ValueError, RuntimeError) as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.exception("squash_commits failed")
            return {"success": False, "error": str(e)}

    elif name == "replace_text_in_history":
        try:
            adapter = create_adapter(args["repo_path"])
            backup = adapter.create_backup() if config.server.auto_backup and not dry_run else None
            result = adapter.replace_text_in_history(args["old_text"], args["new_text"], args.get("file_pattern"), dry_run, not dry_run)
            response = result_to_dict(result)
            if backup:
                response["backup_branch"] = backup
            return response
        except (ValueError, RuntimeError) as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.exception("replace_text_in_history failed")
            return {"success": False, "error": str(e)}

    elif name == "get_file_history":
        try:
            history = create_adapter(args["repo_path"]).get_file_history(args["file_path"])
            return {"success": True, "file_path": args["file_path"], "commits": history, "total_commits": len(history)}
        except (ValueError, RuntimeError) as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.exception("get_file_history failed")
            return {"success": False, "error": str(e)}

    elif name == "list_all_files_in_history":
        try:
            files = create_adapter(args["repo_path"]).list_all_files_in_history()
            return {"success": True, "files": files[:500], "total_files": len(files), "truncated": len(files) > 500}
        except (ValueError, RuntimeError) as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.exception("list_all_files_in_history failed")
            return {"success": False, "error": str(e)}

    elif name == "change_commit_dates":
        try:
            adapter = create_adapter(args["repo_path"])
            backup = adapter.create_backup() if config.server.auto_backup and not dry_run else None
            result = adapter.change_commit_dates(
                args.get("time_range", "evening"), args.get("weekend_only", False),
                args.get("preserve_order", True), args.get("start_date"),
                dry_run=dry_run,
                force=not dry_run,
            )

            response = result_to_dict(result)
            if backup:
                response["backup_branch"] = backup
            return response
        except (ValueError, RuntimeError) as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.exception("change_commit_dates failed")
            return {"success": False, "error": str(e)}

    return {"error": f"Unknown tool: {name}"}


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
