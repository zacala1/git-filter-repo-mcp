# git-filter-repo-mcp

MCP server for git-filter-repo - AI-assisted git history rewriting.

## Features

- **Commit Rewriting** - AI-powered message rewriting (Ollama/OpenAI/Anthropic) or manual mappings
- **Author/Date Changes** - Bulk update author info, move commits to evenings/weekends
- **File Operations** - Remove files, large files, filter paths from history
- **Secret Scanning** - Detect API keys, tokens, credentials with severity levels
- **Text Replacement** - Find and replace across all commits
- **Backup/Restore** - Auto-backup before every destructive operation
- **Squash Commits** - Merge commit ranges into a single commit

## Requirements

- Python 3.10+
- git, git-filter-repo
- (Optional) Ollama / OpenAI / Anthropic API key

## Installation

```bash
git clone https://github.com/zacala1/git-filter-repo-mcp.git
cd git-filter-repo-mcp
uv sync
```

## MCP Setup

Add to Claude Desktop config:

**Windows** (`%APPDATA%\Claude\claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "git-filter-repo": {
      "command": "uv",
      "args": ["--directory", "C:\\path\\to\\git-filter-repo-mcp", "run", "git-filter-repo-mcp"],
      "env": {
        "GIT_FILTER_REPO_AI_PROVIDER": "none"
      }
    }
  }
}
```

**macOS/Linux** (`~/.config/claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "git-filter-repo": {
      "command": "uv",
      "args": ["--directory", "/path/to/git-filter-repo-mcp", "run", "git-filter-repo-mcp"],
      "env": {
        "GIT_FILTER_REPO_AI_PROVIDER": "none"
      }
    }
  }
}
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GIT_FILTER_REPO_AI_PROVIDER` | `ollama`, `openai`, `anthropic`, `none` | `ollama` |
| `GIT_FILTER_REPO_AI_MODEL` | Model name | `llama3.2` |
| `GIT_FILTER_REPO_LOG_LEVEL` | `DEBUG`, `INFO`, `WARNING`, `ERROR` | `INFO` |
| `OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `OPENAI_BASE_URL` | OpenAI-compatible API base URL | `https://api.openai.com/v1` |
| `ANTHROPIC_API_KEY` | Anthropic API key | - |

## Tools (16)

### Analysis (read-only)

| Tool | Description |
|------|-------------|
| `analyze_git_history` | Commit stats, author breakdown, recent commits |
| `get_commit_details` | Full details for a specific commit |
| `get_file_history` | Commit history for a specific file |
| `list_all_files_in_history` | All files that ever existed in the repo |
| `scan_secrets` | Detect API keys, tokens, credentials |

### Modification (destructive - use `dry_run: true` first)

| Tool | Description |
|------|-------------|
| `rewrite_commit_messages` | AI or manual message rewriting |
| `rewrite_single_commit` | Edit one commit's message/author |
| `change_author` | Bulk author/email change |
| `change_commit_dates` | Move commits to evenings/weekends |
| `remove_files_from_history` | Delete files from all history |
| `remove_large_files` | Remove files above a size threshold |
| `filter_paths` | Keep or exclude specific paths |
| `replace_text_in_history` | Search and replace across history |
| `squash_commits` | Merge a commit range into one |

### Backup

| Tool | Description |
|------|-------------|
| `create_backup` | Create a backup branch |
| `restore_backup` | Restore from a backup branch |

## Usage Examples

```text
"Analyze /path/to/repo"
"Rewrite commits to conventional format"
"Remove secrets.json from history"
"Change author old@email.com to new@email.com"
"Move commits to evening hours"
"Find files larger than 10MB"
"Scan for API keys"
"Squash the last 3 commits"
```

## Safety

- All destructive tools default to `dry_run: true`
- Backups are automatically created **before** every destructive operation
- Input validation blocks injection attempts (dash-prefixed args, invalid hashes)
- Timeout protection prevents hanging on large repos
- Use `git push --force-with-lease` after changes
- Coordinate with team before shared branch changes

## Development

```bash
uv sync --all-extras
uv run pytest tests/ -v       # 325 tests
uv run ruff check src/ tests/ # lint
uv run pyright src/            # type check
```

## License

MIT
