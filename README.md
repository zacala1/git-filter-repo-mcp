# git-filter-repo-mcp

MCP server for git-filter-repo - AI-assisted git history rewriting.

## Features

- **Commit Rewriting** - AI-powered message rewriting (Ollama/OpenAI/Anthropic)
- **Author/Date Changes** - Bulk update author info, move commits to evenings/weekends
- **File Operations** - Remove files, large files, filter paths from history
- **Secret Scanning** - Detect API keys, tokens, credentials
- **Text Replacement** - Find and replace across all commits
- **Backup/Restore** - Auto-backup before destructive operations

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
| `ANTHROPIC_API_KEY` | Anthropic API key | - |

## Tools

### Analysis

- `analyze_git_history` - Commit stats, authors
- `get_commit_details` - View specific commit
- `get_file_history` - Track file changes
- `list_all_files_in_history` - List all files
- `scan_secrets` - Find credentials

### Modification

- `rewrite_commit_messages` - AI message rewriting
- `rewrite_single_commit` - Edit one commit
- `change_author` - Bulk author change
- `change_commit_dates` - Move to evenings/weekends
- `remove_files_from_history` - Delete files
- `remove_large_files` - Remove by size
- `filter_paths` - Include/exclude paths
- `replace_text_in_history` - Search and replace
- `squash_commits` - Merge commits

### Utility

- `create_backup` / `restore_backup`

## Usage Examples

```text
"Analyze /path/to/repo"
"Rewrite commits to conventional format"
"Remove secrets.json from history"
"Change author old@email.com to new@email.com"
"Move commits to evening hours"
"Find files larger than 10MB"
"Scan for API keys"
```

## Safety

1. Use `dry_run: true` first
2. Backups are auto-created
3. Use `git push --force-with-lease` after changes
4. Coordinate with team before shared branch changes

## Development

```bash
uv sync --all-extras
uv run pytest tests/ -v
uv run ruff check src/
```

## License

MIT
