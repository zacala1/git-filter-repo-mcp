# git-filter-repo-mcp

An MCP (Model Context Protocol) server wrapping git-filter-repo for AI-assisted git history rewriting.

Tell your LLM things like "rewrite commit messages to conventional format" or "remove secrets.json from history" and it handles the rest.

## Features

- **Commit Message Rewriting** - Rewrite messages with AI (Ollama/OpenAI/Anthropic) or manual mappings
- **Author Changes** - Bulk update author name/email across history
- **File Removal** - Completely remove files from history (secrets, large files)
- **Path Filtering** - Extract subdirectories or exclude specific paths
- **Large File Cleanup** - Find and remove files exceeding size threshold
- **Secret Scanning** - Detect API keys, tokens, passwords in history
- **Text Replacement** - Find and replace text across all commits
- **Commit Squashing** - Combine multiple commits into one
- **Backup/Restore** - Auto-backup before destructive operations

## Requirements

- Python 3.10+
- git
- (Optional) Ollama for local AI, or OpenAI/Anthropic API key

## Installation

```bash
git clone https://github.com/zacala1/git-filter-repo-mcp.git
cd git-filter-repo-mcp
uv pip install -e .
```

Verify:

```bash
git-filter-repo --version
uv run pytest tests/ -v
```

## MCP Setup

Add to your MCP client config:

```json
{
  "mcpServers": {
    "git-filter-repo": {
      "command": "uv",
      "args": ["run", "git-filter-repo-mcp"],
      "cwd": "/path/to/git-filter-repo-mcp"
    }
  }
}
```

## Tools

### Analysis

- `analyze_git_history` - Get commit stats, authors, file counts
- `get_commit_details` - View specific commit info
- `get_file_history` - Track file changes across commits
- `list_all_files_in_history` - List all files ever in repo
- `scan_secrets` - Find API keys, tokens, credentials

### Modification

- `rewrite_commit_messages` - AI-powered message rewriting
- `rewrite_single_commit` - Edit one commit's message/author
- `change_author` - Bulk author email/name change
- `remove_files_from_history` - Delete files from all commits
- `remove_large_files` - Remove files over size limit
- `filter_paths` - Keep or exclude paths
- `replace_text_in_history` - Search and replace in history
- `squash_commits` - Merge commits into one

### Utility

- `create_backup` - Create backup branch
- `restore_backup` - Restore from backup

## Usage

```
"Analyze /path/to/repo history"
"Rewrite commits to conventional format"
"Remove secrets.json from history"
"Change author from old@email.com to new@email.com"
"Find files larger than 10MB"
"Scan for leaked API keys"
"Squash last 5 commits"
```

## Safety

1. Always use `dry_run: true` first
2. Create backups before changes
3. Use `git push --force-with-lease` after rewriting
4. Coordinate with team before shared branch changes

## Configuration

Environment variables:

```bash
export GIT_FILTER_REPO_AI_PROVIDER=ollama
export GIT_FILTER_REPO_AI_MODEL=llama3.2
export OLLAMA_BASE_URL=http://localhost:11434
export OPENAI_API_KEY=sk-xxx
export ANTHROPIC_API_KEY=sk-ant-xxx
```

Or config file at `~/.config/git-filter-repo-mcp/config.json`:

```json
{
  "ai": {
    "provider": "ollama",
    "model": "llama3.2"
  },
  "server": {
    "auto_backup": true
  }
}
```

## AI Setup

**Ollama (local):**

```bash
ollama pull llama3.2
ollama serve
```

**OpenAI:** Set `OPENAI_API_KEY` env var.

**Anthropic:** Set `ANTHROPIC_API_KEY` env var.

## Development

```bash
uv pip install -e ".[dev]"
uv run pytest tests/ -v
ruff check src/
```

## License

MIT
