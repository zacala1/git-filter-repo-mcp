# git-filter-repo-mcp

MCP server for git-filter-repo - AI-assisted git history rewriting.

## Features

- **Commit Rewriting** - AI-powered message rewriting (Ollama/OpenAI/Anthropic/OpenAI-compatible local LLMs) or manual mappings
- **Author/Date Changes** - Bulk update author info, move commits to evenings/weekends
- **File Operations** - Remove files, large files, filter paths from history
- **Secret Scanning** - Detect API keys, tokens, credentials with severity levels
- **Text Replacement** - Find and replace across all commits
- **Backup/Restore** - Auto-backup before every destructive operation
- **Squash Commits** - Merge commit ranges into a single commit

## Requirements

- Python 3.10+
- git, git-filter-repo
- (Optional) Ollama, LM Studio, vLLM, llama.cpp, LocalAI, OpenAI, Anthropic, or OpenRouter

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
        "GIT_FILTER_REPO_AI_PROVIDER": "lmstudio",
        "GIT_FILTER_REPO_AI_MODEL": "local-model",
        "LMSTUDIO_BASE_URL": "http://localhost:1234/v1"
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
        "GIT_FILTER_REPO_AI_PROVIDER": "ollama",
        "GIT_FILTER_REPO_AI_MODEL": "llama3.2"
      }
    }
  }
}
```

Set `GIT_FILTER_REPO_AI_PROVIDER=none` if you want to disable AI rewrites.

### Local LLM Token Offload

When `rewrite_commit_messages` or `rewrite_single_commit` runs with
`use_ai: true`, this MCP server calls the configured provider directly. The MCP
host still spends tokens to request the tool and read the result, but commit
message generation is handled by your local or third-party LLM instead of the
host model.

For commit-message rewrite calls, the provider receives the original commit
message and changed file paths. Full file contents are not sent by the batch
rewrite flow.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GIT_FILTER_REPO_AI_PROVIDER` | `ollama`, `openai`, `anthropic`, `openai-compatible`, `lmstudio`, `vllm`, `llamacpp`, `localai`, `openrouter`, `none` | `ollama` |
| `GIT_FILTER_REPO_AI_MODEL` | Model name | `llama3.2` |
| `GIT_FILTER_REPO_LOG_LEVEL` | `DEBUG`, `INFO`, `WARNING`, `ERROR` | `INFO` |
| `OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `OPENAI_BASE_URL` | Official OpenAI API base URL | `https://api.openai.com/v1` |
| `ANTHROPIC_API_KEY` | Anthropic API key | - |
| `OPENAI_COMPATIBLE_BASE_URL` | Generic OpenAI-compatible API base URL | `http://localhost:1234/v1` |
| `OPENAI_COMPATIBLE_API_KEY` | Optional key for OpenAI-compatible APIs | - |
| `LMSTUDIO_BASE_URL` | LM Studio local server URL | `http://localhost:1234/v1` |
| `VLLM_BASE_URL` | vLLM OpenAI-compatible server URL | `http://localhost:8000/v1` |
| `LLAMACPP_BASE_URL` | llama.cpp OpenAI-compatible server URL | `http://localhost:8080/v1` |
| `LOCALAI_BASE_URL` | LocalAI OpenAI-compatible server URL | `http://localhost:8080/v1` |
| `OPENROUTER_API_KEY` | OpenRouter API key | - |
| `OPENROUTER_BASE_URL` | OpenRouter API base URL | `https://openrouter.ai/api/v1` |

## AI Provider Setup

### Ollama

```json
{
  "GIT_FILTER_REPO_AI_PROVIDER": "ollama",
  "GIT_FILTER_REPO_AI_MODEL": "llama3.2",
  "OLLAMA_BASE_URL": "http://localhost:11434"
}
```

### LM Studio

Start the local server in LM Studio, then use:

```json
{
  "GIT_FILTER_REPO_AI_PROVIDER": "lmstudio",
  "GIT_FILTER_REPO_AI_MODEL": "local-model",
  "LMSTUDIO_BASE_URL": "http://localhost:1234/v1"
}
```

### Generic OpenAI-Compatible Endpoint

Use this for local or hosted services that expose `/v1/chat/completions`:

```json
{
  "GIT_FILTER_REPO_AI_PROVIDER": "openai-compatible",
  "GIT_FILTER_REPO_AI_MODEL": "qwen2.5-coder",
  "OPENAI_COMPATIBLE_BASE_URL": "http://localhost:1234/v1"
}
```

Set `OPENAI_COMPATIBLE_API_KEY` if the endpoint requires auth.

### OpenRouter

```json
{
  "GIT_FILTER_REPO_AI_PROVIDER": "openrouter",
  "GIT_FILTER_REPO_AI_MODEL": "openai/gpt-4o-mini",
  "OPENROUTER_API_KEY": "sk-or-..."
}
```

### Per-Call Overrides

The rewrite tools can override config for a single MCP call:

```json
{
  "repo_path": "/path/to/repo",
  "use_ai": true,
  "ai_provider": "lmstudio",
  "ai_model": "local-model",
  "ai_base_url": "http://localhost:1234/v1",
  "ai_temperature": 0.2,
  "ai_max_tokens": 120,
  "dry_run": true
}
```

## MCP Workflow

Typical local-LLM rewrite flow:

```text
1. "Validate repo safety before rewriting /path/to/repo"
2. "List git-filter-repo MCP AI providers"
3. "Check the LM Studio AI provider"
4. "Rewrite commit messages in /path/to/repo using AI, dry run first"
5. "Apply the rewrite if the preview is correct"
```

Useful providers:

| Provider | Base URL | Notes |
|----------|----------|-------|
| `ollama` | `http://localhost:11434` | Uses Ollama `/api/generate` |
| `lmstudio` | `http://localhost:1234/v1` | API key usually not required |
| `vllm` | `http://localhost:8000/v1` | OpenAI-compatible |
| `llamacpp` | `http://localhost:8080/v1` | llama.cpp server |
| `localai` | `http://localhost:8080/v1` | OpenAI-compatible |
| `openai-compatible` | configurable | Generic local or third-party endpoint |
| `openrouter` | `https://openrouter.ai/api/v1` | Requires `OPENROUTER_API_KEY` |

## Tools (22)

### Analysis (read-only)

| Tool | Description |
|------|-------------|
| `analyze_git_history` | Commit stats, author breakdown, recent commits |
| `validate_repo_safety` | Check branch, HEAD, clean worktree, upstream, backups before rewrites |
| `find_large_files` | Read-only large file discovery before removal |
| `resolve_commit` | Resolve a commit ref to a full hash and metadata |
| `get_commit_details` | Full details for a specific commit |
| `get_file_history` | Commit history for a specific file |
| `list_all_files_in_history` | All files that ever existed in the repo |
| `scan_secrets` | Detect API keys, tokens, credentials |

### AI Provider Tools

| Tool | Description |
|------|-------------|
| `list_ai_providers` | Show supported local and hosted AI providers |
| `check_ai_provider` | Verify AI provider configuration before generation |

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

#### AI Rewrite Options

`rewrite_commit_messages` supports:

| Argument | Description |
|----------|-------------|
| `use_ai` | Enable provider-backed message generation |
| `ai_provider` | Provider override for this call |
| `ai_model` | Model override for this call |
| `ai_base_url` | Endpoint override for Ollama/OpenAI-compatible providers |
| `ai_temperature` | Generation temperature, `0.0` to `2.0` |
| `ai_max_tokens` | Max tokens per generated commit message |
| `ai_check_connection` | Check provider before generation |
| `ai_max_concurrency` | Concurrent AI requests for batch rewrites |
| `max_commits` | Limit rewrite generation to the newest N commits |
| `style` | `conventional`, `gitmoji`, `simple`, or `detailed` |

`rewrite_single_commit` supports the same AI provider overrides except
`ai_max_concurrency` and `max_commits`.

### Backup

| Tool | Description |
|------|-------------|
| `create_backup` | Create a backup branch |
| `list_backups` | List available backup branches |
| `restore_backup` | Restore from a backup branch |

## Usage Examples

```text
"Analyze /path/to/repo"
"Validate repo safety before rewriting"
"Find large files over 25MB"
"Check the LM Studio AI provider"
"Rewrite commits to conventional format"
"Remove secrets.json from history"
"Change author old@email.com to new@email.com"
"Move commits to evening hours"
"Find files larger than 10MB"
"Scan for API keys"
"List backup branches"
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
uv run pytest tests/ -v       # 542+ tests
uv run ruff check src/ tests/ # lint
uv run pyright src/            # type check
```

## License

MIT
