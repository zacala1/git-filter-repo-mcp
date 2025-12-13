"""MCP tool definitions for git-filter-repo operations."""

from pydantic import BaseModel, Field


class AnalyzeHistoryInput(BaseModel):
    """Input for analyze_history tool."""

    repo_path: str = Field(description="Path to the git repository")
    branch: str = Field(default="HEAD", description="Branch to analyze")
    max_count: int = Field(default=100, description="Maximum number of commits to analyze")


class RewriteCommitMessagesInput(BaseModel):
    """Input for rewrite_commit_messages tool."""

    repo_path: str = Field(description="Path to the git repository")
    branch: str = Field(default="HEAD", description="Branch to rewrite")
    style: str = Field(
        default="conventional",
        description="Message style: conventional, gitmoji, simple, or detailed",
    )
    dry_run: bool = Field(default=True, description="If true, only show what would be changed")
    use_ai: bool = Field(default=True, description="Use AI to generate new messages")
    ai_provider: str = Field(default="ollama", description="AI provider: ollama or openai")
    ai_model: str = Field(default="llama3.2", description="AI model to use")
    manual_mappings: dict[str, str] | None = Field(
        default=None, description="Manual message mappings: {old_message: new_message}"
    )


class ChangeAuthorInput(BaseModel):
    """Input for change_author tool."""

    repo_path: str = Field(description="Path to the git repository")
    old_email: str = Field(description="Email address to replace")
    new_name: str = Field(description="New author name")
    new_email: str = Field(description="New author email")
    dry_run: bool = Field(default=True, description="If true, only show what would be changed")


class RemoveFilesInput(BaseModel):
    """Input for remove_files tool."""

    repo_path: str = Field(description="Path to the git repository")
    paths: list[str] = Field(description="List of file paths to remove from history")
    dry_run: bool = Field(default=True, description="If true, only show what would be changed")


class RemoveLargeFilesInput(BaseModel):
    """Input for remove_large_files tool."""

    repo_path: str = Field(description="Path to the git repository")
    size_threshold_mb: float = Field(
        default=10.0, description="Size threshold in MB - files larger than this will be removed"
    )
    dry_run: bool = Field(default=True, description="If true, only show what would be changed")


class FilterPathsInput(BaseModel):
    """Input for filter_paths tool."""

    repo_path: str = Field(description="Path to the git repository")
    include_paths: list[str] | None = Field(
        default=None, description="Paths to include (keep only these)"
    )
    exclude_paths: list[str] | None = Field(
        default=None, description="Paths to exclude (remove these)"
    )
    dry_run: bool = Field(default=True, description="If true, only show what would be changed")


class CreateBackupInput(BaseModel):
    """Input for create_backup tool."""

    repo_path: str = Field(description="Path to the git repository")


class RestoreBackupInput(BaseModel):
    """Input for restore_backup tool."""

    repo_path: str = Field(description="Path to the git repository")
    backup_branch: str = Field(description="Name of the backup branch to restore")


class GetCommitDetailsInput(BaseModel):
    """Input for get_commit_details tool."""

    repo_path: str = Field(description="Path to the git repository")
    commit_hash: str = Field(description="Commit hash to get details for")


class RewriteSingleCommitInput(BaseModel):
    """Input for rewrite_single_commit tool."""

    repo_path: str = Field(description="Path to the git repository")
    commit_hash: str = Field(description="Commit hash to rewrite")
    new_message: str | None = Field(default=None, description="New commit message")
    new_author_name: str | None = Field(default=None, description="New author name")
    new_author_email: str | None = Field(default=None, description="New author email")
    use_ai: bool = Field(default=False, description="Use AI to generate message if not provided")
    ai_provider: str = Field(default="ollama", description="AI provider: ollama or openai")
    ai_model: str = Field(default="llama3.2", description="AI model to use")
    dry_run: bool = Field(default=True, description="If true, only show what would be changed")


class ScanSecretsInput(BaseModel):
    """Input for scan_secrets tool."""

    repo_path: str = Field(description="Path to the git repository")
    branch: str = Field(default="HEAD", description="Branch to scan")
    max_commits: int = Field(default=100, description="Maximum number of commits to scan")


class SquashCommitsInput(BaseModel):
    """Input for squash_commits tool."""

    repo_path: str = Field(description="Path to the git repository")
    start_commit: str = Field(description="Starting commit hash (exclusive)")
    end_commit: str = Field(default="HEAD", description="Ending commit hash (inclusive)")
    new_message: str | None = Field(
        default=None, description="New commit message for squashed commit"
    )
    dry_run: bool = Field(default=True, description="If true, only show what would be changed")


class ChangeCommitDatesInput(BaseModel):
    """Input for change_commit_dates tool."""

    repo_path: str = Field(description="Path to the git repository")
    time_range: str = Field(
        default="evening",
        description="Time range preset: 'evening' (19:00-23:00), 'night' (22:00-02:00), "
        "'weekend' (10:00-22:00 on weekends), 'random' (any time), or custom like '18:00-22:00'",
    )
    weekend_only: bool = Field(
        default=False, description="If true, move all commits to weekends (Sat/Sun)"
    )
    preserve_order: bool = Field(
        default=True, description="If true, maintain relative commit order"
    )
    start_date: str | None = Field(
        default=None,
        description="Start date for the new commit range (YYYY-MM-DD). Defaults to original earliest commit date.",
    )
    dry_run: bool = Field(default=True, description="If true, only show what would be changed")


class ReplaceTextInput(BaseModel):
    """Input for replace_text_in_history tool."""

    repo_path: str = Field(description="Path to the git repository")
    old_text: str = Field(description="Text to find and replace")
    new_text: str = Field(description="Replacement text")
    file_pattern: str | None = Field(
        default=None, description="Glob pattern to filter files (e.g., '*.py')"
    )
    dry_run: bool = Field(default=True, description="If true, only show what would be changed")


class GetFileHistoryInput(BaseModel):
    """Input for get_file_history tool."""

    repo_path: str = Field(description="Path to the git repository")
    file_path: str = Field(description="Path to the file")


class ListAllFilesInput(BaseModel):
    """Input for list_all_files_in_history tool."""

    repo_path: str = Field(description="Path to the git repository")


# Tool definitions for MCP registration
TOOL_DEFINITIONS = [
    {
        "name": "analyze_git_history",
        "description": """Analyze git repository history to understand commits, authors, and files.

Use this tool first to get an overview of the repository before making changes.
Returns statistics about commits, authors, and a preview of recent commits.""",
        "inputSchema": AnalyzeHistoryInput.model_json_schema(),
    },
    {
        "name": "rewrite_commit_messages",
        "description": """Rewrite commit messages in the repository history.

Can use AI (Ollama/OpenAI) to automatically generate better commit messages,
or accept manual mappings for specific messages.

Supports multiple styles:
- conventional: feat:, fix:, docs:, etc.
- gitmoji: with emoji prefixes
- simple: short descriptive messages
- detailed: with body and footer

IMPORTANT: Always use dry_run=true first to preview changes!""",
        "inputSchema": RewriteCommitMessagesInput.model_json_schema(),
    },
    {
        "name": "change_author",
        "description": """Change author/committer information for commits matching an email address.

Use this to fix incorrect author information or standardize author names.

IMPORTANT: Always use dry_run=true first to preview changes!""",
        "inputSchema": ChangeAuthorInput.model_json_schema(),
    },
    {
        "name": "remove_files_from_history",
        "description": """Remove specific files from the entire git history.

Use this to:
- Remove accidentally committed secrets
- Remove large files that shouldn't be in history
- Clean up sensitive data

IMPORTANT: Always use dry_run=true first to preview changes!""",
        "inputSchema": RemoveFilesInput.model_json_schema(),
    },
    {
        "name": "remove_large_files",
        "description": """Find and remove files larger than a threshold from git history.

Useful for cleaning up repositories with accidentally committed large files.

IMPORTANT: Always use dry_run=true first to preview changes!""",
        "inputSchema": RemoveLargeFilesInput.model_json_schema(),
    },
    {
        "name": "filter_paths",
        "description": """Filter repository to include or exclude specific paths.

Use this to:
- Extract a subdirectory into its own repo
- Remove specific directories from history
- Keep only certain paths

IMPORTANT: Always use dry_run=true first to preview changes!""",
        "inputSchema": FilterPathsInput.model_json_schema(),
    },
    {
        "name": "create_backup",
        "description": """Create a backup branch before making changes.

Always recommended before any rewrite operation.
Returns the backup branch name for later restoration.""",
        "inputSchema": CreateBackupInput.model_json_schema(),
    },
    {
        "name": "restore_backup",
        "description": """Restore repository from a backup branch.

Use this to undo changes made by rewrite operations.""",
        "inputSchema": RestoreBackupInput.model_json_schema(),
    },
    {
        "name": "get_commit_details",
        "description": """Get detailed information about a specific commit.

Returns the commit message, author, date, and files changed.""",
        "inputSchema": GetCommitDetailsInput.model_json_schema(),
    },
    {
        "name": "rewrite_single_commit",
        "description": """Rewrite a single commit's message and/or author information.

Can optionally use AI to generate a new message based on the commit's changes.

IMPORTANT: Always use dry_run=true first to preview changes!""",
        "inputSchema": RewriteSingleCommitInput.model_json_schema(),
    },
    {
        "name": "scan_secrets",
        "description": """Scan repository history for potential secrets and sensitive data.

Detects:
- API keys (AWS, OpenAI, Anthropic, Google, Stripe, etc.)
- Private keys and certificates
- Tokens (GitHub, Slack, JWT)
- Passwords in URLs or config files
- Sensitive file names (.env, credentials.json, etc.)

Returns findings with severity levels and redacted matches.""",
        "inputSchema": ScanSecretsInput.model_json_schema(),
    },
    {
        "name": "squash_commits",
        "description": """Squash multiple commits into a single commit.

Combines all commits between start_commit (exclusive) and end_commit (inclusive)
into one commit with a new message.

IMPORTANT: Always use dry_run=true first to preview changes!""",
        "inputSchema": SquashCommitsInput.model_json_schema(),
    },
    {
        "name": "replace_text_in_history",
        "description": """Replace text throughout the entire repository history.

Use this to:
- Remove accidentally committed secrets
- Update outdated URLs or references
- Fix consistent typos across history

IMPORTANT: Always use dry_run=true first to preview changes!""",
        "inputSchema": ReplaceTextInput.model_json_schema(),
    },
    {
        "name": "get_file_history",
        "description": """Get the commit history for a specific file.

Shows all commits that modified the file, including renames.""",
        "inputSchema": GetFileHistoryInput.model_json_schema(),
    },
    {
        "name": "list_all_files_in_history",
        "description": """List all files that have ever existed in the repository.

Includes files that were deleted in later commits.""",
        "inputSchema": ListAllFilesInput.model_json_schema(),
    },
    {
        "name": "change_commit_dates",
        "description": """Change commit dates to different times (e.g., outside work hours).

Use this to:
- Move commits to evening hours (after work)
- Move commits to weekends only
- Randomize commit times within a range

Time range presets:
- 'evening': 19:00-23:00 on weekdays
- 'night': 22:00-02:00
- 'weekend': 10:00-22:00 on Sat/Sun only
- 'random': random times throughout the day
- Custom: specify like '18:00-22:00'

IMPORTANT: Always use dry_run=true first to preview changes!""",
        "inputSchema": ChangeCommitDatesInput.model_json_schema(),
    },
]
