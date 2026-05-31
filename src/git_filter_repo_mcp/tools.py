"""MCP tool definitions for git-filter-repo operations."""

import datetime
import re
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ErrorCode(str, Enum):
    """Structured error codes for MCP tool responses."""

    INVALID_INPUT = "INVALID_INPUT"
    REPO_NOT_FOUND = "REPO_NOT_FOUND"
    TOOL_NOT_FOUND = "TOOL_NOT_FOUND"
    COMMAND_FAILED = "COMMAND_FAILED"
    AI_CONNECTION_FAILED = "AI_CONNECTION_FAILED"
    NO_CHANGES = "NO_CHANGES"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class ToolCategory(str, Enum):
    """High-level tool grouping used for docs and registry metadata."""

    ANALYSIS = "analysis"
    AI = "ai"
    MODIFICATION = "modification"
    BACKUP = "backup"


@dataclass(frozen=True)
class ToolSpec:
    """Single source of truth for an MCP tool's public metadata."""

    name: str
    description: str
    input_model: type[BaseModel]
    category: ToolCategory
    destructive: bool = False

    def to_definition(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_model.model_json_schema(),
        }


# --- Base models ---
#
# Every tool input carries ``repo_path``; destructive tools additionally carry
# ``dry_run``. Centralising these avoids drift in description text and lets us
# add cross-cutting validation in one place.


class _RepoInput(BaseModel):
    """Base for any tool that targets a git repository."""

    model_config = ConfigDict(extra="forbid")

    repo_path: str = Field(
        min_length=1, description="Path to the git repository",
    )

    @field_validator("repo_path")
    @classmethod
    def _validate_repo_path(cls, value: str) -> str:
        return _non_blank(value, "repo_path")


class _DestructiveRepoInput(_RepoInput):
    """Base for tools that mutate history (must support ``dry_run``)."""

    dry_run: bool = Field(default=True, description="If true, only show what would be changed")


_AIProviderField = Field(
    default=None,
    description=(
        "AI provider: ollama, openai, anthropic, openai-compatible, "
        "lmstudio, vllm, llamacpp, localai, or openrouter "
        "(uses config default if not set)"
    ),
)
_AIModelField = Field(
    default=None, description="AI model to use (uses config default if not set)",
)
_AIBaseUrlField = Field(
    default=None,
    description=(
        "Override provider base URL for this call. Useful for local OpenAI-compatible "
        "servers such as LM Studio, vLLM, llama.cpp, or LocalAI."
    ),
)
AIProvider = Literal[
    "ollama",
    "openai",
    "anthropic",
    "openai-compatible",
    "lmstudio",
    "vllm",
    "llamacpp",
    "localai",
    "openrouter",
]

_BACKUP_BRANCH_RE = re.compile(r"^backup_\d{8}_\d{6}_\d{6}$")
_HEX_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{4,64}$")
_TIME_RANGE_PRESETS = {"evening", "night", "weekend", "random"}


def _non_blank(value: str, field_name: str) -> str:
    if not value or not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    if "\x00" in value:
        raise ValueError(f"{field_name} must not contain NUL bytes")
    return value


def _git_ref(value: str, field_name: str) -> str:
    _non_blank(value, field_name)
    if value.startswith("-"):
        raise ValueError(f"{field_name} must not start with '-'")
    if any(char in value for char in "\n\r\t"):
        raise ValueError(f"{field_name} must not contain control characters")
    return value


def _path_value(value: str, field_name: str = "path") -> str:
    _non_blank(value, field_name)
    if value.startswith("-"):
        raise ValueError(f"{field_name} must not start with '-'")
    if any(char in value for char in "\n\r\t"):
        raise ValueError(f"{field_name} must not contain control characters")
    return value


def _path_list(values: list[str] | None, field_name: str) -> list[str] | None:
    if values is None:
        return None
    return [_path_value(value, field_name) for value in values]


def _identity_value(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    _non_blank(value, field_name)
    if any(char in value for char in "<>\n\r\t"):
        raise ValueError(
            f"{field_name} must not contain angle brackets, tabs, or newlines"
        )
    return value


def _url_value(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    _non_blank(value, field_name)
    if any(char in value for char in "\n\r\t"):
        raise ValueError(f"{field_name} must not contain control characters")
    return value


# --- Tool inputs ---


class AnalyzeHistoryInput(_RepoInput):
    """Input for analyze_history tool."""

    branch: str = Field(default="HEAD", description="Branch to analyze")
    max_count: int = Field(default=100, gt=0, le=10000, description="Maximum number of commits to analyze")

    @field_validator("branch")
    @classmethod
    def _validate_branch(cls, value: str) -> str:
        return _git_ref(value, "branch")


class ValidateRepoSafetyInput(_RepoInput):
    """Input for validate_repo_safety tool."""


class FindLargeFilesInput(_RepoInput):
    """Input for find_large_files tool."""

    size_threshold_mb: float = Field(
        default=10.0,
        gt=0.0,
        description="Size threshold in MB - files larger than this will be reported",
    )
    limit: int = Field(
        default=100,
        gt=0,
        le=1000,
        description="Maximum number of large-file records to return",
    )


class ListBackupsInput(_RepoInput):
    """Input for list_backups tool."""

    limit: int = Field(
        default=100,
        gt=0,
        le=1000,
        description="Maximum number of backup branches to return",
    )


class ResolveCommitInput(_RepoInput):
    """Input for resolve_commit tool."""

    commit_ref: str = Field(
        min_length=1,
        description="Commit hash, abbreviated hash, branch, tag, or other git commit ref",
    )

    @field_validator("commit_ref")
    @classmethod
    def _validate_commit_ref(cls, value: str) -> str:
        return _git_ref(value, "commit_ref")


class ListAIProvidersInput(BaseModel):
    """Input for list_ai_providers tool."""

    model_config = ConfigDict(extra="forbid")


class CheckAIProviderInput(BaseModel):
    """Input for check_ai_provider tool."""

    model_config = ConfigDict(extra="forbid")

    ai_provider: AIProvider | None = _AIProviderField
    ai_model: str | None = _AIModelField
    ai_base_url: str | None = _AIBaseUrlField
    ai_temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Override generation temperature for the provider check metadata",
    )
    ai_max_tokens: int | None = Field(
        default=None,
        gt=0,
        le=4096,
        description="Override max tokens for generated commit messages",
    )
    check_connection: bool = Field(
        default=True,
        description="If false, only resolve provider configuration without contacting it",
    )

    @field_validator("ai_base_url")
    @classmethod
    def _validate_ai_base_url(cls, value: str | None) -> str | None:
        return _url_value(value, "ai_base_url")


class RewriteCommitMessagesInput(_DestructiveRepoInput):
    """Input for rewrite_commit_messages tool."""

    branch: str = Field(default="HEAD", description="Branch to rewrite")
    style: Literal["conventional", "gitmoji", "simple", "detailed"] = Field(
        default="conventional",
        description="Message style: conventional, gitmoji, simple, or detailed",
    )
    use_ai: bool = Field(default=False, description="Use AI to generate new messages")
    ai_provider: AIProvider | None = _AIProviderField
    ai_model: str | None = _AIModelField
    ai_base_url: str | None = _AIBaseUrlField
    ai_temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Override generation temperature",
    )
    ai_max_tokens: int | None = Field(
        default=None,
        gt=0,
        le=4096,
        description="Override max tokens for each generated commit message",
    )
    ai_check_connection: bool = Field(
        default=True,
        description="Check provider connectivity before generating messages",
    )
    ai_max_concurrency: int = Field(
        default=5,
        gt=0,
        le=20,
        description="Maximum concurrent AI requests when rewriting a batch",
    )
    max_commits: int | None = Field(
        default=None,
        gt=0,
        le=10000,
        description="Limit AI rewriting to the newest N commits on the selected branch",
    )
    manual_mappings: dict[str, str] | None = Field(
        default=None, description="Manual message mappings: {old_message: new_message}"
    )

    @field_validator("branch")
    @classmethod
    def _validate_branch(cls, value: str) -> str:
        return _git_ref(value, "branch")

    @field_validator("ai_base_url")
    @classmethod
    def _validate_ai_base_url(cls, value: str | None) -> str | None:
        return _url_value(value, "ai_base_url")


class ChangeAuthorInput(_DestructiveRepoInput):
    """Input for change_author tool."""

    old_email: str = Field(min_length=1, description="Email address to replace")
    new_name: str = Field(min_length=1, description="New author name")
    new_email: str = Field(min_length=1, description="New author email")

    @field_validator("old_email", "new_name", "new_email")
    @classmethod
    def _validate_identity(cls, value: str, info) -> str:
        return _identity_value(value, info.field_name) or value


class RemoveFilesInput(_DestructiveRepoInput):
    """Input for remove_files tool."""

    paths: list[str] = Field(
        min_length=1, description="List of file paths to remove from history",
    )

    @field_validator("paths")
    @classmethod
    def _validate_paths(cls, values: list[str]) -> list[str]:
        return _path_list(values, "path") or []


class RemoveLargeFilesInput(_DestructiveRepoInput):
    """Input for remove_large_files tool."""

    size_threshold_mb: float = Field(
        default=10.0, gt=0.0, description="Size threshold in MB - files larger than this will be removed"
    )


class FilterPathsInput(_DestructiveRepoInput):
    """Input for filter_paths tool."""

    include_paths: list[str] | None = Field(
        default=None, description="Paths to include (keep only these)"
    )
    exclude_paths: list[str] | None = Field(
        default=None, description="Paths to exclude (remove these)"
    )

    @field_validator("include_paths", "exclude_paths")
    @classmethod
    def _validate_paths(cls, values: list[str] | None) -> list[str] | None:
        return _path_list(values, "path")

    @model_validator(mode="after")
    def _validate_filter_mode(self) -> "FilterPathsInput":
        include_paths = self.include_paths or []
        exclude_paths = self.exclude_paths or []
        if not include_paths and not exclude_paths:
            raise ValueError("Provide include_paths or exclude_paths")
        if include_paths and exclude_paths:
            raise ValueError("Cannot use include_paths and exclude_paths together")
        return self


class CreateBackupInput(_RepoInput):
    """Input for create_backup tool."""


class RestoreBackupInput(_RepoInput):
    """Input for restore_backup tool."""

    backup_branch: str = Field(
        min_length=1, description="Name of the backup branch to restore",
    )

    @field_validator("backup_branch")
    @classmethod
    def _validate_backup_branch(cls, value: str) -> str:
        _git_ref(value, "backup_branch")
        if not _BACKUP_BRANCH_RE.fullmatch(value):
            raise ValueError(
                "backup_branch must match backup_YYYYMMDD_HHMMSS_ffffff"
            )
        return value


class GetCommitDetailsInput(_RepoInput):
    """Input for get_commit_details tool."""

    commit_hash: str = Field(
        min_length=1, description="Commit hash to get details for",
    )

    @field_validator("commit_hash")
    @classmethod
    def _validate_commit_ref(cls, value: str) -> str:
        return _git_ref(value, "commit_hash")


class RewriteSingleCommitInput(_DestructiveRepoInput):
    """Input for rewrite_single_commit tool."""

    commit_hash: str = Field(min_length=1, description="Commit hash to rewrite")
    new_message: str | None = Field(default=None, description="New commit message")
    new_author_name: str | None = Field(default=None, description="New author name")
    new_author_email: str | None = Field(default=None, description="New author email")
    use_ai: bool = Field(default=False, description="Use AI to generate message if not provided")
    ai_provider: AIProvider | None = _AIProviderField
    ai_model: str | None = _AIModelField
    ai_base_url: str | None = _AIBaseUrlField
    ai_temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Override generation temperature",
    )
    ai_max_tokens: int | None = Field(
        default=None,
        gt=0,
        le=4096,
        description="Override max tokens for the generated commit message",
    )
    ai_check_connection: bool = Field(
        default=True,
        description="Check provider connectivity before generating the message",
    )

    @field_validator("commit_hash")
    @classmethod
    def _validate_commit_hash(cls, value: str) -> str:
        _non_blank(value, "commit_hash")
        if not _HEX_COMMIT_RE.fullmatch(value):
            raise ValueError("commit_hash must be hexadecimal")
        return value

    @field_validator("new_author_name", "new_author_email")
    @classmethod
    def _validate_author_identity(cls, value: str | None, info) -> str | None:
        return _identity_value(value, info.field_name)

    @field_validator("ai_base_url")
    @classmethod
    def _validate_ai_base_url(cls, value: str | None) -> str | None:
        return _url_value(value, "ai_base_url")

    @model_validator(mode="after")
    def _validate_author_pair(self) -> "RewriteSingleCommitInput":
        if bool(self.new_author_email) != bool(self.new_author_name):
            missing = "new_author_name" if self.new_author_email else "new_author_email"
            raise ValueError(
                f"Both new_author_name and new_author_email are required (missing: {missing})"
            )
        return self


class ScanSecretsInput(_RepoInput):
    """Input for scan_secrets tool."""

    branch: str = Field(default="HEAD", description="Branch to scan")
    max_commits: int = Field(default=100, gt=0, le=10000, description="Maximum number of commits to scan")

    @field_validator("branch")
    @classmethod
    def _validate_branch(cls, value: str) -> str:
        return _git_ref(value, "branch")


class SquashCommitsInput(_DestructiveRepoInput):
    """Input for squash_commits tool."""

    start_commit: str = Field(min_length=1, description="Starting commit hash (exclusive)")
    end_commit: str = Field(
        default="HEAD", min_length=1, description="Ending commit hash (inclusive)",
    )
    new_message: str | None = Field(
        default=None, description="New commit message for squashed commit"
    )

    @field_validator("start_commit", "end_commit")
    @classmethod
    def _validate_refs(cls, value: str, info) -> str:
        return _git_ref(value, info.field_name)


class ChangeCommitDatesInput(_DestructiveRepoInput):
    """Input for change_commit_dates tool."""

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

    @field_validator("time_range")
    @classmethod
    def _validate_time_range(cls, value: str) -> str:
        _non_blank(value, "time_range")
        if value in _TIME_RANGE_PRESETS:
            return value
        if "-" not in value:
            raise ValueError(
                "time_range must be a preset or custom range like '18:00-22:00'"
            )
        try:
            start_str, end_str = value.split("-", 1)
            start_parts = start_str.strip().split(":")
            end_parts = end_str.strip().split(":")
            start_hour = int(start_parts[0])
            start_min = int(start_parts[1]) if len(start_parts) > 1 else 0
            end_hour = int(end_parts[0])
            end_min = int(end_parts[1]) if len(end_parts) > 1 else 0
        except (ValueError, IndexError) as exc:
            raise ValueError(
                "time_range must be a preset or custom range like '18:00-22:00'"
            ) from exc

        if not (
            0 <= start_hour <= 23
            and 0 <= end_hour <= 23
            and 0 <= start_min <= 59
            and 0 <= end_min <= 59
        ):
            raise ValueError("time_range hours must be 0-23 and minutes must be 0-59")
        return value

    @field_validator("start_date")
    @classmethod
    def _validate_start_date(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            datetime.date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("start_date must use YYYY-MM-DD format") from exc
        return value


class ReplaceTextInput(_DestructiveRepoInput):
    """Input for replace_text_in_history tool."""

    old_text: str = Field(min_length=1, description="Text to find and replace")
    new_text: str = Field(description="Replacement text")  # may be empty (deletion)
    file_pattern: str | None = Field(
        default=None, description="Glob pattern to filter files (e.g., '*.py')"
    )

    @field_validator("old_text", "new_text")
    @classmethod
    def _validate_text_no_newlines(cls, value: str, info) -> str:
        if info.field_name == "old_text":
            _non_blank(value, "old_text")
        if "\n" in value or "\r" in value:
            raise ValueError(f"{info.field_name} must not contain newlines")
        return value

    @field_validator("file_pattern")
    @classmethod
    def _validate_file_pattern(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _path_value(value, "file_pattern")


class GetFileHistoryInput(_RepoInput):
    """Input for get_file_history tool."""

    file_path: str = Field(min_length=1, description="Path to the file")

    @field_validator("file_path")
    @classmethod
    def _validate_file_path(cls, value: str) -> str:
        return _path_value(value, "file_path")


class ListAllFilesInput(_RepoInput):
    """Input for list_all_files_in_history tool."""


# Tool definitions for MCP registration. Keep all public tool metadata here so
# docs, server defaults, and tests can derive from one registry.
TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="analyze_git_history",
        category=ToolCategory.ANALYSIS,
        input_model=AnalyzeHistoryInput,
        description="""Analyze git repository history to understand commits, authors, and files.

Use this tool first to get an overview of the repository before making changes.
Returns statistics about commits, authors, and a preview of recent commits.""",
    ),
    ToolSpec(
        name="validate_repo_safety",
        category=ToolCategory.ANALYSIS,
        input_model=ValidateRepoSafetyInput,
        description="""Validate repository safety before running destructive history rewrites.

Reports current branch, HEAD, clean/dirty working tree status, upstream
tracking status, ahead/behind counts, backup branch count, and safety warnings.""",
    ),
    ToolSpec(
        name="find_large_files",
        category=ToolCategory.ANALYSIS,
        input_model=FindLargeFilesInput,
        description="""Find large files in git history without modifying the repository.

Use this read-only tool before remove_large_files to inspect paths and sizes
above a threshold.""",
    ),
    ToolSpec(
        name="list_backups",
        category=ToolCategory.BACKUP,
        input_model=ListBackupsInput,
        description="""List backup branches created by git-filter-repo-mcp.

Use this before restore_backup to choose an available backup branch.""",
    ),
    ToolSpec(
        name="resolve_commit",
        category=ToolCategory.ANALYSIS,
        input_model=ResolveCommitInput,
        description="""Resolve a commit ref to a full commit and preview its metadata.

Accepts a commit hash, abbreviated hash, branch, tag, or other git commit ref.""",
    ),
    ToolSpec(
        name="list_ai_providers",
        category=ToolCategory.AI,
        input_model=ListAIProvidersInput,
        description="""List AI providers supported by git-filter-repo-mcp.

Use this to discover local and third-party provider names before calling
rewrite_commit_messages or rewrite_single_commit with use_ai=true.""",
    ),
    ToolSpec(
        name="check_ai_provider",
        category=ToolCategory.AI,
        input_model=CheckAIProviderInput,
        description="""Resolve and optionally check an AI provider configuration.

Use this before AI-based rewrite tools to verify Ollama, LM Studio, vLLM,
llama.cpp, LocalAI, OpenAI, Anthropic, or OpenRouter settings without
modifying the repository.""",
    ),
    ToolSpec(
        name="rewrite_commit_messages",
        category=ToolCategory.MODIFICATION,
        input_model=RewriteCommitMessagesInput,
        destructive=True,
        description="""Rewrite commit messages in the repository history.

Can use AI (Ollama/OpenAI/Anthropic/OpenAI-compatible local or third-party
providers) to automatically generate better commit messages, or accept manual
mappings for specific messages.

Supports multiple styles:
- conventional: feat:, fix:, docs:, etc.
- gitmoji: with emoji prefixes
- simple: short descriptive messages
- detailed: with body and footer

IMPORTANT: Always use dry_run=true first to preview changes!""",
    ),
    ToolSpec(
        name="change_author",
        category=ToolCategory.MODIFICATION,
        input_model=ChangeAuthorInput,
        destructive=True,
        description="""Change author/committer information for commits matching an email address.

Use this to fix incorrect author information or standardize author names.

IMPORTANT: Always use dry_run=true first to preview changes!""",
    ),
    ToolSpec(
        name="remove_files_from_history",
        category=ToolCategory.MODIFICATION,
        input_model=RemoveFilesInput,
        destructive=True,
        description="""Remove specific files from the entire git history.

Use this to:
- Remove accidentally committed secrets
- Remove large files that shouldn't be in history
- Clean up sensitive data

IMPORTANT: Always use dry_run=true first to preview changes!""",
    ),
    ToolSpec(
        name="remove_large_files",
        category=ToolCategory.MODIFICATION,
        input_model=RemoveLargeFilesInput,
        destructive=True,
        description="""Find and remove files larger than a threshold from git history.

Useful for cleaning up repositories with accidentally committed large files.

IMPORTANT: Always use dry_run=true first to preview changes!""",
    ),
    ToolSpec(
        name="filter_paths",
        category=ToolCategory.MODIFICATION,
        input_model=FilterPathsInput,
        destructive=True,
        description="""Filter repository to include or exclude specific paths.

Use this to:
- Extract a subdirectory into its own repo
- Remove specific directories from history
- Keep only certain paths

IMPORTANT: Always use dry_run=true first to preview changes!""",
    ),
    ToolSpec(
        name="create_backup",
        category=ToolCategory.BACKUP,
        input_model=CreateBackupInput,
        description="""Create a backup branch before making changes.

Always recommended before any rewrite operation.
Returns the backup branch name for later restoration.""",
    ),
    ToolSpec(
        name="restore_backup",
        category=ToolCategory.BACKUP,
        input_model=RestoreBackupInput,
        description="""Restore repository from a backup branch.

Use this to undo changes made by rewrite operations.""",
    ),
    ToolSpec(
        name="get_commit_details",
        category=ToolCategory.ANALYSIS,
        input_model=GetCommitDetailsInput,
        description="""Get detailed information about a specific commit.

Returns the commit message, author, date, and files changed.""",
    ),
    ToolSpec(
        name="rewrite_single_commit",
        category=ToolCategory.MODIFICATION,
        input_model=RewriteSingleCommitInput,
        destructive=True,
        description="""Rewrite a single commit's message and/or author information.

Can optionally use AI to generate a new message based on the commit's changes.

IMPORTANT: Always use dry_run=true first to preview changes!""",
    ),
    ToolSpec(
        name="scan_secrets",
        category=ToolCategory.ANALYSIS,
        input_model=ScanSecretsInput,
        description="""Scan repository history for potential secrets and sensitive data.

Detects:
- API keys (AWS, OpenAI, Anthropic, Google, Stripe, etc.)
- Private keys and certificates
- Tokens (GitHub, Slack, JWT)
- Passwords in URLs or config files
- Sensitive file names (.env, credentials.json, etc.)

Returns findings with severity levels and redacted matches.""",
    ),
    ToolSpec(
        name="squash_commits",
        category=ToolCategory.MODIFICATION,
        input_model=SquashCommitsInput,
        destructive=True,
        description="""Squash multiple commits into a single commit.

Combines all commits between start_commit (exclusive) and end_commit (inclusive)
into one commit with a new message.

IMPORTANT: Always use dry_run=true first to preview changes!""",
    ),
    ToolSpec(
        name="replace_text_in_history",
        category=ToolCategory.MODIFICATION,
        input_model=ReplaceTextInput,
        destructive=True,
        description="""Replace text throughout the entire repository history.

Use this to:
- Remove accidentally committed secrets
- Update outdated URLs or references
- Fix consistent typos across history

IMPORTANT: Always use dry_run=true first to preview changes!""",
    ),
    ToolSpec(
        name="get_file_history",
        category=ToolCategory.ANALYSIS,
        input_model=GetFileHistoryInput,
        description="""Get the commit history for a specific file.

Shows all commits that modified the file, including renames.""",
    ),
    ToolSpec(
        name="list_all_files_in_history",
        category=ToolCategory.ANALYSIS,
        input_model=ListAllFilesInput,
        description="""List all files that have ever existed in the repository.

Includes files that were deleted in later commits.""",
    ),
    ToolSpec(
        name="change_commit_dates",
        category=ToolCategory.MODIFICATION,
        input_model=ChangeCommitDatesInput,
        destructive=True,
        description="""Change commit dates to different times (e.g., outside work hours).

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
    ),
)

TOOL_NAMES = frozenset(spec.name for spec in TOOL_SPECS)
DESTRUCTIVE_TOOL_NAMES = frozenset(spec.name for spec in TOOL_SPECS if spec.destructive)
TOOL_DEFINITIONS = [spec.to_definition() for spec in TOOL_SPECS]
