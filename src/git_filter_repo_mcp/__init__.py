"""git-filter-repo MCP Server - AI-powered git history rewriting."""

__version__ = "0.1.0"

from .adapter import CommitInfo, FilterResult, GitFilterRepoAdapter
from .ai_engine import (
    AICommitEngine,
    AnthropicProvider,
    CommitContext,
    MessageStyle,
    OllamaProvider,
    OpenAIProvider,
    RewriteResult,
    get_provider,
)
from .config import AIConfig, Config, ServerConfig, get_config, reload_config
from .secrets import SecretFinding, SecretPattern, redact_secret, scan_content

__all__ = [
    # Version
    "__version__",
    # Adapter
    "GitFilterRepoAdapter",
    "FilterResult",
    "CommitInfo",
    # AI Engine
    "AICommitEngine",
    "MessageStyle",
    "CommitContext",
    "RewriteResult",
    "get_provider",
    "OllamaProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    # Config
    "Config",
    "AIConfig",
    "ServerConfig",
    "get_config",
    "reload_config",
    # Secrets
    "SecretPattern",
    "SecretFinding",
    "scan_content",
    "redact_secret",
]
