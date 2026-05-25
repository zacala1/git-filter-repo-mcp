"""Configuration management for git-filter-repo-mcp."""

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)


@dataclass
class AIConfig:
    """AI provider configuration."""

    provider: Literal["ollama", "openai", "anthropic", "none"] = "ollama"
    model: str = "llama3.2"

    # Ollama settings
    ollama_base_url: str = "http://localhost:11434"

    # OpenAI settings
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"

    # Anthropic settings
    anthropic_api_key: str | None = None


@dataclass
class ServerConfig:
    """Server configuration."""

    log_level: str = "INFO"
    default_dry_run: bool = True
    auto_backup: bool = True


@dataclass
class Config:
    """Main configuration."""

    ai: AIConfig = field(default_factory=AIConfig)
    server: ServerConfig = field(default_factory=ServerConfig)


def load_config() -> Config:
    """
    Load configuration from multiple sources (in priority order, highest first):
    1. Environment variables (highest priority)
    2. Local config file (./config.json) — project-level override
    3. User config file (~/.config/git-filter-repo-mcp/config.json)
    4. Default values (lowest priority)

    This mirrors typical conventions (e.g. git, npm) where project-local
    settings override the user-global ones.
    """
    config = Config()

    # Listed lowest-priority first so the loop below applies them in order
    # and later entries override earlier ones.
    config_paths = [
        Path.home() / ".config" / "git-filter-repo-mcp" / "config.json",
        Path("./config.json"),
    ]

    for config_path in config_paths:
        try:
            if not config_path.exists():
                continue
            # ``utf-8-sig`` transparently strips a UTF-8 BOM (common when the
            # file was edited with Notepad on Windows).
            with open(config_path, encoding="utf-8-sig") as f:
                data = json.load(f)
                _apply_config_dict(config, data)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse config file %s: %s", config_path, e)
        except OSError as e:
            # IOError is an alias for OSError; also covers broken symlinks etc.
            logger.warning("Failed to read config file %s: %s", config_path, e)

    _apply_env_vars(config)

    return config


def _apply_config_dict(config: Config, data: dict) -> None:
    """Apply configuration from a dictionary."""
    if "ai" in data:
        ai_data = data["ai"]
        if "provider" in ai_data:
            config.ai.provider = ai_data["provider"]
        if "model" in ai_data:
            config.ai.model = ai_data["model"]
        if "ollama_base_url" in ai_data:
            config.ai.ollama_base_url = ai_data["ollama_base_url"]
        if "openai_api_key" in ai_data:
            config.ai.openai_api_key = ai_data["openai_api_key"]
        if "openai_base_url" in ai_data:
            config.ai.openai_base_url = ai_data["openai_base_url"]
        if "anthropic_api_key" in ai_data:
            config.ai.anthropic_api_key = ai_data["anthropic_api_key"]

    if "server" in data:
        server_data = data["server"]
        if "log_level" in server_data:
            config.server.log_level = server_data["log_level"]
        if "default_dry_run" in server_data:
            config.server.default_dry_run = server_data["default_dry_run"]
        if "auto_backup" in server_data:
            config.server.auto_backup = server_data["auto_backup"]


_VALID_AI_PROVIDERS = {"ollama", "openai", "anthropic", "none"}
_VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def _apply_env_vars(config: Config) -> None:
    """Apply environment variables to config.

    Invalid values are logged and skipped rather than silently accepted —
    downstream code annotates ``ai.provider`` as a ``Literal`` and
    ``log_level`` as something ``logging.setLevel`` accepts, so an unchecked
    typo would fail much later with a confusing error.
    """
    if provider := os.getenv("GIT_FILTER_REPO_AI_PROVIDER"):
        if provider in _VALID_AI_PROVIDERS:
            config.ai.provider = provider  # type: ignore[assignment]
        else:
            logger.warning(
                "Ignoring GIT_FILTER_REPO_AI_PROVIDER=%r (valid: %s)",
                provider, sorted(_VALID_AI_PROVIDERS),
            )

    if model := os.getenv("GIT_FILTER_REPO_AI_MODEL"):
        config.ai.model = model

    if ollama_url := os.getenv("OLLAMA_BASE_URL"):
        config.ai.ollama_base_url = ollama_url

    if openai_key := os.getenv("OPENAI_API_KEY"):
        config.ai.openai_api_key = openai_key

    if openai_url := os.getenv("OPENAI_BASE_URL"):
        config.ai.openai_base_url = openai_url

    if anthropic_key := os.getenv("ANTHROPIC_API_KEY"):
        config.ai.anthropic_api_key = anthropic_key

    if log_level := os.getenv("GIT_FILTER_REPO_LOG_LEVEL"):
        normalised = log_level.upper()
        if normalised in _VALID_LOG_LEVELS:
            config.server.log_level = normalised
        else:
            logger.warning(
                "Ignoring GIT_FILTER_REPO_LOG_LEVEL=%r (valid: %s)",
                log_level, sorted(_VALID_LOG_LEVELS),
            )


def create_default_config_file(path: Path | None = None) -> Path:
    """Create a default configuration file."""
    if path is None:
        path = Path.home() / ".config" / "git-filter-repo-mcp" / "config.json"

    path.parent.mkdir(parents=True, exist_ok=True)

    default_config = {
        "ai": {
            "provider": "ollama",
            "model": "llama3.2",
            "ollama_base_url": "http://localhost:11434",
            "openai_api_key": None,
            "openai_base_url": "https://api.openai.com/v1",
            "anthropic_api_key": None,
        },
        "server": {"log_level": "INFO", "default_dry_run": True, "auto_backup": True},
    }

    with open(path, "w") as f:
        json.dump(default_config, f, indent=2)

    return path


# Thread-safe global config instance
_config: Config | None = None
_config_lock = threading.Lock()


def get_config() -> Config:
    """Get the global configuration instance (thread-safe)."""
    global _config
    if _config is None:
        with _config_lock:
            # Double-check locking pattern
            if _config is None:
                _config = load_config()
    return _config


def reload_config() -> Config:
    """Reload configuration from sources (thread-safe)."""
    global _config
    with _config_lock:
        _config = load_config()
        return _config
