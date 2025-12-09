"""Tests for configuration management."""

from git_filter_repo_mcp.config import (
    AIConfig,
    Config,
    ServerConfig,
    _apply_env_vars,
    get_config,
)


class TestConfig:
    """Test configuration classes."""

    def test_default_config(self):
        config = Config()
        assert config.ai.provider == "ollama"
        assert config.ai.model == "llama3.2"
        assert config.server.log_level == "INFO"
        assert config.server.default_dry_run is True
        assert config.server.auto_backup is True

    def test_ai_config_defaults(self):
        ai_config = AIConfig()
        assert ai_config.provider == "ollama"
        assert ai_config.ollama_base_url == "http://localhost:11434"
        assert ai_config.openai_api_key is None

    def test_server_config_defaults(self):
        server_config = ServerConfig()
        assert server_config.log_level == "INFO"
        assert server_config.default_dry_run is True
        assert server_config.auto_backup is True


class TestEnvVars:
    """Test environment variable configuration."""

    def test_ai_provider_env(self, monkeypatch):
        monkeypatch.setenv("GIT_FILTER_REPO_AI_PROVIDER", "openai")
        config = Config()
        _apply_env_vars(config)
        assert config.ai.provider == "openai"

    def test_ai_model_env(self, monkeypatch):
        monkeypatch.setenv("GIT_FILTER_REPO_AI_MODEL", "gpt-4")
        config = Config()
        _apply_env_vars(config)
        assert config.ai.model == "gpt-4"

    def test_openai_key_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test123")
        config = Config()
        _apply_env_vars(config)
        assert config.ai.openai_api_key == "sk-test123"

    def test_ollama_url_env(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://custom:11434")
        config = Config()
        _apply_env_vars(config)
        assert config.ai.ollama_base_url == "http://custom:11434"


class TestGetConfig:
    """Test config singleton."""

    def test_get_config_returns_config(self):
        config = get_config()
        assert isinstance(config, Config)

    def test_get_config_singleton(self):
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2
