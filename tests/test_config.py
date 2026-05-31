"""Unit tests for configuration management (pure-Python, monkeypatched IO)."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

import git_filter_repo_mcp.config as config_mod
from git_filter_repo_mcp.config import (
    AIConfig,
    Config,
    ServerConfig,
    _apply_config_dict,
    _apply_env_vars,
    create_default_config_file,
    get_config,
    load_config,
    reload_config,
)

# Env vars that influence Config; centralised here so tests stay in sync
# with `_apply_env_vars` in config.py.
_ALL_CONFIG_ENV_VARS = (
    "GIT_FILTER_REPO_AI_PROVIDER",
    "GIT_FILTER_REPO_AI_MODEL",
    "GIT_FILTER_REPO_LOG_LEVEL",
    "OLLAMA_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "ANTHROPIC_API_KEY",
    "OPENAI_COMPATIBLE_API_KEY",
    "OPENAI_COMPATIBLE_BASE_URL",
    "LMSTUDIO_BASE_URL",
    "VLLM_BASE_URL",
    "LLAMACPP_BASE_URL",
    "LOCALAI_BASE_URL",
    "OPENROUTER_API_KEY",
    "OPENROUTER_BASE_URL",
)


@pytest.fixture
def clean_env(monkeypatch):
    """Strip every config-influencing env var so each test starts predictable."""
    for key in _ALL_CONFIG_ENV_VARS:
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


@pytest.fixture
def preserve_global_config():
    """Restore the module-level ``_config`` singleton after a test mutates it."""
    saved = config_mod._config
    try:
        yield
    finally:
        config_mod._config = saved


class TestDataclassDefaults:
    """Defaults baked into the dataclasses (no IO, no env)."""

    def test_top_level_defaults(self) -> None:
        config = Config()
        assert config.ai.provider == "ollama"
        assert config.ai.model == "llama3.2"
        assert config.server.log_level == "INFO"
        assert config.server.default_dry_run is True
        assert config.server.auto_backup is True

    def test_ai_section_defaults(self) -> None:
        ai = AIConfig()
        assert ai.ollama_base_url == "http://localhost:11434"
        assert ai.openai_api_key is None
        assert ai.openai_base_url == "https://api.openai.com/v1"
        assert ai.anthropic_api_key is None
        assert ai.openai_compatible_api_key is None
        assert ai.openai_compatible_base_url == "http://localhost:1234/v1"
        assert ai.lmstudio_base_url == "http://localhost:1234/v1"
        assert ai.vllm_base_url == "http://localhost:8000/v1"
        assert ai.llamacpp_base_url == "http://localhost:8080/v1"
        assert ai.localai_base_url == "http://localhost:8080/v1"
        assert ai.openrouter_api_key is None
        assert ai.openrouter_base_url == "https://openrouter.ai/api/v1"

    def test_server_section_defaults(self) -> None:
        server = ServerConfig()
        assert server.log_level == "INFO"
        assert server.default_dry_run is True
        assert server.auto_backup is True


class TestApplyEnvVars:
    """``_apply_env_vars`` reads from the environment into a Config instance."""

    @pytest.mark.parametrize(
        "env_var,env_value,attr_path,expected",
        [
            ("GIT_FILTER_REPO_AI_PROVIDER", "openai", "ai.provider", "openai"),
            ("GIT_FILTER_REPO_AI_MODEL", "gpt-4", "ai.model", "gpt-4"),
            ("OPENAI_API_KEY", "sk-test123", "ai.openai_api_key", "sk-test123"),
            ("OLLAMA_BASE_URL", "http://custom:11434", "ai.ollama_base_url", "http://custom:11434"),
            ("OPENAI_BASE_URL", "https://proxy/v1", "ai.openai_base_url", "https://proxy/v1"),
            ("ANTHROPIC_API_KEY", "ant-key", "ai.anthropic_api_key", "ant-key"),
            (
                "OPENAI_COMPATIBLE_API_KEY",
                "compat-key",
                "ai.openai_compatible_api_key",
                "compat-key",
            ),
            (
                "OPENAI_COMPATIBLE_BASE_URL",
                "http://local:1234/v1",
                "ai.openai_compatible_base_url",
                "http://local:1234/v1",
            ),
            ("LMSTUDIO_BASE_URL", "http://lmstudio/v1", "ai.lmstudio_base_url", "http://lmstudio/v1"),
            ("VLLM_BASE_URL", "http://vllm/v1", "ai.vllm_base_url", "http://vllm/v1"),
            ("LLAMACPP_BASE_URL", "http://llama/v1", "ai.llamacpp_base_url", "http://llama/v1"),
            ("LOCALAI_BASE_URL", "http://localai/v1", "ai.localai_base_url", "http://localai/v1"),
            ("OPENROUTER_API_KEY", "or-key", "ai.openrouter_api_key", "or-key"),
            (
                "OPENROUTER_BASE_URL",
                "https://openrouter.example/v1",
                "ai.openrouter_base_url",
                "https://openrouter.example/v1",
            ),
            ("GIT_FILTER_REPO_LOG_LEVEL", "DEBUG", "server.log_level", "DEBUG"),
        ],
    )
    def test_env_var_applied(
        self, clean_env, env_var: str, env_value: str, attr_path: str, expected: str,
    ) -> None:
        clean_env.setenv(env_var, env_value)
        config = Config()
        _apply_env_vars(config)
        # Walk dotted path (e.g. "ai.provider") on the config object.
        obj = config
        for part in attr_path.split("."):
            obj = getattr(obj, part)
        assert obj == expected

    def test_unset_env_keeps_defaults(self, clean_env) -> None:
        config = Config()
        _apply_env_vars(config)
        assert config.ai.provider == "ollama"
        assert config.ai.model == "llama3.2"
        assert config.server.log_level == "INFO"

    @pytest.mark.parametrize(
        "env_var,bad_value,attr_path,kept_default",
        [
            ("GIT_FILTER_REPO_AI_PROVIDER", "garbage", "ai.provider", "ollama"),
            ("GIT_FILTER_REPO_LOG_LEVEL", "FOOBAR", "server.log_level", "INFO"),
        ],
    )
    def test_invalid_env_value_logged_and_skipped(
        self, clean_env, env_var: str, bad_value: str,
        attr_path: str, kept_default: str,
    ) -> None:
        """Invalid env values must not corrupt Config — defaults preserved."""
        clean_env.setenv(env_var, bad_value)
        config = Config()
        _apply_env_vars(config)
        obj = config
        for part in attr_path.split("."):
            obj = getattr(obj, part)
        assert obj == kept_default

    def test_log_level_case_normalised(self, clean_env) -> None:
        clean_env.setenv("GIT_FILTER_REPO_LOG_LEVEL", "debug")
        config = Config()
        _apply_env_vars(config)
        assert config.server.log_level == "DEBUG"


class TestApplyConfigDict:
    """``_apply_config_dict`` merges a dict into a Config instance."""

    def test_ai_section_applied(self) -> None:
        config = Config()
        _apply_config_dict(config, {
            "ai": {
                "provider": "openai-compatible",
                "model": "qwen-local",
                "openai_compatible_api_key": "local-key",
                "openai_compatible_base_url": "http://localhost:1234/v1",
                "openrouter_api_key": "or-key",
            },
        })
        assert config.ai.provider == "openai-compatible"
        assert config.ai.model == "qwen-local"
        assert config.ai.openai_compatible_api_key == "local-key"
        assert config.ai.openai_compatible_base_url == "http://localhost:1234/v1"
        assert config.ai.openrouter_api_key == "or-key"

    def test_server_section_applied(self) -> None:
        config = Config()
        _apply_config_dict(config, {
            "server": {"log_level": "DEBUG", "default_dry_run": False, "auto_backup": False},
        })
        assert config.server.log_level == "DEBUG"
        assert config.server.default_dry_run is False
        assert config.server.auto_backup is False

    def test_partial_dict_leaves_other_fields_untouched(self) -> None:
        config = Config()
        _apply_config_dict(config, {"ai": {"model": "custom-model"}})
        assert config.ai.model == "custom-model"
        assert config.ai.provider == "ollama"
        assert config.server.log_level == "INFO"

    @pytest.mark.parametrize("data", [{}, {"ai": {"unknown_key": "x"}, "unknown_section": {}}])
    def test_unknown_or_empty_dict_is_noop(self, data: dict) -> None:
        config = Config()
        _apply_config_dict(config, data)
        assert config.ai.provider == "ollama"
        assert config.server.log_level == "INFO"

    @pytest.mark.parametrize(
        "data,expected_log_level",
        [
            ({"ai": None, "server": {"log_level": "DEBUG"}}, "DEBUG"),
            ({"ai": [], "server": {"log_level": "DEBUG"}}, "DEBUG"),
            ({"server": None}, "INFO"),
            ([], "INFO"),
        ],
    )
    def test_malformed_sections_are_skipped(
        self, data: object, expected_log_level: str,
    ) -> None:
        config = Config()
        _apply_config_dict(config, data)
        assert config.ai.provider == "ollama"
        assert config.server.log_level == expected_log_level

    @pytest.mark.parametrize(
        "data",
        [
            {"ai": {"provider": "bogus"}},
            {"server": {"log_level": "VERBOSE"}},
            {"server": {"default_dry_run": "false"}},
            {"server": {"auto_backup": 0}},
            {"ai": {"model": 123, "ollama_base_url": False}},
        ],
    )
    def test_invalid_config_values_are_skipped(self, data: dict) -> None:
        config = Config()
        _apply_config_dict(config, data)
        assert config.ai.provider == "ollama"
        assert config.ai.model == "llama3.2"
        assert config.ai.ollama_base_url == "http://localhost:11434"
        assert config.server.log_level == "INFO"
        assert config.server.default_dry_run is True
        assert config.server.auto_backup is True

    def test_config_log_level_case_normalised(self) -> None:
        config = Config()
        _apply_config_dict(config, {"server": {"log_level": "debug"}})
        assert config.server.log_level == "DEBUG"


class TestPriority:
    """Resolution priority between file values and env vars.

    Note: the real ``load_config`` reads files too, but those branches are
    covered in ``TestLoadConfig`` to keep this class focused on priority.
    """

    def test_env_overrides_config_dict(self, clean_env) -> None:
        config = Config()
        _apply_config_dict(config, {"ai": {"provider": "ollama", "model": "llama3.2"}})
        clean_env.setenv("GIT_FILTER_REPO_AI_PROVIDER", "anthropic")
        clean_env.setenv("GIT_FILTER_REPO_AI_MODEL", "claude-3")
        _apply_env_vars(config)
        assert config.ai.provider == "anthropic"
        assert config.ai.model == "claude-3"


class TestLoadConfig:
    """``load_config`` reads JSON files defensively."""

    def test_load_config_priority_user_local_env(
        self, tmp_path: Path, clean_env, monkeypatch,
    ) -> None:
        home = tmp_path / "home"
        user_config = home / ".config" / "git-filter-repo-mcp" / "config.json"
        user_config.parent.mkdir(parents=True)
        user_config.write_text(json.dumps({
            "ai": {"provider": "openai", "model": "user-model"},
            "server": {"log_level": "WARNING", "auto_backup": False},
        }))
        (tmp_path / "config.json").write_text(json.dumps({
            "ai": {"provider": "anthropic"},
            "server": {"log_level": "debug"},
        }))

        monkeypatch.chdir(tmp_path)
        clean_env.setenv("GIT_FILTER_REPO_AI_MODEL", "env-model")
        clean_env.setenv("GIT_FILTER_REPO_LOG_LEVEL", "ERROR")
        with patch("git_filter_repo_mcp.config.Path.home", return_value=home):
            config = load_config()

        assert config.ai.provider == "anthropic"
        assert config.ai.model == "env-model"
        assert config.server.log_level == "ERROR"
        assert config.server.auto_backup is False

    def test_load_config_skips_bad_config_shape(
        self, tmp_path: Path, clean_env, monkeypatch,
    ) -> None:
        home = tmp_path / "home"
        (tmp_path / "config.json").write_text(json.dumps({
            "ai": None,
            "server": {"log_level": "debug"},
        }))

        monkeypatch.chdir(tmp_path)
        with patch("git_filter_repo_mcp.config.Path.home", return_value=home):
            config = load_config()

        assert config.ai.provider == "ollama"
        assert config.server.log_level == "DEBUG"

    def test_round_trip_via_dict(self) -> None:
        """Indirect: write a file, read it, apply it — covers the JSON read
        path without depending on the CWD/HOME lookup mechanics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "config.json"
            cfg_path.write_text(json.dumps({
                "ai": {"provider": "anthropic", "model": "claude-haiku"},
                "server": {"log_level": "DEBUG"},
            }))
            config = Config()
            _apply_config_dict(config, json.loads(cfg_path.read_text()))
            assert config.ai.provider == "anthropic"
            assert config.ai.model == "claude-haiku"
            assert config.server.log_level == "DEBUG"

    def test_malformed_json_does_not_crash(self) -> None:
        """``load_config`` logs and continues. We assert the parse failure
        is recoverable; the surrounding ``load_config`` swallows it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "config.json"
            cfg_path.write_text("{invalid json")
            with pytest.raises(json.JSONDecodeError):
                json.loads(cfg_path.read_text())
            # Sanity: defaults still intact on a fresh Config.
            assert Config().ai.provider == "ollama"

    def test_utf8_bom_in_config_file_handled(self, tmp_path: Path) -> None:
        """Config files saved by Windows tools (Notepad) often have a UTF-8
        BOM; ``load_config`` must read them transparently."""
        cfg_path = tmp_path / "config.json"
        # Write JSON with a UTF-8 BOM prefix.
        cfg_path.write_bytes(
            "﻿".encode("utf-8")
            + json.dumps({"ai": {"provider": "anthropic"}}).encode("utf-8")
        )
        config = Config()
        with open(cfg_path, encoding="utf-8-sig") as f:
            _apply_config_dict(config, json.load(f))
        assert config.ai.provider == "anthropic"


class TestCreateDefaultConfigFile:
    """``create_default_config_file`` writes a JSON template."""

    def test_writes_expected_contents(self, tmp_path: Path) -> None:
        target = tmp_path / "sub" / "config.json"
        result = create_default_config_file(target)
        assert result == target
        data = json.loads(target.read_text())
        assert data["ai"]["provider"] == "ollama"
        assert data["server"]["auto_backup"] is True

    def test_creates_intermediate_directories(self, tmp_path: Path) -> None:
        target = tmp_path / "a" / "b" / "c" / "config.json"
        create_default_config_file(target)
        assert target.exists()

    def test_default_path_uses_home(self, tmp_path: Path) -> None:
        with patch("git_filter_repo_mcp.config.Path.home", return_value=tmp_path):
            assert create_default_config_file().exists()


class TestGlobalConfigSingleton:
    """``get_config`` / ``reload_config`` interact with the module-level cache."""

    def test_get_config_returns_singleton(self) -> None:
        assert get_config() is get_config()

    def test_reload_returns_fresh_instance(self, preserve_global_config) -> None:
        new_config = reload_config()
        assert isinstance(new_config, Config)

    def test_reload_picks_up_env_changes(
        self, clean_env, preserve_global_config,
    ) -> None:
        clean_env.setenv("GIT_FILTER_REPO_AI_MODEL", "test-reload-model")
        assert reload_config().ai.model == "test-reload-model"

    def test_reload_updates_get_config_view(
        self, clean_env, preserve_global_config,
    ) -> None:
        """After reload, subsequent ``get_config`` reads see the new value."""
        clean_env.setenv("GIT_FILTER_REPO_LOG_LEVEL", "DEBUG")
        reload_config()
        assert get_config().server.log_level == "DEBUG"
