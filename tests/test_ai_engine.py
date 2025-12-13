"""AI engine tests."""

import pytest

from git_filter_repo_mcp.ai_engine import (
    AICommitEngine,
    AIConnectionError,
    AnthropicProvider,
    CommitContext,
    MessageStyle,
    OllamaProvider,
    OpenAIProvider,
    build_prompt,
    get_provider,
)


class TestOllamaProvider:
    def test_init_defaults(self):
        provider = OllamaProvider()
        assert provider.base_url == "http://localhost:11434"
        assert provider.model == "llama3.2"

    def test_init_custom(self):
        provider = OllamaProvider(base_url="http://custom:1234", model="custom-model")
        assert provider.base_url == "http://custom:1234"
        assert provider.model == "custom-model"

    def test_build_prompt_conventional(self):
        context = CommitContext(
            original_message="fix bug", commit_hash="abc123", files_changed=["main.py", "utils.py"]
        )
        prompt = build_prompt(context, MessageStyle.CONVENTIONAL)
        assert "fix bug" in prompt
        assert "main.py" in prompt
        assert "conventional" in prompt.lower() or "feat:" in prompt.lower()

    def test_build_prompt_gitmoji(self):
        context = CommitContext(
            original_message="add feature", commit_hash="abc123", files_changed=[]
        )
        prompt = build_prompt(context, MessageStyle.GITMOJI)
        assert "add feature" in prompt
        assert ":sparkles:" in prompt

    def test_parse_response_conventional_adds_prefix(self):
        provider = OllamaProvider()
        result = provider._parse_response("update config", MessageStyle.CONVENTIONAL)
        assert result.startswith("chore:")

    def test_parse_response_conventional_keeps_prefix(self):
        provider = OllamaProvider()
        result = provider._parse_response("feat: add new feature", MessageStyle.CONVENTIONAL)
        assert result == "feat: add new feature"

    def test_parse_response_strips_quotes(self):
        provider = OllamaProvider()
        result = provider._parse_response('"some message"', MessageStyle.SIMPLE)
        assert result == "some message"


class TestOpenAIProvider:
    def test_init(self):
        provider = OpenAIProvider(api_key="test-key")
        assert provider.api_key == "test-key"
        assert provider.model == "gpt-4o-mini"

    def test_init_custom_model(self):
        provider = OpenAIProvider(api_key="test-key", model="gpt-4")
        assert provider.model == "gpt-4"

    def test_build_prompt(self):
        context = CommitContext(original_message="test", commit_hash="abc123", files_changed=[])
        prompt = build_prompt(context, MessageStyle.SIMPLE)
        assert "test" in prompt


class TestAnthropicProvider:
    def test_init(self):
        provider = AnthropicProvider(api_key="test-key")
        assert provider.api_key == "test-key"
        assert provider.model == "claude-sonnet-4-20250514"

    def test_init_custom_model(self):
        provider = AnthropicProvider(api_key="test-key", model="claude-3-opus-20240229")
        assert provider.model == "claude-3-opus-20240229"

    def test_headers(self):
        provider = AnthropicProvider(api_key="test-key")
        headers = provider.client.headers
        assert headers["x-api-key"] == "test-key"
        assert headers["anthropic-version"] == "2024-01-01"

    def test_build_prompt(self):
        context = CommitContext(
            original_message="test message", commit_hash="abc123", files_changed=["file.py"]
        )
        prompt = build_prompt(context, MessageStyle.CONVENTIONAL)
        assert "test message" in prompt
        assert "file.py" in prompt


class TestGetProvider:
    def test_get_ollama_provider(self):
        provider = get_provider("ollama")
        assert isinstance(provider, OllamaProvider)

    def test_get_ollama_provider_custom(self):
        provider = get_provider("ollama", base_url="http://custom:1234", model="custom-model")
        assert isinstance(provider, OllamaProvider)
        assert provider.base_url == "http://custom:1234"
        assert provider.model == "custom-model"

    def test_get_openai_provider(self):
        provider = get_provider("openai", api_key="test-key")
        assert isinstance(provider, OpenAIProvider)

    def test_get_openai_provider_no_key(self):
        with pytest.raises(ValueError, match="OpenAI API key required"):
            get_provider("openai")

    def test_get_anthropic_provider(self):
        provider = get_provider("anthropic", api_key="test-key")
        assert isinstance(provider, AnthropicProvider)

    def test_get_anthropic_provider_custom_model(self):
        provider = get_provider("anthropic", api_key="test-key", model="claude-3-opus-20240229")
        assert isinstance(provider, AnthropicProvider)
        assert provider.model == "claude-3-opus-20240229"

    def test_get_anthropic_provider_no_key(self):
        with pytest.raises(ValueError, match="Anthropic API key required"):
            get_provider("anthropic")

    def test_get_unknown_provider(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("unknown")


class TestAICommitEngine:
    def test_init_default(self):
        engine = AICommitEngine()
        assert isinstance(engine.provider, OllamaProvider)
        assert engine.style == MessageStyle.CONVENTIONAL

    def test_init_custom(self):
        provider = OpenAIProvider(api_key="test")
        engine = AICommitEngine(provider=provider, style=MessageStyle.GITMOJI)
        assert engine.provider == provider
        assert engine.style == MessageStyle.GITMOJI


class TestMessageStyle:
    def test_values(self):
        assert MessageStyle.CONVENTIONAL.value == "conventional"
        assert MessageStyle.GITMOJI.value == "gitmoji"
        assert MessageStyle.SIMPLE.value == "simple"
        assert MessageStyle.DETAILED.value == "detailed"


class TestCommitContext:
    def test_minimal(self):
        ctx = CommitContext(original_message="test", commit_hash="abc123", files_changed=[])
        assert ctx.original_message == "test"
        assert ctx.commit_hash == "abc123"
        assert ctx.files_changed == []
        assert ctx.diff_summary is None
        assert ctx.author is None

    def test_full(self):
        ctx = CommitContext(
            original_message="test",
            commit_hash="abc123",
            files_changed=["a.py", "b.py"],
            diff_summary="Added 10 lines",
            author="Test User",
        )
        assert ctx.files_changed == ["a.py", "b.py"]
        assert ctx.diff_summary == "Added 10 lines"
        assert ctx.author == "Test User"


class TestAIConnectionError:
    def test_basic_error(self):
        error = AIConnectionError("Ollama", "Connection refused")
        assert "Ollama" in str(error)
        assert "Connection refused" in str(error)
        assert error.provider == "Ollama"

    def test_error_with_original(self):
        original = ConnectionError("Network unreachable")
        error = AIConnectionError("OpenAI", "API error", original)
        assert error.original_error == original
        assert error.provider == "OpenAI"


class TestProviderRaiseOnError:
    def test_ollama_raise_on_error_default(self):
        provider = OllamaProvider()
        assert provider.raise_on_error is True

    def test_ollama_raise_on_error_disabled(self):
        provider = OllamaProvider(raise_on_error=False)
        assert provider.raise_on_error is False

    def test_openai_raise_on_error_default(self):
        provider = OpenAIProvider(api_key="test")
        assert provider.raise_on_error is True

    def test_openai_raise_on_error_disabled(self):
        provider = OpenAIProvider(api_key="test", raise_on_error=False)
        assert provider.raise_on_error is False

    def test_anthropic_raise_on_error_default(self):
        provider = AnthropicProvider(api_key="test")
        assert provider.raise_on_error is True

    def test_anthropic_raise_on_error_disabled(self):
        provider = AnthropicProvider(api_key="test", raise_on_error=False)
        assert provider.raise_on_error is False


class TestProviderLastError:
    def test_ollama_last_error_initially_none(self):
        provider = OllamaProvider()
        assert provider._last_error is None

    def test_openai_last_error_initially_none(self):
        provider = OpenAIProvider(api_key="test")
        assert provider._last_error is None

    def test_anthropic_last_error_initially_none(self):
        provider = AnthropicProvider(api_key="test")
        assert provider._last_error is None
