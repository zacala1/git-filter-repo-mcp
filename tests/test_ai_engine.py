"""AI engine tests."""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from git_filter_repo_mcp.ai_engine import (
    AICommitEngine,
    AIConnectionError,
    AnthropicProvider,
    CommitContext,
    MessageStyle,
    OllamaProvider,
    OpenAIProvider,
    RewriteResult,
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
        assert provider.base_url == "https://api.openai.com/v1"

    def test_init_custom_model(self):
        provider = OpenAIProvider(api_key="test-key", model="gpt-4")
        assert provider.model == "gpt-4"

    def test_init_custom_base_url(self):
        provider = OpenAIProvider(api_key="test-key", base_url="https://custom.api.com/v1")
        assert provider.base_url == "https://custom.api.com/v1"

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
        assert headers["anthropic-version"] == "2023-06-01"

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

    def test_get_openai_provider_custom_base_url(self):
        provider = get_provider("openai", api_key="test-key", base_url="https://custom.com/v1")
        assert isinstance(provider, OpenAIProvider)
        assert provider.base_url == "https://custom.com/v1"

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


class TestRewriteBatch:
    """Test AICommitEngine.rewrite_batch with asyncio.gather."""

    @pytest.mark.asyncio
    async def test_batch_empty_list(self):
        engine = AICommitEngine()
        results = await engine.rewrite_batch([])
        assert results == []

    @pytest.mark.asyncio
    async def test_batch_calls_all_commits(self):
        engine = AICommitEngine()
        engine.rewrite_message = AsyncMock(
            side_effect=lambda msg, h, files: RewriteResult(
                original=msg, rewritten=f"rewritten: {msg}", commit_hash=h,
            )
        )
        commits = [
            ("hash1", "msg1", ["a.py"]),
            ("hash2", "msg2", ["b.py"]),
            ("hash3", "msg3", ["c.py"]),
        ]
        results = await engine.rewrite_batch(commits)
        assert len(results) == 3
        assert results[0].rewritten == "rewritten: msg1"
        assert results[2].commit_hash == "hash3"
        assert engine.rewrite_message.call_count == 3

    @pytest.mark.asyncio
    async def test_batch_respects_concurrency_limit(self):
        """Verify semaphore limits concurrent execution."""
        import asyncio

        max_concurrent = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        async def _tracked_rewrite(msg, h, files):
            nonlocal max_concurrent, current_concurrent
            async with lock:
                current_concurrent += 1
                if current_concurrent > max_concurrent:
                    max_concurrent = current_concurrent
            await asyncio.sleep(0.01)
            async with lock:
                current_concurrent -= 1
            return RewriteResult(original=msg, rewritten=msg, commit_hash=h)

        engine = AICommitEngine()
        engine.rewrite_message = _tracked_rewrite
        commits = [(f"h{i}", f"m{i}", []) for i in range(10)]
        results = await engine.rewrite_batch(commits, max_concurrency=3)
        assert len(results) == 10
        assert max_concurrent <= 3


def _make_mock_response(status_code=200, json_data=None, text=""):
    """Create a mock httpx.Response (sync methods like json/raise_for_status)."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.text = text
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=resp
        )
    return resp


class TestOllamaProviderHTTP:
    """Test Ollama provider HTTP flows with mocked httpx."""

    @pytest.mark.asyncio
    async def test_generate_message_success(self):
        provider = OllamaProvider()
        provider.client = AsyncMock()
        provider.client.post.return_value = _make_mock_response(
            json_data={"response": "feat: add new feature"}
        )
        ctx = CommitContext(original_message="add feature", commit_hash="abc", files_changed=["a.py"])
        result = await provider.generate_message(ctx, MessageStyle.CONVENTIONAL)
        assert result == "feat: add new feature"
        provider.client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_message_connect_error_raises(self):
        provider = OllamaProvider(raise_on_error=True)
        provider.client = AsyncMock()
        provider.client.post.side_effect = httpx.ConnectError("Connection refused")
        ctx = CommitContext(original_message="test", commit_hash="abc", files_changed=[])
        with pytest.raises(AIConnectionError, match="Ollama"):
            await provider.generate_message(ctx, MessageStyle.SIMPLE)

    @pytest.mark.asyncio
    async def test_generate_message_connect_error_fallback(self):
        provider = OllamaProvider(raise_on_error=False)
        provider.client = AsyncMock()
        provider.client.post.side_effect = httpx.ConnectError("Connection refused")
        ctx = CommitContext(original_message="original msg", commit_hash="abc", files_changed=[])
        result = await provider.generate_message(ctx, MessageStyle.SIMPLE)
        assert result == "original msg"
        assert provider._last_error is not None

    @pytest.mark.asyncio
    async def test_check_connection_success(self):
        provider = OllamaProvider(model="llama3.2")
        provider.client = AsyncMock()
        provider.client.get.return_value = _make_mock_response(
            json_data={"models": [{"name": "llama3.2:latest"}]}
        )
        connected, status = await provider.check_connection()
        assert connected is True
        assert status == "Connected"

    @pytest.mark.asyncio
    async def test_check_connection_model_not_found(self):
        provider = OllamaProvider(model="nonexistent")
        provider.client = AsyncMock()
        provider.client.get.return_value = _make_mock_response(
            json_data={"models": [{"name": "llama3.2:latest"}]}
        )
        connected, status = await provider.check_connection()
        assert connected is False
        assert "not found" in status


class TestOpenAIProviderHTTP:
    """Test OpenAI provider HTTP flows with mocked httpx."""

    @pytest.mark.asyncio
    async def test_generate_message_success(self):
        provider = OpenAIProvider(api_key="test-key")
        provider.client = AsyncMock()
        provider.client.post.return_value = _make_mock_response(
            json_data={"choices": [{"message": {"content": "fix: resolve null pointer"}}]}
        )
        ctx = CommitContext(original_message="fix bug", commit_hash="abc", files_changed=["main.py"])
        result = await provider.generate_message(ctx, MessageStyle.CONVENTIONAL)
        assert result == "fix: resolve null pointer"

    @pytest.mark.asyncio
    async def test_check_connection_invalid_key(self):
        provider = OpenAIProvider(api_key="bad-key")
        provider.client = AsyncMock()
        provider.client.get.return_value = _make_mock_response(status_code=401)
        connected, status = await provider.check_connection()
        assert connected is False
        assert "Invalid API key" in status

    @pytest.mark.asyncio
    async def test_unexpected_response_raises(self):
        provider = OpenAIProvider(api_key="test-key", raise_on_error=True)
        provider.client = AsyncMock()
        provider.client.post.return_value = _make_mock_response(
            json_data={"unexpected": "format"}
        )
        ctx = CommitContext(original_message="test", commit_hash="abc", files_changed=[])
        with pytest.raises(AIConnectionError, match="OpenAI"):
            await provider.generate_message(ctx, MessageStyle.SIMPLE)


class TestAnthropicProviderHTTP:
    """Test Anthropic provider HTTP flows with mocked httpx."""

    @pytest.mark.asyncio
    async def test_generate_message_success(self):
        provider = AnthropicProvider(api_key="test-key")
        provider.client = AsyncMock()
        provider.client.post.return_value = _make_mock_response(
            json_data={"content": [{"text": "docs: update README"}]}
        )
        ctx = CommitContext(original_message="update readme", commit_hash="abc", files_changed=["README.md"])
        result = await provider.generate_message(ctx, MessageStyle.CONVENTIONAL)
        assert result == "docs: update README"

    @pytest.mark.asyncio
    async def test_check_connection_rate_limited_is_ok(self):
        provider = AnthropicProvider(api_key="test-key")
        provider.client = AsyncMock()
        provider.client.post.return_value = _make_mock_response(status_code=429)
        connected, status = await provider.check_connection()
        assert connected is True
        assert status == "Connected"
