"""Unit tests for the AI engine.

Layout:
- ``TestPromptBuilding``: pure-function tests for ``build_prompt``.
- ``TestParseResponse``: provider-agnostic response normalisation.
- ``TestProviderInit``: every provider's __init__ knobs in one matrix.
- ``TestGetProviderFactory``: the ``get_provider`` switch and its errors.
- ``TestProviderHTTP``: HTTP request/response paths, with httpx mocked.
- ``TestProviderClose``: provider/engine teardown lifecycle.
- ``TestRewriteBatch``: AICommitEngine.rewrite_batch concurrency.
- ``TestDataclasses``: trivial dataclass/enum invariants.
"""

import asyncio
from typing import Callable
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from git_filter_repo_mcp.ai_engine import (
    AICommitEngine,
    AIConnectionError,
    AnthropicProvider,
    BaseProvider,
    CommitContext,
    MessageStyle,
    OllamaProvider,
    OpenAICompatibleProvider,
    OpenAIProvider,
    RewriteResult,
    build_prompt,
    get_provider,
)


# --- Test helpers --------------------------------------------------------


def _make_response(status_code: int = 200, json_data: dict | None = None) -> MagicMock:
    """Mocked httpx.Response with sync .json() and .raise_for_status()."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=resp,
        )
    return resp


def _ctx(message: str = "test", files: list[str] | None = None, **kwargs) -> CommitContext:
    """Shortcut for a CommitContext with sensible defaults."""
    return CommitContext(
        original_message=message,
        commit_hash=kwargs.pop("commit_hash", "abc123"),
        files_changed=files if files is not None else [],
        **kwargs,
    )


def _ollama_factory() -> OllamaProvider:
    return OllamaProvider()


def _openai_factory() -> OpenAIProvider:
    return OpenAIProvider(api_key="test")


def _openai_compatible_factory() -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider()


def _anthropic_factory() -> AnthropicProvider:
    return AnthropicProvider(api_key="test")


# Cycle through every concrete provider exactly once via parametrize.
# Each entry: (provider_id_for_pytest, factory_callable).
ALL_PROVIDERS: list[tuple[str, Callable[[], BaseProvider]]] = [
    ("ollama", _ollama_factory),
    ("openai", _openai_factory),
    ("openai-compatible", _openai_compatible_factory),
    ("anthropic", _anthropic_factory),
]


# --- Pure functions ------------------------------------------------------


class TestPromptBuilding:
    """``build_prompt`` is a pure function — exercise the surface it owns."""

    def test_original_message_included(self) -> None:
        prompt = build_prompt(_ctx("fix bug"), MessageStyle.CONVENTIONAL)
        assert "fix bug" in prompt

    def test_files_included_when_present(self) -> None:
        prompt = build_prompt(_ctx(files=["main.py", "utils.py"]), MessageStyle.SIMPLE)
        assert "main.py" in prompt and "utils.py" in prompt

    def test_files_section_omitted_when_empty(self) -> None:
        prompt = build_prompt(_ctx(), MessageStyle.DETAILED)
        assert "Files changed" not in prompt

    def test_more_than_ten_files_truncated(self) -> None:
        prompt = build_prompt(
            _ctx(files=[f"file{i}.py" for i in range(20)]), MessageStyle.SIMPLE,
        )
        assert "+10 more" in prompt

    def test_long_diff_summary_capped(self) -> None:
        prompt = build_prompt(_ctx(diff_summary="x" * 1000), MessageStyle.SIMPLE)
        # 500-char cap + boilerplate stays well below 1500.
        assert len(prompt) < 1500

    @pytest.mark.parametrize(
        "style,marker",
        [
            (MessageStyle.CONVENTIONAL, "feat:"),
            (MessageStyle.GITMOJI, ":sparkles:"),
        ],
    )
    def test_style_specific_instruction_present(self, style: MessageStyle, marker: str) -> None:
        prompt = build_prompt(_ctx(), style)
        assert marker in prompt


class TestParseResponse:
    """``_parse_response`` lives on ``BaseProvider`` — test it via Ollama."""

    @pytest.fixture
    def provider(self) -> OllamaProvider:
        return OllamaProvider()

    @pytest.mark.parametrize(
        "raw,style,expected",
        [
            ("update config", MessageStyle.CONVENTIONAL, "chore: update config"),
            ("feat: add new feature", MessageStyle.CONVENTIONAL, "feat: add new feature"),
            ('"some message"', MessageStyle.SIMPLE, "some message"),
            ("FEAT: add feature", MessageStyle.CONVENTIONAL, "FEAT: add feature"),
        ],
    )
    def test_normalisation(
        self, provider: OllamaProvider, raw: str, style: MessageStyle, expected: str,
    ) -> None:
        assert provider._parse_response(raw, style) == expected

    @pytest.mark.parametrize(
        "prefix",
        ["feat:", "fix:", "docs:", "style:", "refactor:", "test:",
         "chore:", "perf:", "ci:", "build:", "revert:"],
    )
    def test_recognised_conventional_prefixes_not_re_prefixed(
        self, provider: OllamaProvider, prefix: str,
    ) -> None:
        out = provider._parse_response(f"{prefix} something", MessageStyle.CONVENTIONAL)
        assert not out.startswith(f"chore: {prefix}")

    @pytest.mark.parametrize("empty", ["", "   ", '""', "''"])
    def test_empty_response_falls_back_to_original(
        self, provider: OllamaProvider, empty: str,
    ) -> None:
        """Empty AI output must NOT produce ``"chore: "`` — fall back to the
        commit's original message instead."""
        out = provider._parse_response(
            empty, MessageStyle.CONVENTIONAL, fallback="original msg",
        )
        assert out == "original msg"


# --- Provider init -------------------------------------------------------


class TestProviderInit:
    """``__init__`` knobs across the three providers, in one matrix."""

    def test_ollama_defaults(self) -> None:
        p = OllamaProvider()
        assert p.base_url == "http://localhost:11434"
        assert p.model == "llama3.2"

    def test_ollama_overrides(self) -> None:
        p = OllamaProvider(base_url="http://custom:1234", model="custom-model")
        assert (p.base_url, p.model) == ("http://custom:1234", "custom-model")

    def test_openai_defaults(self) -> None:
        p = OpenAIProvider(api_key="test-key")
        assert p.api_key == "test-key"
        assert p.model == "gpt-4o-mini"
        assert p.base_url == "https://api.openai.com/v1"

    @pytest.mark.parametrize(
        "kwargs,attr,expected",
        [
            ({"model": "gpt-4"}, "model", "gpt-4"),
            ({"base_url": "https://custom.api.com/v1"}, "base_url", "https://custom.api.com/v1"),
        ],
    )
    def test_openai_overrides(self, kwargs: dict, attr: str, expected: str) -> None:
        p = OpenAIProvider(api_key="test-key", **kwargs)
        assert getattr(p, attr) == expected

    def test_openai_compatible_defaults(self) -> None:
        p = OpenAICompatibleProvider()
        assert p.api_key is None
        assert p.model == "local-model"
        assert p.base_url == "http://localhost:1234/v1"
        assert p.provider_name == "OpenAI-compatible"

    def test_openai_compatible_optional_auth_header(self) -> None:
        without_key = OpenAICompatibleProvider()
        assert "authorization" not in {key.lower() for key in without_key.client.headers}

        with_key = OpenAICompatibleProvider(api_key="local-key")
        assert with_key.client.headers["authorization"] == "Bearer local-key"

    def test_anthropic_defaults(self) -> None:
        p = AnthropicProvider(api_key="test-key")
        assert p.api_key == "test-key"
        assert p.model == "claude-sonnet-4-20250514"

    def test_anthropic_model_override(self) -> None:
        assert AnthropicProvider(
            api_key="x", model="claude-3-opus-20240229",
        ).model == "claude-3-opus-20240229"

    def test_anthropic_headers_set(self) -> None:
        p = AnthropicProvider(api_key="test-key")
        assert p.client.headers["x-api-key"] == "test-key"
        assert p.client.headers["anthropic-version"] == "2023-06-01"

    @pytest.mark.parametrize("_, factory", ALL_PROVIDERS, ids=[p for p, _ in ALL_PROVIDERS])
    def test_raise_on_error_default_true(self, _: str, factory: Callable) -> None:
        assert factory().raise_on_error is True

    @pytest.mark.parametrize(
        "name",
        ["ollama", "openai", "anthropic"],
    )
    def test_raise_on_error_can_be_disabled(self, name: str) -> None:
        if name == "ollama":
            p = OllamaProvider(raise_on_error=False)
        elif name == "openai":
            p = OpenAIProvider(api_key="x", raise_on_error=False)
        else:
            p = AnthropicProvider(api_key="x", raise_on_error=False)
        assert p.raise_on_error is False

    @pytest.mark.parametrize("_, factory", ALL_PROVIDERS, ids=[p for p, _ in ALL_PROVIDERS])
    def test_last_error_initially_none(self, _: str, factory: Callable) -> None:
        assert factory()._last_error is None


# --- get_provider factory ------------------------------------------------


class TestGetProviderFactory:
    """``get_provider`` selects the right concrete class and forwards kwargs."""

    def test_ollama_default(self) -> None:
        assert isinstance(get_provider("ollama"), OllamaProvider)

    def test_ollama_forwards_kwargs(self) -> None:
        p = get_provider("ollama", base_url="http://custom:1234", model="custom-model")
        assert isinstance(p, OllamaProvider)
        assert (p.base_url, p.model) == ("http://custom:1234", "custom-model")

    def test_openai_with_key(self) -> None:
        assert isinstance(get_provider("openai", api_key="x"), OpenAIProvider)

    def test_openai_custom_base_url(self) -> None:
        p = get_provider("openai", api_key="x", base_url="https://custom.com/v1")
        assert isinstance(p, OpenAIProvider) and p.base_url == "https://custom.com/v1"

    def test_openai_compatible_without_key(self) -> None:
        p = get_provider(
            "openai-compatible",
            base_url="http://localhost:9999/v1",
            model="qwen-local",
        )
        assert isinstance(p, OpenAICompatibleProvider)
        assert p.api_key is None
        assert (p.base_url, p.model) == ("http://localhost:9999/v1", "qwen-local")

    @pytest.mark.parametrize(
        "provider_type,label",
        [
            ("lmstudio", "LM Studio"),
            ("vllm", "vLLM"),
            ("llamacpp", "llama.cpp"),
            ("localai", "LocalAI"),
        ],
    )
    def test_openai_compatible_aliases(self, provider_type: str, label: str) -> None:
        p = get_provider(provider_type, base_url="http://local/v1", model="local-model")
        assert isinstance(p, OpenAICompatibleProvider)
        assert p.provider_name == label

    def test_openrouter_requires_key_and_uses_compatible_provider(self) -> None:
        p = get_provider(
            "openrouter",
            api_key="router-key",
            base_url="https://openrouter.ai/api/v1",
            model="openai/gpt-4o-mini",
        )
        assert isinstance(p, OpenAICompatibleProvider)
        assert p.provider_name == "OpenRouter"

    def test_anthropic_with_key(self) -> None:
        assert isinstance(get_provider("anthropic", api_key="x"), AnthropicProvider)

    def test_anthropic_custom_model(self) -> None:
        p = get_provider("anthropic", api_key="x", model="claude-3-opus-20240229")
        assert isinstance(p, AnthropicProvider) and p.model == "claude-3-opus-20240229"

    @pytest.mark.parametrize(
        "provider_type,error_fragment",
        [
            ("openai", "OpenAI API key required"),
            ("anthropic", "Anthropic API key required"),
            ("openrouter", "OpenRouter API key required"),
            ("unknown", "Unknown provider"),
        ],
    )
    def test_factory_errors(self, provider_type: str, error_fragment: str) -> None:
        with pytest.raises(ValueError, match=error_fragment):
            get_provider(provider_type)


# --- Provider HTTP flows (mocked httpx) ---------------------------------


class TestProviderHTTP:
    """End-to-end ``generate_message`` and ``check_connection`` paths,
    asserting the request shape and response interpretation."""

    @pytest.mark.asyncio
    async def test_ollama_generate_success(self) -> None:
        provider = OllamaProvider()
        provider.client = AsyncMock()
        provider.client.post.return_value = _make_response(
            json_data={"response": "feat: add new feature"},
        )
        result = await provider.generate_message(
            _ctx("add feature", files=["a.py"]), MessageStyle.CONVENTIONAL,
        )
        assert result == "feat: add new feature"
        provider.client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_openai_generate_success(self) -> None:
        provider = OpenAIProvider(api_key="x")
        provider.client = AsyncMock()
        provider.client.post.return_value = _make_response(
            json_data={"choices": [{"message": {"content": "fix: resolve null pointer"}}]},
        )
        result = await provider.generate_message(
            _ctx("fix bug", files=["main.py"]), MessageStyle.CONVENTIONAL,
        )
        assert result == "fix: resolve null pointer"

    @pytest.mark.asyncio
    async def test_openai_compatible_generate_success(self) -> None:
        provider = OpenAICompatibleProvider(base_url="http://localhost:1234/v1")
        provider.client = AsyncMock()
        provider.client.post.return_value = _make_response(
            json_data={"choices": [{"message": {"content": "chore: clean commits"}}]},
        )
        result = await provider.generate_message(
            _ctx("cleanup", files=["main.py"]), MessageStyle.CONVENTIONAL,
        )
        assert result == "chore: clean commits"
        provider.client.post.assert_called_once()
        url = provider.client.post.call_args.args[0]
        assert url == "http://localhost:1234/v1/chat/completions"

    @pytest.mark.asyncio
    async def test_anthropic_generate_success(self) -> None:
        provider = AnthropicProvider(api_key="x")
        provider.client = AsyncMock()
        provider.client.post.return_value = _make_response(
            json_data={"content": [{"text": "docs: update README"}]},
        )
        result = await provider.generate_message(
            _ctx("update readme", files=["README.md"]), MessageStyle.CONVENTIONAL,
        )
        assert result == "docs: update README"

    @pytest.mark.asyncio
    async def test_ollama_check_connection_success(self) -> None:
        provider = OllamaProvider(model="llama3.2")
        provider.client = AsyncMock()
        provider.client.get.return_value = _make_response(
            json_data={"models": [{"name": "llama3.2:latest"}]},
        )
        connected, status = await provider.check_connection()
        assert connected is True
        assert status == "Connected"

    @pytest.mark.asyncio
    async def test_ollama_check_connection_missing_model(self) -> None:
        provider = OllamaProvider(model="nonexistent")
        provider.client = AsyncMock()
        provider.client.get.return_value = _make_response(
            json_data={"models": [{"name": "llama3.2:latest"}]},
        )
        connected, status = await provider.check_connection()
        assert connected is False
        assert "not found" in status

    @pytest.mark.asyncio
    async def test_ollama_check_connection_malformed_response(self) -> None:
        provider = OllamaProvider(model="llama3.2")
        provider.client = AsyncMock()
        response = _make_response()
        response.json.side_effect = ValueError("bad json")
        provider.client.get.return_value = response
        connected, status = await provider.check_connection()
        assert connected is False
        assert "Malformed response" in status

    @pytest.mark.asyncio
    async def test_openai_check_connection_unauthorized(self) -> None:
        provider = OpenAIProvider(api_key="bad-key")
        provider.client = AsyncMock()
        provider.client.get.return_value = _make_response(status_code=401)
        connected, status = await provider.check_connection()
        assert connected is False
        assert "Invalid API key" in status

    @pytest.mark.asyncio
    async def test_openai_compatible_check_connection_model_endpoint_optional(self) -> None:
        provider = OpenAICompatibleProvider()
        provider.client = AsyncMock()
        provider.client.get.return_value = _make_response(status_code=404)
        connected, status = await provider.check_connection()
        assert connected is True
        assert "unavailable" in status

    @pytest.mark.asyncio
    async def test_anthropic_check_connection_rate_limit_is_ok(self) -> None:
        """429 should still count as 'reachable'."""
        provider = AnthropicProvider(api_key="x")
        provider.client = AsyncMock()
        provider.client.post.return_value = _make_response(status_code=429)
        connected, status = await provider.check_connection()
        assert connected is True
        assert status == "Connected"

    @pytest.mark.asyncio
    async def test_openai_unexpected_response_raises(self) -> None:
        provider = OpenAIProvider(api_key="x", raise_on_error=True)
        provider.client = AsyncMock()
        provider.client.post.return_value = _make_response(json_data={"unexpected": "format"})
        with pytest.raises(AIConnectionError, match="OpenAI"):
            await provider.generate_message(_ctx(), MessageStyle.SIMPLE)

    # Error-handling matrix: ConnectError + HTTPStatusError × raise_on_error on/off.
    # Parametrised because the BaseProvider.generate_message branch under test is
    # the same regardless of which subclass is in front of it.
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "exception",
        [
            httpx.ConnectError("Connection refused"),
            httpx.HTTPStatusError("500", request=MagicMock(), response=MagicMock(status_code=500)),
        ],
        ids=["connect_error", "http_status_error"],
    )
    async def test_error_raises_when_raise_on_error(self, exception: Exception) -> None:
        provider = OllamaProvider(raise_on_error=True)
        provider.client = AsyncMock()
        provider.client.post.side_effect = exception
        with pytest.raises(AIConnectionError, match="Ollama"):
            await provider.generate_message(_ctx(), MessageStyle.SIMPLE)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "exception",
        [
            httpx.ConnectError("Connection refused"),
            httpx.HTTPStatusError("500", request=MagicMock(), response=MagicMock(status_code=500)),
        ],
        ids=["connect_error", "http_status_error"],
    )
    async def test_error_falls_back_when_silent(self, exception: Exception) -> None:
        provider = OllamaProvider(raise_on_error=False)
        provider.client = AsyncMock()
        provider.client.post.side_effect = exception
        result = await provider.generate_message(_ctx("original msg"), MessageStyle.SIMPLE)
        assert result == "original msg"
        assert provider._last_error is not None


# --- Lifecycle -----------------------------------------------------------


class TestProviderClose:
    """``close()`` must await the underlying httpx client's aclose."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("_, factory", ALL_PROVIDERS, ids=[p for p, _ in ALL_PROVIDERS])
    async def test_provider_close(self, _: str, factory: Callable) -> None:
        provider = factory()
        provider.client = AsyncMock()
        await provider.close()
        provider.client.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_engine_close_delegates_to_provider(self) -> None:
        provider = OllamaProvider()
        provider.client = AsyncMock()
        await AICommitEngine(provider=provider).close()
        provider.client.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_provider_close_is_idempotent(self) -> None:
        """Calling ``close()`` twice must not raise or hit the network twice."""
        provider = OllamaProvider()
        provider.client = AsyncMock()
        await provider.close()
        await provider.close()  # must not raise
        provider.client.aclose.assert_called_once()


# --- Engine batch --------------------------------------------------------


class TestRewriteBatch:
    """``AICommitEngine.rewrite_batch`` distributes work via asyncio.gather."""

    @pytest.mark.asyncio
    async def test_empty_input_returns_empty(self) -> None:
        assert await AICommitEngine().rewrite_batch([]) == []

    @pytest.mark.asyncio
    async def test_each_commit_processed_once(self) -> None:
        engine = AICommitEngine()
        engine.rewrite_message = AsyncMock(
            side_effect=lambda msg, h, files: RewriteResult(
                original=msg, rewritten=f"rewritten: {msg}", commit_hash=h,
            ),
        )
        commits = [("hash1", "msg1", ["a.py"]),
                   ("hash2", "msg2", ["b.py"]),
                   ("hash3", "msg3", ["c.py"])]
        results = await engine.rewrite_batch(commits)
        assert [r.rewritten for r in results] == [
            "rewritten: msg1", "rewritten: msg2", "rewritten: msg3",
        ]
        assert engine.rewrite_message.call_count == 3

    @pytest.mark.asyncio
    async def test_failure_in_one_task_does_not_drop_others(self) -> None:
        """Regression for gather-cancels-siblings default: one task raising
        must NOT lose results from the others. The failing commit keeps its
        original message (no-op rewrite)."""
        async def flaky_rewrite(message: str, commit_hash: str, files: list[str]) -> RewriteResult:
            if commit_hash == "bad":
                raise RuntimeError("transient AI error")
            return RewriteResult(
                original=message, rewritten=f"new: {message}", commit_hash=commit_hash,
            )

        engine = AICommitEngine()
        engine.rewrite_message = flaky_rewrite  # type: ignore[assignment]
        results = await engine.rewrite_batch([
            ("ok1", "msg1", []),
            ("bad", "msg2", []),
            ("ok2", "msg3", []),
        ])
        assert len(results) == 3
        # ok1 and ok2 succeed normally; bad keeps its original message.
        rewrites = {r.commit_hash: r.rewritten for r in results}
        assert rewrites["ok1"] == "new: msg1"
        assert rewrites["ok2"] == "new: msg3"
        assert rewrites["bad"] == "msg2"  # original preserved

    @pytest.mark.asyncio
    async def test_cancelled_task_propagates(self) -> None:
        async def cancelled_rewrite(message: str, commit_hash: str, files: list[str]) -> RewriteResult:
            if commit_hash == "cancel":
                raise asyncio.CancelledError()
            return RewriteResult(original=message, rewritten=message, commit_hash=commit_hash)

        engine = AICommitEngine()
        engine.rewrite_message = cancelled_rewrite  # type: ignore[assignment]

        with pytest.raises(asyncio.CancelledError):
            await engine.rewrite_batch([("cancel", "msg", [])])

    @pytest.mark.asyncio
    async def test_concurrency_limit_respected(self) -> None:
        """Asserts max concurrent in-flight tasks never exceeds the semaphore."""
        max_concurrent = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        async def _tracked_rewrite(msg, h, files):
            nonlocal max_concurrent, current_concurrent
            async with lock:
                current_concurrent += 1
                max_concurrent = max(max_concurrent, current_concurrent)
            await asyncio.sleep(0.01)
            async with lock:
                current_concurrent -= 1
            return RewriteResult(original=msg, rewritten=msg, commit_hash=h)

        engine = AICommitEngine()
        engine.rewrite_message = _tracked_rewrite
        results = await engine.rewrite_batch(
            [(f"h{i}", f"m{i}", []) for i in range(10)], max_concurrency=3,
        )
        assert len(results) == 10
        assert max_concurrent <= 3


# --- Dataclasses / enums -------------------------------------------------


class TestDataclasses:
    """Trivial invariants on the small data types."""

    def test_message_style_values(self) -> None:
        assert {s.value for s in MessageStyle} == {
            "conventional", "gitmoji", "simple", "detailed",
        }

    def test_commit_context_minimal(self) -> None:
        ctx = CommitContext(original_message="t", commit_hash="abc", files_changed=[])
        assert ctx.diff_summary is None and ctx.author is None

    def test_commit_context_full(self) -> None:
        ctx = CommitContext(
            original_message="t", commit_hash="abc",
            files_changed=["a.py", "b.py"],
            diff_summary="Added 10 lines", author="Test User",
        )
        assert ctx.files_changed == ["a.py", "b.py"]
        assert ctx.diff_summary == "Added 10 lines"
        assert ctx.author == "Test User"

    def test_ai_connection_error_basic(self) -> None:
        err = AIConnectionError("Ollama", "Connection refused")
        assert "Ollama" in str(err) and "Connection refused" in str(err)
        assert err.provider == "Ollama"

    def test_ai_connection_error_chained(self) -> None:
        original = ConnectionError("Network unreachable")
        err = AIConnectionError("OpenAI", "API error", original)
        assert err.original_error is original and err.provider == "OpenAI"

    def test_engine_defaults_to_ollama(self) -> None:
        engine = AICommitEngine()
        assert isinstance(engine.provider, OllamaProvider)
        assert engine.style == MessageStyle.CONVENTIONAL

    def test_engine_accepts_custom_provider_and_style(self) -> None:
        provider = OpenAIProvider(api_key="x")
        engine = AICommitEngine(provider=provider, style=MessageStyle.GITMOJI)
        assert engine.provider is provider
        assert engine.style == MessageStyle.GITMOJI
