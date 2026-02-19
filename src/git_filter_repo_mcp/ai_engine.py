"""AI-powered commit message engine using Ollama, OpenAI, or Anthropic."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import httpx

logger = logging.getLogger(__name__)


class AIConnectionError(Exception):
    """AI connection failed."""

    def __init__(self, provider: str, message: str, original_error: Exception | None = None):
        self.provider = provider
        self.original_error = original_error
        super().__init__(f"{provider}: {message}")


class MessageStyle(str, Enum):
    """Commit message style."""

    CONVENTIONAL = "conventional"
    GITMOJI = "gitmoji"
    SIMPLE = "simple"
    DETAILED = "detailed"


@dataclass
class CommitContext:
    """Commit context."""

    original_message: str
    commit_hash: str
    files_changed: list[str]
    diff_summary: str | None = None
    author: str | None = None


@dataclass
class RewriteResult:
    """Rewrite result."""

    original: str
    rewritten: str
    commit_hash: str
    reasoning: str | None = None


class AIProvider(Protocol):
    """AI provider interface."""

    async def generate_message(self, context: CommitContext, style: MessageStyle) -> str: ...


STYLE_INSTRUCTIONS = {
    MessageStyle.CONVENTIONAL: """
Use conventional commit format:
- feat: for new features
- fix: for bug fixes
- docs: for documentation
- style: for formatting
- refactor: for code refactoring
- test: for tests
- chore: for maintenance

Example: "feat: add user authentication"
""",
    MessageStyle.GITMOJI: """
Use gitmoji format with emoji at the start:
- :sparkles: for new features
- :bug: for bug fixes
- :memo: for documentation
- :art: for formatting
- :recycle: for refactoring
- :white_check_mark: for tests

Example: ":sparkles: add user authentication"
""",
    MessageStyle.SIMPLE: """
Write a short, clear commit message (max 50 chars).
Use imperative mood (e.g., "Add" not "Added").

Example: "Add user authentication"
""",
    MessageStyle.DETAILED: """
Write a detailed commit message with:
1. Subject line (max 50 chars, imperative mood)
2. Blank line
3. Body explaining what and why

Example:
"Add user authentication

Implement JWT-based authentication for API endpoints.
This allows secure access control for user resources."
""",
}


def build_prompt(context: CommitContext, style: MessageStyle) -> str:
    """Build prompt."""
    files_info = ""
    if context.files_changed:
        files_info = f"\nFiles changed: {', '.join(context.files_changed[:10])}"
        if len(context.files_changed) > 10:
            files_info += f" (+{len(context.files_changed) - 10} more)"

    diff_info = ""
    if context.diff_summary:
        diff_info = f"\nDiff summary:\n{context.diff_summary[:500]}"

    return f"""You are a git commit message writer. Rewrite the following commit message to be clearer and more descriptive.

{STYLE_INSTRUCTIONS[style]}

Original commit message: "{context.original_message}"
{files_info}
{diff_info}

Respond with ONLY the new commit message, nothing else. Do not include quotes around the message."""


class BaseProvider(ABC):
    """Base class for AI providers with shared error handling."""

    provider_name: str = "base"

    def __init__(self, model: str, raise_on_error: bool = True):
        self.model = model
        self.raise_on_error = raise_on_error
        self._last_error: str | None = None

    @abstractmethod
    def _create_client(self) -> httpx.AsyncClient: ...

    @abstractmethod
    async def check_connection(self) -> tuple[bool, str]: ...

    @abstractmethod
    async def _call_api(self, prompt: str) -> str:
        """Call the provider API and return the raw message text."""
        ...

    async def generate_message(self, context: CommitContext, style: MessageStyle) -> str:
        """Generate message with shared error handling."""
        prompt = build_prompt(context, style)
        self._last_error = None

        try:
            raw = await self._call_api(prompt)
            return self._parse_response(raw, style)
        except httpx.ConnectError as e:
            self._last_error = f"Cannot connect to {self.provider_name}"
            logger.warning(f"{self.provider_name.lower()} connect: {e}")
            if self.raise_on_error:
                raise AIConnectionError(self.provider_name, self._last_error, e)
            return context.original_message
        except httpx.HTTPError as e:
            self._last_error = str(e)
            logger.warning(f"{self.provider_name.lower()}: {e}")
            if self.raise_on_error:
                raise AIConnectionError(self.provider_name, self._last_error, e)
            return context.original_message

    def _parse_response(self, response: str, style: MessageStyle) -> str:
        """Parse and normalize response text."""
        message = response.strip().strip("\"'")

        if style == MessageStyle.CONVENTIONAL:
            valid_prefixes = [
                "feat:", "fix:", "docs:", "style:", "refactor:",
                "test:", "chore:", "perf:", "ci:", "build:", "revert:",
            ]
            if not any(message.lower().startswith(p) for p in valid_prefixes):
                message = f"chore: {message}"

        return message

    async def close(self):
        await self.client.aclose()


class OllamaProvider(BaseProvider):
    """Ollama provider."""

    provider_name = "Ollama"

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2",
        raise_on_error: bool = True,
    ):
        super().__init__(model, raise_on_error)
        self.base_url = base_url
        self.client = self._create_client()

    def _create_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=60.0)

    async def check_connection(self) -> tuple[bool, str]:
        try:
            response = await self.client.get(f"{self.base_url}/api/tags", timeout=5.0)
            response.raise_for_status()
            models = response.json().get("models", [])
            model_names = [m.get("name", "").split(":")[0] for m in models]
            if self.model.split(":")[0] not in model_names:
                return False, f"Model '{self.model}' not found. Available: {model_names}"
            return True, "Connected"
        except httpx.ConnectError:
            return False, f"Cannot connect to Ollama at {self.base_url}"
        except httpx.HTTPError as e:
            return False, f"Ollama: {e}"

    async def _call_api(self, prompt: str) -> str:
        response = await self.client.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "top_p": 0.9},
            },
        )
        response.raise_for_status()
        return response.json().get("response", "")


class OpenAIProvider(BaseProvider):
    """OpenAI provider."""

    provider_name = "OpenAI"

    def __init__(self, api_key: str, model: str = "gpt-4o-mini", raise_on_error: bool = True):
        super().__init__(model, raise_on_error)
        self.api_key = api_key
        self.client = self._create_client()

    def _create_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=30.0,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )

    async def check_connection(self) -> tuple[bool, str]:
        try:
            response = await self.client.get("https://api.openai.com/v1/models", timeout=5.0)
            if response.status_code == 401:
                return False, "Invalid API key"
            response.raise_for_status()
            return True, "Connected"
        except httpx.ConnectError:
            return False, "Cannot connect to OpenAI"
        except httpx.HTTPError as e:
            return False, f"OpenAI: {e}"

    async def _call_api(self, prompt: str) -> str:
        response = await self.client.post(
            "https://api.openai.com/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a git commit message writer. Respond only with the commit message."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.3,
                "max_tokens": 200,
            },
        )
        response.raise_for_status()
        result = response.json()
        try:
            return result["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            logger.warning(f"openai unexpected response: {result}")
            if self.raise_on_error:
                raise AIConnectionError("OpenAI", "Unexpected response format")
            return ""


class AnthropicProvider(BaseProvider):
    """Anthropic provider."""

    provider_name = "Anthropic"

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514", raise_on_error: bool = True):
        super().__init__(model, raise_on_error)
        self.api_key = api_key
        self.client = self._create_client()

    def _create_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=30.0,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
        )

    async def check_connection(self) -> tuple[bool, str]:
        try:
            response = await self.client.post(
                "https://api.anthropic.com/v1/messages",
                json={
                    "model": self.model,
                    "max_tokens": 1,
                    "messages": [{"role": "user", "content": "hi"}],
                },
                timeout=5.0,
            )
            if response.status_code == 401:
                return False, "Invalid API key"
            if response.status_code in (200, 429):
                return True, "Connected"
            response.raise_for_status()
            return True, "Connected"
        except httpx.ConnectError:
            return False, "Cannot connect to Anthropic"
        except httpx.HTTPError as e:
            return False, f"Anthropic: {e}"

    async def _call_api(self, prompt: str) -> str:
        response = await self.client.post(
            "https://api.anthropic.com/v1/messages",
            json={
                "model": self.model,
                "max_tokens": 200,
                "messages": [{"role": "user", "content": prompt}],
                "system": "You are a git commit message writer. Respond only with the commit message, nothing else.",
            },
        )
        response.raise_for_status()
        result = response.json()
        try:
            return result["content"][0]["text"] or ""
        except (KeyError, IndexError, TypeError):
            logger.warning(f"anthropic unexpected response: {result}")
            if self.raise_on_error:
                raise AIConnectionError("Anthropic", "Unexpected response format")
            return ""


class AICommitEngine:
    """AI commit message engine."""

    def __init__(
        self,
        provider: AIProvider | None = None,
        style: MessageStyle = MessageStyle.CONVENTIONAL,
    ):
        self.provider = provider or OllamaProvider()
        self.style = style

    async def rewrite_message(
        self,
        original_message: str,
        commit_hash: str,
        files_changed: list[str] | None = None,
        diff_summary: str | None = None,
    ) -> RewriteResult:
        """Rewrite single message."""
        context = CommitContext(
            original_message=original_message,
            commit_hash=commit_hash,
            files_changed=files_changed or [],
            diff_summary=diff_summary,
        )

        new_message = await self.provider.generate_message(context, self.style)

        return RewriteResult(
            original=original_message,
            rewritten=new_message,
            commit_hash=commit_hash,
        )

    async def rewrite_batch(
        self,
        commits: list[tuple[str, str, list[str]]],
    ) -> list[RewriteResult]:
        """Batch rewrite."""
        results = []
        for commit_hash, message, files in commits:
            result = await self.rewrite_message(message, commit_hash, files)
            results.append(result)
        return results

    async def close(self):
        if hasattr(self.provider, "close"):
            await self.provider.close()


def get_provider(
    provider_type: str = "ollama",
    **kwargs,
) -> AIProvider:
    """Provider factory."""
    if provider_type == "ollama":
        return OllamaProvider(
            base_url=kwargs.get("base_url", "http://localhost:11434"),
            model=kwargs.get("model", "llama3.2"),
        )
    elif provider_type == "openai":
        api_key = kwargs.get("api_key")
        if not api_key:
            raise ValueError("OpenAI API key required")
        return OpenAIProvider(
            api_key=api_key,
            model=kwargs.get("model", "gpt-4o-mini"),
        )
    elif provider_type == "anthropic":
        api_key = kwargs.get("api_key")
        if not api_key:
            raise ValueError("Anthropic API key required")
        return AnthropicProvider(
            api_key=api_key,
            model=kwargs.get("model", "claude-sonnet-4-20250514"),
        )
    else:
        raise ValueError(f"Unknown provider: {provider_type}")
