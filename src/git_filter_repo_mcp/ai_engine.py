"""AI-powered commit message engine using Ollama, OpenAI, or Anthropic."""

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Protocol

import httpx

logger = logging.getLogger(__name__)


class MessageStyle(str, Enum):
    """Commit message styles."""

    CONVENTIONAL = "conventional"  # feat: xxx, fix: xxx
    GITMOJI = "gitmoji"  # :sparkles: xxx
    SIMPLE = "simple"  # Short descriptive message
    DETAILED = "detailed"  # With body and footer


@dataclass
class CommitContext:
    """Context for AI commit message generation."""

    original_message: str
    commit_hash: str
    files_changed: list[str]
    diff_summary: str | None = None
    author: str | None = None


@dataclass
class RewriteResult:
    """Result of AI message rewrite."""

    original: str
    rewritten: str
    commit_hash: str
    reasoning: str | None = None


class AIProvider(Protocol):
    """Protocol for AI providers."""

    async def generate_message(self, context: CommitContext, style: MessageStyle) -> str: ...


# Style instructions for prompt building
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
    """Build the prompt for commit message generation."""
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


class OllamaProvider:
    """Ollama-based AI provider for local inference."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2",
    ):
        self.base_url = base_url
        self.model = model
        self.client = httpx.AsyncClient(timeout=60.0)

    async def generate_message(self, context: CommitContext, style: MessageStyle) -> str:
        """Generate a commit message using Ollama."""
        prompt = build_prompt(context, style)

        try:
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.9,
                    },
                },
            )
            response.raise_for_status()
            result = response.json()
            return self._parse_response(result.get("response", ""), style)
        except httpx.HTTPError as e:
            logger.warning(f"Ollama request failed: {e}")
            return context.original_message

    def _parse_response(self, response: str, style: MessageStyle) -> str:
        """Parse and clean the AI response."""
        # Remove quotes if present
        message = response.strip().strip("\"'")

        # Ensure proper format for conventional commits
        if style == MessageStyle.CONVENTIONAL:
            valid_prefixes = [
                "feat:",
                "fix:",
                "docs:",
                "style:",
                "refactor:",
                "test:",
                "chore:",
                "perf:",
                "ci:",
                "build:",
                "revert:",
            ]
            has_prefix = any(message.lower().startswith(p) for p in valid_prefixes)
            if not has_prefix:
                # Try to infer the type
                message = f"chore: {message}"

        return message

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


class OpenAIProvider:
    """OpenAI-based AI provider."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model = model
        self.client = httpx.AsyncClient(
            timeout=30.0,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    async def generate_message(self, context: CommitContext, style: MessageStyle) -> str:
        """Generate a commit message using OpenAI."""
        prompt = build_prompt(context, style)

        try:
            response = await self.client.post(
                "https://api.openai.com/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a git commit message writer. Respond only with the commit message.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 200,
                },
            )
            response.raise_for_status()
            result = response.json()
            message = result["choices"][0]["message"]["content"]
            return message.strip().strip("\"'")
        except httpx.HTTPError as e:
            logger.warning(f"OpenAI request failed: {e}")
            return context.original_message

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


class AnthropicProvider:
    """Anthropic Claude-based AI provider."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key
        self.model = model
        self.client = httpx.AsyncClient(
            timeout=30.0,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2024-01-01",
                "content-type": "application/json",
            },
        )

    async def generate_message(self, context: CommitContext, style: MessageStyle) -> str:
        """Generate a commit message using Anthropic Claude."""
        prompt = build_prompt(context, style)

        try:
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
            message = result["content"][0]["text"]
            return message.strip().strip("\"'")
        except httpx.HTTPError as e:
            logger.warning(f"Anthropic request failed: {e}")
            return context.original_message

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


class AICommitEngine:
    """Main engine for AI-powered commit message rewriting."""

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
        """Rewrite a single commit message."""
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
        commits: list[tuple[str, str, list[str]]],  # (hash, message, files)
    ) -> list[RewriteResult]:
        """Rewrite multiple commit messages."""
        results = []
        for commit_hash, message, files in commits:
            result = await self.rewrite_message(message, commit_hash, files)
            results.append(result)
        return results

    def create_callback(self) -> Callable[[str, str], str]:
        """Create a synchronous callback for use with git-filter-repo."""
        cache: dict[str, str] = {}
        # Create a single event loop for all callbacks
        loop = asyncio.new_event_loop()

        def callback(message: str, commit_hash: str) -> str:
            if commit_hash in cache:
                return cache[commit_hash]

            try:
                result = loop.run_until_complete(self.rewrite_message(message, commit_hash))
                cache[commit_hash] = result.rewritten
                return result.rewritten
            except Exception as e:
                logger.error(f"Failed to rewrite commit {commit_hash}: {e}")
                return message

        # Store loop reference for cleanup
        callback._loop = loop  # type: ignore
        return callback

    async def close(self):
        """Close the provider."""
        if hasattr(self.provider, "close"):
            await self.provider.close()


# Utility functions for quick usage
async def rewrite_with_ollama(
    message: str,
    commit_hash: str = "",
    model: str = "llama3.2",
    style: MessageStyle = MessageStyle.CONVENTIONAL,
) -> str:
    """Quick helper to rewrite a message with Ollama."""
    provider = OllamaProvider(model=model)
    engine = AICommitEngine(provider, style)
    try:
        result = await engine.rewrite_message(message, commit_hash)
        return result.rewritten
    finally:
        await engine.close()


def get_provider(
    provider_type: str = "ollama",
    **kwargs,
) -> AIProvider:
    """Factory function to get an AI provider."""
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
