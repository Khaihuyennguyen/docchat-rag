"""The language model behind a tiny interface, so tests can be a fake."""

import os
from typing import Protocol

class MissingAPIKey(RuntimeError):
    """Raised when the API key is missing."""

class LLM(Protocol):
    def complete(self, system: str, prompt: str) -> str: ...

class ClaudeLLM:
    def __init__(self, model: str, max_tokens: int = 1024) -> None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            # Fail early with a message a person can act on, not a deep SDK error
            raise MissingAPIKey("ANTHROPIC_API_KEY is not set. Ingest and eval work without it; asking needs it.")
        import anthropic # import here so tests never need an API key

        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, system: str, prompt: str) -> str:
        response = self.client.messages.create(
            model = self.model,
            max_tokens=self.max_tokens,
            # cache_control marks the system prompt as cacheable. Caching only
            # kicks in once the cached prefix is long (about 1024+ tokens).
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text")

class FakeLLM:
    """Return a fixed reply and remembers every call. For testing only"""
    def __init__(self, reply: str = "Fake answer [1]") -> None:
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, prompt: str) -> str:
        self.calls.append((system, prompt))
        return self.reply
    