"""
Anthropic LLM Provider

LLM provider using Anthropic's Claude API.
"""

import os
from typing import Optional

from chunkwise.llm.base import BaseLLM
from chunkwise.exceptions import LLMError


class AnthropicLLM(BaseLLM):
    """
    LLM provider using Anthropic's Claude API.

    Supports Claude 3 models: Opus, Sonnet, Haiku.

    Example:
        >>> llm = AnthropicLLM()
        >>> response = llm.generate("Hello, how are you?")

        >>> # With specific model
        >>> llm = AnthropicLLM(model_name="claude-3-opus-20240229")
    """

    def __init__(
        self,
        model_name: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ):
        """
        Initialize Anthropic LLM provider.

        Args:
            model_name: Anthropic model name
            api_key: API key (or use ANTHROPIC_API_KEY env var)
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
        """
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None
        self._async_client = None

    @property
    def client(self):
        """Lazy load synchronous client."""
        if self._client is None:
            try:
                import anthropic

                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise LLMError(
                    "anthropic not installed. Install with: pip install anthropic"
                )
        return self._client

    @property
    def async_client(self):
        """Lazy load async client."""
        if self._async_client is None:
            try:
                import anthropic

                self._async_client = anthropic.AsyncAnthropic(api_key=self.api_key)
            except ImportError:
                raise LLMError(
                    "anthropic not installed. Install with: pip install anthropic"
                )
        return self._async_client

    def generate(self, prompt: str) -> str:
        """
        Generate a response.

        Args:
            prompt: Input prompt

        Returns:
            Generated response
        """
        if not self.api_key:
            raise LLMError("Anthropic API key not provided")

        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception as e:
            raise LLMError(f"Anthropic generation failed: {e}")

    async def generate_async(self, prompt: str) -> str:
        """
        Asynchronously generate a response.

        Args:
            prompt: Input prompt

        Returns:
            Generated response
        """
        if not self.api_key:
            raise LLMError("Anthropic API key not provided")

        try:
            response = await self.async_client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception as e:
            raise LLMError(f"Anthropic async generation failed: {e}")

    def __repr__(self) -> str:
        return f"AnthropicLLM(model='{self.model_name}')"
