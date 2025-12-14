"""
OpenAI LLM Provider

LLM provider using OpenAI's API.
"""

import os
from typing import Optional

from chunkwise.llm.base import BaseLLM
from chunkwise.exceptions import LLMError


class OpenAILLM(BaseLLM):
    """
    LLM provider using OpenAI's API.

    Supports GPT-4, GPT-3.5, and other OpenAI models.

    Example:
        >>> llm = OpenAILLM()
        >>> response = llm.generate("Hello, how are you?")

        >>> # With specific model
        >>> llm = OpenAILLM(model_name="gpt-4-turbo")
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ):
        """
        Initialize OpenAI LLM provider.

        Args:
            model_name: OpenAI model name
            api_key: API key (or use OPENAI_API_KEY env var)
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
        """
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None
        self._async_client = None

    @property
    def client(self):
        """Lazy load synchronous client."""
        if self._client is None:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise LLMError("openai not installed. Install with: pip install openai")
        return self._client

    @property
    def async_client(self):
        """Lazy load async client."""
        if self._async_client is None:
            try:
                from openai import AsyncOpenAI

                self._async_client = AsyncOpenAI(api_key=self.api_key)
            except ImportError:
                raise LLMError("openai not installed. Install with: pip install openai")
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
            raise LLMError("OpenAI API key not provided")

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            raise LLMError(f"OpenAI generation failed: {e}")

    async def generate_async(self, prompt: str) -> str:
        """
        Asynchronously generate a response.

        Args:
            prompt: Input prompt

        Returns:
            Generated response
        """
        if not self.api_key:
            raise LLMError("OpenAI API key not provided")

        try:
            response = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            raise LLMError(f"OpenAI async generation failed: {e}")

    def __repr__(self) -> str:
        return f"OpenAILLM(model='{self.model_name}')"
