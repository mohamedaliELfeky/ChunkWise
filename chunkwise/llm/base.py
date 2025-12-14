"""
Base LLM Interface

Abstract base class for LLM providers.
"""

from abc import ABC, abstractmethod
from typing import Optional, List
import asyncio


class BaseLLM(ABC):
    """
    Abstract base class for LLM providers.

    Used for agentic chunking strategies.
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate a response for a prompt.

        Args:
            prompt: Input prompt

        Returns:
            Generated text response
        """
        pass

    @abstractmethod
    async def generate_async(self, prompt: str) -> str:
        """
        Asynchronously generate a response.

        Args:
            prompt: Input prompt

        Returns:
            Generated text response
        """
        pass

    def generate_batch(self, prompts: List[str]) -> List[str]:
        """
        Generate responses for multiple prompts.

        Args:
            prompts: List of prompts

        Returns:
            List of responses
        """
        return [self.generate(prompt) for prompt in prompts]

    async def generate_batch_async(self, prompts: List[str]) -> List[str]:
        """
        Asynchronously generate responses for multiple prompts.

        Args:
            prompts: List of prompts

        Returns:
            List of responses
        """
        tasks = [self.generate_async(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks)


def get_llm_provider(
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    **kwargs,
) -> BaseLLM:
    """
    Factory function to get an LLM provider.

    Args:
        provider: Provider name ("openai", "anthropic", "ollama")
        model: Model name
        api_key: API key
        **kwargs: Additional arguments

    Returns:
        LLM provider instance
    """
    if provider == "openai":
        from chunkwise.llm.openai_llm import OpenAILLM

        return OpenAILLM(model_name=model, api_key=api_key, **kwargs)
    elif provider == "anthropic":
        from chunkwise.llm.anthropic_llm import AnthropicLLM

        return AnthropicLLM(model_name=model, api_key=api_key, **kwargs)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
