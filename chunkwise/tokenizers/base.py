"""
Base Tokenizer Interface

Abstract base class for all tokenizers.
"""

from abc import ABC, abstractmethod
from typing import List


class BaseTokenizer(ABC):
    """
    Abstract base class for tokenizers.

    All tokenizers must implement encode and decode methods.
    """

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text

        Returns:
            List of token IDs
        """
        pass

    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        """
        Decode token IDs to text.

        Args:
            tokens: List of token IDs

        Returns:
            Decoded text
        """
        pass

    def count(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Input text

        Returns:
            Number of tokens
        """
        return len(self.encode(text))

    def truncate(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to max_tokens.

        Args:
            text: Input text
            max_tokens: Maximum number of tokens

        Returns:
            Truncated text
        """
        tokens = self.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self.decode(tokens[:max_tokens])

    def split_by_tokens(self, text: str, chunk_size: int) -> List[str]:
        """
        Split text into chunks of approximately chunk_size tokens.

        Args:
            text: Input text
            chunk_size: Target number of tokens per chunk

        Returns:
            List of text chunks
        """
        tokens = self.encode(text)
        chunks = []

        for i in range(0, len(tokens), chunk_size):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = self.decode(chunk_tokens)
            chunks.append(chunk_text)

        return chunks


def get_tokenizer(name: str = "tiktoken", model: str = "cl100k_base") -> BaseTokenizer:
    """
    Factory function to get a tokenizer by name.

    Args:
        name: Tokenizer name ("tiktoken", "simple")
        model: Model name for the tokenizer

    Returns:
        Tokenizer instance
    """
    if name == "tiktoken":
        from chunkwise.tokenizers.tiktoken_tokenizer import TiktokenTokenizer

        return TiktokenTokenizer(model=model)
    elif name == "simple":
        from chunkwise.tokenizers.simple import SimpleTokenizer

        return SimpleTokenizer()
    else:
        raise ValueError(f"Unknown tokenizer: {name}")
