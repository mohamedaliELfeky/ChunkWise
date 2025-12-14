"""
Tiktoken Tokenizer

OpenAI's tiktoken-based tokenizer for accurate token counting.
"""

from typing import List, Optional

from chunkwise.tokenizers.base import BaseTokenizer


class TiktokenTokenizer(BaseTokenizer):
    """
    Tokenizer using OpenAI's tiktoken library.

    Supports various encoding models:
    - cl100k_base: GPT-4, GPT-3.5-turbo, text-embedding-ada-002
    - p50k_base: Codex models
    - r50k_base: GPT-3 models
    - o200k_base: GPT-4o models

    Example:
        >>> tokenizer = TiktokenTokenizer()
        >>> tokens = tokenizer.encode("Hello world")
        >>> len(tokens)
        2
        >>> tokenizer.decode(tokens)
        'Hello world'
    """

    # Model to encoding mapping
    MODEL_TO_ENCODING = {
        # GPT-4o and GPT-4o-mini
        "gpt-4o": "o200k_base",
        "gpt-4o-mini": "o200k_base",
        # GPT-4
        "gpt-4": "cl100k_base",
        "gpt-4-turbo": "cl100k_base",
        "gpt-4-32k": "cl100k_base",
        # GPT-3.5
        "gpt-3.5-turbo": "cl100k_base",
        "gpt-3.5-turbo-16k": "cl100k_base",
        # Embeddings
        "text-embedding-ada-002": "cl100k_base",
        "text-embedding-3-small": "cl100k_base",
        "text-embedding-3-large": "cl100k_base",
    }

    def __init__(self, model: str = "cl100k_base"):
        """
        Initialize tiktoken tokenizer.

        Args:
            model: Encoding model name or OpenAI model name
        """
        self.model = model
        self._encoder = None

    @property
    def encoder(self):
        """Lazy load the encoder."""
        if self._encoder is None:
            import tiktoken

            # Check if model is an OpenAI model name
            if self.model in self.MODEL_TO_ENCODING:
                encoding_name = self.MODEL_TO_ENCODING[self.model]
            else:
                encoding_name = self.model

            try:
                self._encoder = tiktoken.get_encoding(encoding_name)
            except Exception:
                # Fallback to cl100k_base
                self._encoder = tiktoken.get_encoding("cl100k_base")

        return self._encoder

    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text

        Returns:
            List of token IDs
        """
        if not text:
            return []
        return self.encoder.encode(text)

    def decode(self, tokens: List[int]) -> str:
        """
        Decode token IDs to text.

        Args:
            tokens: List of token IDs

        Returns:
            Decoded text
        """
        if not tokens:
            return ""
        return self.encoder.decode(tokens)

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """
        Encode multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List of token ID lists
        """
        return [self.encode(text) for text in texts]

    def count_batch(self, texts: List[str]) -> List[int]:
        """
        Count tokens for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List of token counts
        """
        return [self.count(text) for text in texts]

    @classmethod
    def for_model(cls, model_name: str) -> "TiktokenTokenizer":
        """
        Create tokenizer for a specific OpenAI model.

        Args:
            model_name: OpenAI model name (e.g., "gpt-4", "gpt-3.5-turbo")

        Returns:
            TiktokenTokenizer instance
        """
        return cls(model=model_name)

    def get_special_tokens(self) -> dict:
        """
        Get special tokens for this encoding.

        Returns:
            Dictionary of special tokens
        """
        return dict(self.encoder._special_tokens)

    def __repr__(self) -> str:
        return f"TiktokenTokenizer(model='{self.model}')"
