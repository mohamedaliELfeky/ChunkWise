"""
Simple Tokenizer

Basic whitespace-based tokenizer that doesn't require external dependencies.
"""

import re
from typing import List

from chunkwise.tokenizers.base import BaseTokenizer


class SimpleTokenizer(BaseTokenizer):
    """
    Simple whitespace-based tokenizer.

    This tokenizer splits text on whitespace and punctuation.
    It's less accurate than tiktoken but has no dependencies.

    Example:
        >>> tokenizer = SimpleTokenizer()
        >>> tokens = tokenizer.encode("Hello, world!")
        >>> len(tokens)
        4
    """

    # Pattern for tokenization
    TOKEN_PATTERN = re.compile(r'\w+|[^\w\s]')

    def __init__(self):
        """Initialize simple tokenizer."""
        self._vocab = {}
        self._reverse_vocab = {}
        self._next_id = 0

    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.

        Note: This creates vocabulary on-the-fly, so the same text
        will always produce the same IDs within the same instance.

        Args:
            text: Input text

        Returns:
            List of token IDs
        """
        if not text:
            return []

        tokens = self.TOKEN_PATTERN.findall(text)
        ids = []

        for token in tokens:
            if token not in self._vocab:
                self._vocab[token] = self._next_id
                self._reverse_vocab[self._next_id] = token
                self._next_id += 1
            ids.append(self._vocab[token])

        return ids

    def decode(self, tokens: List[int]) -> str:
        """
        Decode token IDs to text.

        Args:
            tokens: List of token IDs

        Returns:
            Decoded text (tokens joined with spaces)
        """
        if not tokens:
            return ""

        words = []
        for token_id in tokens:
            if token_id in self._reverse_vocab:
                words.append(self._reverse_vocab[token_id])
            else:
                words.append(f"[UNK:{token_id}]")

        # Smart joining - no space before punctuation
        result = []
        for i, word in enumerate(words):
            if i > 0 and not re.match(r'[^\w\s]', word):
                result.append(" ")
            result.append(word)

        return "".join(result)

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text to string tokens.

        Args:
            text: Input text

        Returns:
            List of string tokens
        """
        if not text:
            return []
        return self.TOKEN_PATTERN.findall(text)

    def count(self, text: str) -> int:
        """
        Count tokens in text.

        This is more efficient than encode() as it doesn't
        build the vocabulary.

        Args:
            text: Input text

        Returns:
            Number of tokens
        """
        if not text:
            return 0
        return len(self.TOKEN_PATTERN.findall(text))

    def __repr__(self) -> str:
        return f"SimpleTokenizer(vocab_size={len(self._vocab)})"


class WordTokenizer(BaseTokenizer):
    """
    Simple word-based tokenizer.

    Splits text on whitespace only, treating each word as a token.

    Example:
        >>> tokenizer = WordTokenizer()
        >>> tokenizer.count("Hello world")
        2
    """

    def __init__(self):
        """Initialize word tokenizer."""
        self._vocab = {}
        self._reverse_vocab = {}
        self._next_id = 0

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

        words = text.split()
        ids = []

        for word in words:
            if word not in self._vocab:
                self._vocab[word] = self._next_id
                self._reverse_vocab[self._next_id] = word
                self._next_id += 1
            ids.append(self._vocab[word])

        return ids

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

        words = []
        for token_id in tokens:
            if token_id in self._reverse_vocab:
                words.append(self._reverse_vocab[token_id])
            else:
                words.append(f"[UNK:{token_id}]")

        return " ".join(words)

    def count(self, text: str) -> int:
        """
        Count words in text.

        Args:
            text: Input text

        Returns:
            Number of words
        """
        if not text:
            return 0
        return len(text.split())

    def __repr__(self) -> str:
        return f"WordTokenizer(vocab_size={len(self._vocab)})"


class CharacterTokenizer(BaseTokenizer):
    """
    Character-level tokenizer.

    Each character is a token.

    Example:
        >>> tokenizer = CharacterTokenizer()
        >>> tokenizer.count("Hello")
        5
    """

    def encode(self, text: str) -> List[int]:
        """
        Encode text to character codes.

        Args:
            text: Input text

        Returns:
            List of character codes (Unicode code points)
        """
        if not text:
            return []
        return [ord(c) for c in text]

    def decode(self, tokens: List[int]) -> str:
        """
        Decode character codes to text.

        Args:
            tokens: List of character codes

        Returns:
            Decoded text
        """
        if not tokens:
            return ""
        return "".join(chr(t) for t in tokens)

    def count(self, text: str) -> int:
        """
        Count characters in text.

        Args:
            text: Input text

        Returns:
            Number of characters
        """
        return len(text) if text else 0

    def __repr__(self) -> str:
        return "CharacterTokenizer()"
