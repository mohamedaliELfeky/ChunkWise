"""
Fixed-Size Chunking Strategies

Chunkers that split text into fixed-size pieces based on:
- Characters
- Tokens
- Words
"""

from typing import List, Optional

from chunkwise.base import BaseChunker
from chunkwise.chunk import Chunk
from chunkwise.config import ChunkConfig
from chunkwise.tokenizers.base import get_tokenizer


class CharacterChunker(BaseChunker):
    """
    Split text into fixed-size character chunks.

    This is the simplest chunking strategy that splits text
    based on character count.

    Example:
        >>> chunker = CharacterChunker(chunk_size=100, chunk_overlap=20)
        >>> chunks = chunker.chunk("Your long text here...")
    """

    def _chunk_text(self, text: str) -> List[Chunk]:
        """
        Split text into character-based chunks.

        Args:
            text: Input text

        Returns:
            List of Chunk objects
        """
        if not text:
            return []

        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap
        step = chunk_size - overlap

        chunks = []
        start = 0
        index = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))

            # Try to find a good breakpoint (word boundary)
            if end < len(text):
                # Look for space within last 10% of chunk
                search_start = max(start, end - chunk_size // 10)
                last_space = text.rfind(" ", search_start, end)
                if last_space > start:
                    end = last_space + 1

            chunk_text = text[start:end]

            chunks.append(
                Chunk(
                    content=chunk_text,
                    index=index,
                    start_char=start,
                    end_char=end,
                    metadata={"strategy": "character"},
                )
            )

            start += step
            index += 1

            # Avoid infinite loop
            if step <= 0:
                break

        return chunks


class TokenChunker(BaseChunker):
    """
    Split text into fixed-size token chunks.

    Uses tiktoken or other tokenizers for accurate token counting.
    This is the recommended strategy for LLM applications.

    Example:
        >>> chunker = TokenChunker(chunk_size=512, chunk_overlap=50)
        >>> chunks = chunker.chunk("Your long text here...")

        >>> # Use specific model's tokenizer
        >>> chunker = TokenChunker(chunk_size=512, tokenizer_model="gpt-4")
        >>> chunks = chunker.chunk(text)
    """

    def __init__(self, config: Optional[ChunkConfig] = None, **kwargs):
        """
        Initialize token chunker.

        Args:
            config: ChunkConfig instance
            **kwargs: Override config parameters
        """
        super().__init__(config, **kwargs)
        self._tokenizer = None

    @property
    def tokenizer(self):
        """Lazy load tokenizer."""
        if self._tokenizer is None:
            self._tokenizer = get_tokenizer(
                name=self.config.tokenizer,
                model=self.config.tokenizer_model,
            )
        return self._tokenizer

    def _chunk_text(self, text: str) -> List[Chunk]:
        """
        Split text into token-based chunks.

        Args:
            text: Input text

        Returns:
            List of Chunk objects
        """
        if not text:
            return []

        # Encode text to tokens
        tokens = self.tokenizer.encode(text)

        if len(tokens) <= self.config.chunk_size:
            return [
                Chunk(
                    content=text,
                    index=0,
                    start_char=0,
                    end_char=len(text),
                    metadata={"strategy": "token", "token_count": len(tokens)},
                )
            ]

        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap
        step = chunk_size - overlap

        chunks = []
        index = 0
        char_position = 0

        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i : i + chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens)

            # Calculate character positions
            start_char = text.find(chunk_text.strip()[:20], char_position)
            if start_char < 0:
                start_char = char_position
            end_char = start_char + len(chunk_text)

            chunks.append(
                Chunk(
                    content=chunk_text,
                    index=index,
                    start_char=start_char,
                    end_char=end_char,
                    metadata={
                        "strategy": "token",
                        "token_count": len(chunk_tokens),
                    },
                )
            )

            char_position = max(char_position, end_char - self.config.chunk_overlap * 4)
            index += 1

            if i + chunk_size >= len(tokens):
                break

        return chunks

    def _count_tokens(self, text: str) -> int:
        """Override to use our tokenizer."""
        return self.tokenizer.count(text)


class WordChunker(BaseChunker):
    """
    Split text into fixed-size word chunks.

    Splits on word boundaries, keeping a fixed number of words per chunk.

    Example:
        >>> chunker = WordChunker(chunk_size=100, chunk_overlap=10)  # 100 words per chunk
        >>> chunks = chunker.chunk("Your long text here...")
    """

    def _chunk_text(self, text: str) -> List[Chunk]:
        """
        Split text into word-based chunks.

        Args:
            text: Input text

        Returns:
            List of Chunk objects
        """
        if not text:
            return []

        words = text.split()

        if len(words) <= self.config.chunk_size:
            return [
                Chunk(
                    content=text,
                    index=0,
                    start_char=0,
                    end_char=len(text),
                    metadata={"strategy": "word", "word_count": len(words)},
                )
            ]

        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap
        step = max(1, chunk_size - overlap)

        chunks = []
        index = 0

        for i in range(0, len(words), step):
            chunk_words = words[i : i + chunk_size]
            chunk_text = " ".join(chunk_words)

            # Calculate character positions
            start_char = self._find_word_position(text, words, i)
            end_char = start_char + len(chunk_text)

            chunks.append(
                Chunk(
                    content=chunk_text,
                    index=index,
                    start_char=start_char,
                    end_char=end_char,
                    metadata={
                        "strategy": "word",
                        "word_count": len(chunk_words),
                    },
                )
            )

            index += 1

            if i + chunk_size >= len(words):
                break

        return chunks

    def _find_word_position(self, text: str, words: List[str], word_index: int) -> int:
        """
        Find character position of a word in text.

        Args:
            text: Full text
            words: List of words
            word_index: Index of word to find

        Returns:
            Character position
        """
        if word_index == 0:
            return 0

        # Approximate position based on average word length
        avg_word_len = len(text) / max(len(words), 1)
        approx_pos = int(word_index * avg_word_len)

        # Search for exact position
        word = words[word_index]
        pos = text.find(word, max(0, approx_pos - 100))
        if pos >= 0:
            return pos

        return approx_pos
