"""
Sliding Window Chunking Strategy

Chunker using a sliding window approach with configurable overlap.
"""

from typing import List, Optional

from chunkwise.base import BaseChunker
from chunkwise.chunk import Chunk
from chunkwise.config import ChunkConfig
from chunkwise.tokenizers.base import get_tokenizer
from chunkwise.utils.overlap import sliding_window_positions


class SlidingWindowChunker(BaseChunker):
    """
    Split text using a sliding window approach.

    Creates overlapping chunks by moving a fixed-size window across the text.
    Unlike other chunkers where overlap is the amount of repeated content,
    here the step size determines how much the window moves.

    The overlap is: window_size - step_size

    Example:
        >>> # 500 char window with 100 char overlap
        >>> chunker = SlidingWindowChunker(chunk_size=500, chunk_overlap=100)
        >>> chunks = chunker.chunk(text)
        >>> # Window moves 400 chars at a time (500 - 100)

        >>> # Token-based sliding window
        >>> chunker = SlidingWindowChunker(
        ...     chunk_size=512,
        ...     chunk_overlap=50,
        ...     use_tokens=True
        ... )
    """

    def __init__(
        self,
        config: Optional[ChunkConfig] = None,
        use_tokens: bool = False,
        **kwargs,
    ):
        """
        Initialize sliding window chunker.

        Args:
            config: ChunkConfig instance
            use_tokens: If True, window size is in tokens, not characters
            **kwargs: Override config parameters
        """
        super().__init__(config, **kwargs)
        self.use_tokens = use_tokens
        self._tokenizer = None

    @property
    def tokenizer(self):
        """Lazy load tokenizer."""
        if self._tokenizer is None and self.use_tokens:
            self._tokenizer = get_tokenizer(
                name=self.config.tokenizer,
                model=self.config.tokenizer_model,
            )
        return self._tokenizer

    def _chunk_text(self, text: str) -> List[Chunk]:
        """
        Split text using sliding window.

        Args:
            text: Input text

        Returns:
            List of Chunk objects
        """
        if not text:
            return []

        if self.use_tokens:
            return self._chunk_by_tokens(text)
        else:
            return self._chunk_by_chars(text)

    def _chunk_by_chars(self, text: str) -> List[Chunk]:
        """
        Character-based sliding window.

        Args:
            text: Input text

        Returns:
            List of Chunk objects
        """
        window_size = self.config.chunk_size
        step_size = window_size - self.config.chunk_overlap

        if step_size <= 0:
            step_size = window_size // 2

        positions = sliding_window_positions(len(text), window_size, step_size)

        chunks = []
        for index, (start, end) in enumerate(positions):
            chunk_text = text[start:end]

            # Try to adjust to word boundaries
            if end < len(text) and not text[end - 1].isspace():
                # Find next space
                adjusted_end = text.find(" ", end)
                if adjusted_end > 0 and adjusted_end - end < 50:
                    chunk_text = text[start:adjusted_end]
                    end = adjusted_end

            chunks.append(
                Chunk(
                    content=chunk_text,
                    index=index,
                    start_char=start,
                    end_char=end,
                    metadata={
                        "strategy": "sliding_window",
                        "window_size": window_size,
                        "step_size": step_size,
                    },
                )
            )

        return chunks

    def _chunk_by_tokens(self, text: str) -> List[Chunk]:
        """
        Token-based sliding window.

        Args:
            text: Input text

        Returns:
            List of Chunk objects
        """
        tokens = self.tokenizer.encode(text)

        window_size = self.config.chunk_size
        step_size = window_size - self.config.chunk_overlap

        if step_size <= 0:
            step_size = window_size // 2

        chunks = []
        index = 0

        for i in range(0, len(tokens), step_size):
            window_tokens = tokens[i : i + window_size]
            chunk_text = self.tokenizer.decode(window_tokens)

            # Estimate character positions
            # This is approximate since token boundaries don't align perfectly
            char_ratio = len(text) / max(len(tokens), 1)
            start_char = int(i * char_ratio)
            end_char = int(min(i + len(window_tokens), len(tokens)) * char_ratio)

            chunks.append(
                Chunk(
                    content=chunk_text,
                    index=index,
                    start_char=start_char,
                    end_char=end_char,
                    metadata={
                        "strategy": "sliding_window",
                        "window_size": window_size,
                        "step_size": step_size,
                        "token_count": len(window_tokens),
                    },
                )
            )

            index += 1

            if i + window_size >= len(tokens):
                break

        return chunks


class OverlapChunker(BaseChunker):
    """
    Simple chunker that creates chunks with specified overlap percentage.

    Example:
        >>> chunker = OverlapChunker(chunk_size=1000, overlap_percent=0.2)
        >>> # Creates 1000 char chunks with 200 char (20%) overlap
    """

    def __init__(
        self,
        config: Optional[ChunkConfig] = None,
        overlap_percent: float = 0.1,
        **kwargs,
    ):
        """
        Initialize overlap chunker.

        Args:
            config: ChunkConfig instance
            overlap_percent: Overlap as percentage of chunk size (0.0 to 0.5)
            **kwargs: Override config parameters
        """
        super().__init__(config, **kwargs)
        self.overlap_percent = min(max(overlap_percent, 0.0), 0.5)

        # Override chunk_overlap based on percentage
        self.config.chunk_overlap = int(self.config.chunk_size * self.overlap_percent)

    def _chunk_text(self, text: str) -> List[Chunk]:
        """
        Split text with percentage-based overlap.

        Args:
            text: Input text

        Returns:
            List of Chunk objects
        """
        # Use sliding window internally
        slider = SlidingWindowChunker(config=self.config)
        return slider._chunk_text(text)
