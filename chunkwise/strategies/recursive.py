"""
Recursive Chunking Strategy

LangChain-style recursive text splitting with hierarchical separators.
"""

from typing import List, Optional, Callable

from chunkwise.base import BaseChunker
from chunkwise.chunk import Chunk
from chunkwise.config import ChunkConfig, Language
from chunkwise.language.detector import detect_language


class RecursiveChunker(BaseChunker):
    """
    Recursively split text using a hierarchy of separators.

    This is the most popular chunking strategy, similar to LangChain's
    RecursiveCharacterTextSplitter. It tries to keep semantically related
    text together by splitting on larger units first, then falling back
    to smaller units if chunks are still too large.

    Separator hierarchy (default):
    1. Double newline (paragraphs)
    2. Single newline (lines)
    3. Period + space (sentences)
    4. Comma + space (clauses)
    5. Space (words)
    6. Empty string (characters)

    Example:
        >>> chunker = RecursiveChunker(chunk_size=512, chunk_overlap=50)
        >>> chunks = chunker.chunk(long_document)

        >>> # Custom separators for code
        >>> chunker = RecursiveChunker(
        ...     chunk_size=1000,
        ...     separators=["\\n\\nclass ", "\\n\\ndef ", "\\n\\n", "\\n", " "]
        ... )
    """

    def __init__(
        self,
        config: Optional[ChunkConfig] = None,
        length_function: Optional[Callable[[str], int]] = None,
        **kwargs,
    ):
        """
        Initialize recursive chunker.

        Args:
            config: ChunkConfig instance
            length_function: Custom function to measure text length
            **kwargs: Override config parameters
        """
        super().__init__(config, **kwargs)
        self.length_function = length_function or len

    def _chunk_text(self, text: str) -> List[Chunk]:
        """
        Recursively split text.

        Args:
            text: Input text

        Returns:
            List of Chunk objects
        """
        if not text:
            return []

        # Detect language and get appropriate separators
        if self.config.language == Language.AUTO:
            detected = detect_language(text)
            if detected == "ar":
                separators = self.config._arabic_separators()
            else:
                separators = self.config._english_separators()
        else:
            separators = self.config.separators or self.config._multilingual_separators()

        # Perform recursive splitting
        splits = self._recursive_split(text, separators)

        # Create chunks with overlap
        return self._create_chunks_with_overlap(splits, text)

    def _recursive_split(
        self,
        text: str,
        separators: List[str],
        depth: int = 0,
    ) -> List[str]:
        """
        Recursively split text using separator hierarchy.

        Args:
            text: Text to split
            separators: List of separators to try
            depth: Current recursion depth

        Returns:
            List of text splits
        """
        if not text:
            return []

        if not separators:
            # No more separators, split by characters
            return self._split_by_size(text)

        # If text is small enough, return as-is
        if self.length_function(text) <= self.config.chunk_size:
            return [text]

        separator = separators[0]
        remaining_separators = separators[1:]

        # Try to split with current separator
        if separator:
            splits = text.split(separator)
        else:
            # Empty separator = character split
            splits = list(text)

        # Filter empty splits
        splits = [s for s in splits if s.strip()]

        if len(splits) <= 1:
            # Current separator didn't help, try next
            return self._recursive_split(text, remaining_separators, depth + 1)

        # Process each split
        result = []
        for split in splits:
            if separator and split != splits[-1]:
                # Add separator back (except for last split)
                split = split + separator

            if self.length_function(split) <= self.config.chunk_size:
                result.append(split)
            else:
                # Recursively split with remaining separators
                sub_splits = self._recursive_split(split, remaining_separators, depth + 1)
                result.extend(sub_splits)

        return result

    def _split_by_size(self, text: str) -> List[str]:
        """
        Split text by size when no separators work.

        Args:
            text: Text to split

        Returns:
            List of chunks
        """
        chunks = []
        start = 0
        size = self.config.chunk_size

        while start < len(text):
            end = min(start + size, len(text))

            # Try to find word boundary
            if end < len(text):
                space_pos = text.rfind(" ", start, end)
                if space_pos > start:
                    end = space_pos + 1

            chunks.append(text[start:end])
            start = end

        return chunks

    def _create_chunks_with_overlap(
        self, splits: List[str], original_text: str
    ) -> List[Chunk]:
        """
        Create chunks from splits with overlap handling.

        Args:
            splits: List of text splits
            original_text: Original text

        Returns:
            List of Chunk objects
        """
        if not splits:
            return []

        chunks = []
        current_content = []
        current_length = 0
        index = 0
        char_position = 0

        for split in splits:
            split_length = self.length_function(split)

            # Check if adding this split exceeds chunk size
            if current_length + split_length > self.config.chunk_size and current_content:
                # Create chunk from current content
                chunk_text = "".join(current_content)

                start_char = self._find_position(original_text, chunk_text, char_position)
                end_char = start_char + len(chunk_text)

                chunks.append(
                    Chunk(
                        content=chunk_text.strip(),
                        index=index,
                        start_char=start_char,
                        end_char=end_char,
                        metadata={"strategy": "recursive"},
                    )
                )

                char_position = end_char
                index += 1

                # Handle overlap
                if self.config.chunk_overlap > 0:
                    overlap_content = self._get_overlap_content(
                        current_content, self.config.chunk_overlap
                    )
                    current_content = overlap_content
                    current_length = sum(self.length_function(c) for c in current_content)
                else:
                    current_content = []
                    current_length = 0

            current_content.append(split)
            current_length += split_length

        # Last chunk
        if current_content:
            chunk_text = "".join(current_content)

            start_char = self._find_position(original_text, chunk_text, char_position)
            end_char = start_char + len(chunk_text)

            chunks.append(
                Chunk(
                    content=chunk_text.strip(),
                    index=index,
                    start_char=start_char,
                    end_char=end_char,
                    metadata={"strategy": "recursive"},
                )
            )

        return chunks

    def _get_overlap_content(
        self, content: List[str], target_overlap: int
    ) -> List[str]:
        """
        Get content from end for overlap.

        Args:
            content: Current content list
            target_overlap: Target overlap size

        Returns:
            Content for overlap
        """
        overlap = []
        current_length = 0

        for item in reversed(content):
            item_length = self.length_function(item)
            if current_length + item_length <= target_overlap:
                overlap.insert(0, item)
                current_length += item_length
            else:
                break

        return overlap

    def _find_position(self, text: str, search: str, start: int = 0) -> int:
        """Find position of text in original."""
        search_start = search[:50].strip() if len(search) > 50 else search.strip()
        pos = text.find(search_start, max(0, start - 100))
        return pos if pos >= 0 else start
