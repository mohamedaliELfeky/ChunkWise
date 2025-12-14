"""
Paragraph-Based Chunking Strategy

Chunker that splits text on paragraph boundaries.
"""

import re
from typing import List, Optional

from chunkwise.base import BaseChunker
from chunkwise.chunk import Chunk
from chunkwise.config import ChunkConfig


class ParagraphChunker(BaseChunker):
    """
    Split text into chunks based on paragraph boundaries.

    Paragraphs are detected by double newlines or other configurable markers.
    Respects natural document structure.

    Example:
        >>> chunker = ParagraphChunker(chunk_size=1000)
        >>> text = '''First paragraph.

        Second paragraph.

        Third paragraph.'''
        >>> chunks = chunker.chunk(text)
    """

    # Patterns for paragraph detection
    PARAGRAPH_PATTERNS = [
        r"\n\s*\n",  # Double newline with optional whitespace
        r"\r\n\s*\r\n",  # Windows-style
        r"\n{2,}",  # Multiple newlines
    ]

    def __init__(
        self,
        config: Optional[ChunkConfig] = None,
        min_paragraph_length: int = 50,
        **kwargs,
    ):
        """
        Initialize paragraph chunker.

        Args:
            config: ChunkConfig instance
            min_paragraph_length: Minimum length for a standalone paragraph
            **kwargs: Override config parameters
        """
        super().__init__(config, **kwargs)
        self.min_paragraph_length = min_paragraph_length
        self._paragraph_pattern = re.compile("|".join(self.PARAGRAPH_PATTERNS))

    def _chunk_text(self, text: str) -> List[Chunk]:
        """
        Split text into paragraph-based chunks.

        Args:
            text: Input text

        Returns:
            List of Chunk objects
        """
        if not text:
            return []

        # Split into paragraphs
        paragraphs = self._split_paragraphs(text)

        if not paragraphs:
            return [
                Chunk(
                    content=text,
                    index=0,
                    start_char=0,
                    end_char=len(text),
                    metadata={"strategy": "paragraph"},
                )
            ]

        # Merge small paragraphs and split large ones
        return self._process_paragraphs(paragraphs, text)

    def _split_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs.

        Args:
            text: Input text

        Returns:
            List of paragraphs
        """
        # Split by paragraph markers
        paragraphs = self._paragraph_pattern.split(text)

        # Clean up
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        return paragraphs

    def _process_paragraphs(self, paragraphs: List[str], text: str) -> List[Chunk]:
        """
        Process paragraphs into chunks.

        Args:
            paragraphs: List of paragraph strings
            text: Original text

        Returns:
            List of Chunk objects
        """
        chunks = []
        current_paragraphs = []
        current_length = 0
        index = 0
        char_position = 0

        for paragraph in paragraphs:
            para_length = len(paragraph)

            # If single paragraph exceeds max size, split it
            if para_length > self.config.chunk_size:
                # Save current if any
                if current_paragraphs:
                    chunk = self._create_chunk(
                        current_paragraphs, text, index, char_position
                    )
                    chunks.append(chunk)
                    char_position = chunk.end_char
                    index += 1
                    current_paragraphs = []
                    current_length = 0

                # Split large paragraph
                sub_chunks = self._split_large_paragraph(paragraph, text, index, char_position)
                chunks.extend(sub_chunks)
                index += len(sub_chunks)
                if sub_chunks:
                    char_position = sub_chunks[-1].end_char
                continue

            # Check if adding this paragraph exceeds limit
            if current_length + para_length > self.config.chunk_size and current_paragraphs:
                chunk = self._create_chunk(
                    current_paragraphs, text, index, char_position
                )
                chunks.append(chunk)
                char_position = chunk.end_char
                index += 1

                # Handle overlap
                if self.config.chunk_overlap > 0:
                    overlap_paragraphs = self._get_overlap_paragraphs(
                        current_paragraphs, self.config.chunk_overlap
                    )
                    current_paragraphs = overlap_paragraphs
                    current_length = sum(len(p) for p in current_paragraphs)
                else:
                    current_paragraphs = []
                    current_length = 0

            current_paragraphs.append(paragraph)
            current_length += para_length + 2  # +2 for paragraph separator

        # Last chunk
        if current_paragraphs:
            chunk = self._create_chunk(current_paragraphs, text, index, char_position)
            chunks.append(chunk)

        return chunks

    def _create_chunk(
        self,
        paragraphs: List[str],
        text: str,
        index: int,
        start_search: int,
    ) -> Chunk:
        """
        Create a chunk from paragraphs.

        Args:
            paragraphs: List of paragraphs
            text: Original text
            index: Chunk index
            start_search: Position to start searching

        Returns:
            Chunk object
        """
        chunk_text = "\n\n".join(paragraphs)

        # Find position
        start_char = text.find(paragraphs[0][:30], start_search)
        if start_char < 0:
            start_char = start_search
        end_char = start_char + len(chunk_text)

        return Chunk(
            content=chunk_text,
            index=index,
            start_char=start_char,
            end_char=end_char,
            metadata={
                "strategy": "paragraph",
                "paragraph_count": len(paragraphs),
            },
        )

    def _split_large_paragraph(
        self,
        paragraph: str,
        text: str,
        start_index: int,
        start_char: int,
    ) -> List[Chunk]:
        """
        Split a large paragraph that exceeds chunk size.

        Args:
            paragraph: Large paragraph text
            text: Original text
            start_index: Starting chunk index
            start_char: Starting character position

        Returns:
            List of Chunk objects
        """
        # Use sentence splitting for large paragraphs
        from chunkwise.language.english.sentence_splitter import split_english_sentences

        sentences = split_english_sentences(paragraph, min_length=0)

        if not sentences:
            # Fallback to character chunking
            return [
                Chunk(
                    content=paragraph,
                    index=start_index,
                    start_char=start_char,
                    end_char=start_char + len(paragraph),
                    metadata={"strategy": "paragraph", "oversized": True},
                )
            ]

        chunks = []
        current = []
        current_len = 0
        char_pos = start_char
        idx = start_index

        for sentence in sentences:
            if current_len + len(sentence) > self.config.chunk_size and current:
                chunk_text = " ".join(current)
                chunks.append(
                    Chunk(
                        content=chunk_text,
                        index=idx,
                        start_char=char_pos,
                        end_char=char_pos + len(chunk_text),
                        metadata={"strategy": "paragraph", "from_split": True},
                    )
                )
                char_pos += len(chunk_text) + 1
                idx += 1
                current = []
                current_len = 0

            current.append(sentence)
            current_len += len(sentence) + 1

        if current:
            chunk_text = " ".join(current)
            chunks.append(
                Chunk(
                    content=chunk_text,
                    index=idx,
                    start_char=char_pos,
                    end_char=char_pos + len(chunk_text),
                    metadata={"strategy": "paragraph", "from_split": True},
                )
            )

        return chunks

    def _get_overlap_paragraphs(
        self, paragraphs: List[str], target_overlap: int
    ) -> List[str]:
        """
        Get paragraphs for overlap.

        Args:
            paragraphs: List of paragraphs
            target_overlap: Target overlap size

        Returns:
            List of paragraphs for overlap
        """
        overlap = []
        current_len = 0

        for para in reversed(paragraphs):
            if current_len + len(para) <= target_overlap:
                overlap.insert(0, para)
                current_len += len(para) + 2
            else:
                break

        return overlap
