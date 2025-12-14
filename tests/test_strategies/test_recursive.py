"""
Tests for Recursive Chunking Strategy
"""

import pytest
from chunkwise.strategies.recursive import RecursiveChunker
from chunkwise.config import ChunkConfig


class TestRecursiveChunker:
    """Tests for RecursiveChunker."""

    def test_basic_chunking(self, english_text):
        """Test basic recursive chunking."""
        chunker = RecursiveChunker(chunk_size=300, chunk_overlap=30)
        chunks = chunker.chunk(english_text)

        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.content) <= 350  # Allow some flexibility

    def test_respects_paragraphs(self, markdown_text):
        """Test that chunking respects paragraph boundaries."""
        chunker = RecursiveChunker(chunk_size=500)
        chunks = chunker.chunk(markdown_text)

        # Most chunks should not start or end mid-sentence
        for chunk in chunks:
            content = chunk.content.strip()
            # Should start with capital letter or header
            assert content[0].isupper() or content.startswith("#")

    def test_arabic_chunking(self, arabic_text):
        """Test recursive chunking for Arabic."""
        chunker = RecursiveChunker(chunk_size=200, language="ar")
        chunks = chunker.chunk(arabic_text)

        assert len(chunks) >= 1

    def test_custom_separators(self, english_text):
        """Test with custom separators."""
        chunker = RecursiveChunker(
            chunk_size=300,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = chunker.chunk(english_text)

        assert len(chunks) >= 1

    def test_overlap_content(self, english_text):
        """Test that overlap produces actual overlapping content."""
        chunker = RecursiveChunker(chunk_size=300, chunk_overlap=50)
        chunks = chunker.chunk(english_text)

        if len(chunks) > 1:
            # Some content from end of chunk 0 should appear in chunk 1
            chunk0_words = set(chunks[0].content[-100:].split())
            chunk1_words = set(chunks[1].content[:100].split())
            common = chunk0_words & chunk1_words
            assert len(common) > 0  # Should have some overlap

    def test_chunk_positions(self, english_text):
        """Test that chunk positions are valid."""
        chunker = RecursiveChunker(chunk_size=300)
        chunks = chunker.chunk(english_text)

        for chunk in chunks:
            assert chunk.start_char >= 0
            assert chunk.end_char > chunk.start_char
