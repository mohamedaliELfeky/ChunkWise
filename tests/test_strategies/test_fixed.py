"""
Tests for Fixed-Size Chunking Strategies
"""

import pytest
from chunkwise.strategies.fixed import (
    CharacterChunker,
    TokenChunker,
    WordChunker,
)
from chunkwise.config import ChunkConfig


class TestCharacterChunker:
    """Tests for CharacterChunker."""

    def test_basic_chunking(self, english_text):
        """Test basic character-based chunking."""
        chunker = CharacterChunker(chunk_size=200, chunk_overlap=20)
        chunks = chunker.chunk(english_text)

        assert len(chunks) > 1
        assert all(len(c.content) <= 220 for c in chunks)  # Allow some flexibility

    def test_small_text(self, short_text):
        """Test chunking text smaller than chunk_size."""
        chunker = CharacterChunker(chunk_size=500)
        chunks = chunker.chunk(short_text)

        assert len(chunks) == 1
        assert chunks[0].content.strip() == short_text.strip()

    def test_empty_text(self, empty_text):
        """Test chunking empty text."""
        chunker = CharacterChunker(chunk_size=100)
        chunks = chunker.chunk(empty_text)

        assert len(chunks) == 0

    def test_chunk_indices(self, english_text):
        """Test that chunk indices are sequential."""
        chunker = CharacterChunker(chunk_size=200)
        chunks = chunker.chunk(english_text)

        for i, chunk in enumerate(chunks):
            assert chunk.index == i

    def test_overlap(self, english_text):
        """Test that overlap is applied correctly."""
        chunker = CharacterChunker(chunk_size=200, chunk_overlap=50)
        chunks = chunker.chunk(english_text)

        if len(chunks) > 1:
            # Check that consecutive chunks have overlapping content
            for i in range(len(chunks) - 1):
                chunk1_end = chunks[i].content[-30:]
                chunk2_start = chunks[i + 1].content[:30]
                # There should be some common words
                assert any(word in chunk2_start for word in chunk1_end.split() if len(word) > 3)


class TestWordChunker:
    """Tests for WordChunker."""

    def test_basic_chunking(self, english_text):
        """Test basic word-based chunking."""
        chunker = WordChunker(chunk_size=50, chunk_overlap=5)
        chunks = chunker.chunk(english_text)

        assert len(chunks) > 1
        for chunk in chunks:
            word_count = len(chunk.content.split())
            assert word_count <= 60  # Allow some flexibility

    def test_word_count_metadata(self, english_text):
        """Test that word count is in metadata."""
        chunker = WordChunker(chunk_size=50)
        chunks = chunker.chunk(english_text)

        for chunk in chunks:
            assert "word_count" in chunk.metadata


class TestTokenChunker:
    """Tests for TokenChunker."""

    def test_basic_chunking(self, english_text):
        """Test basic token-based chunking."""
        chunker = TokenChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk(english_text)

        assert len(chunks) > 1

    def test_token_count_metadata(self, english_text):
        """Test that token count is in metadata."""
        chunker = TokenChunker(chunk_size=100)
        chunks = chunker.chunk(english_text)

        for chunk in chunks:
            assert "token_count" in chunk.metadata
            assert chunk.metadata["token_count"] <= 120  # Allow some flexibility
