"""
Tests for Sentence-Based Chunking Strategies
"""

import pytest
from chunkwise.strategies.sentence import (
    SentenceChunker,
    MultiSentenceChunker,
)
from chunkwise.config import ChunkConfig


class TestSentenceChunker:
    """Tests for SentenceChunker."""

    def test_english_chunking(self, english_text):
        """Test sentence chunking for English text."""
        chunker = SentenceChunker(chunk_size=500)
        chunks = chunker.chunk(english_text)

        assert len(chunks) >= 1
        # Each chunk should end with sentence-ending punctuation
        for chunk in chunks:
            assert chunk.content.rstrip()[-1] in ".!?"

    def test_arabic_chunking(self, arabic_text):
        """Test sentence chunking for Arabic text."""
        chunker = SentenceChunker(chunk_size=300, language="ar")
        chunks = chunker.chunk(arabic_text)

        assert len(chunks) >= 1
        # Each chunk should end with sentence-ending punctuation
        for chunk in chunks:
            last_char = chunk.content.rstrip()[-1]
            assert last_char in ".!?؟؛"

    def test_sentence_count_metadata(self, english_text):
        """Test that sentence count is in metadata."""
        chunker = SentenceChunker(chunk_size=500)
        chunks = chunker.chunk(english_text)

        for chunk in chunks:
            assert "sentence_count" in chunk.metadata
            assert chunk.metadata["sentence_count"] >= 1

    def test_language_detection(self, english_text, arabic_text):
        """Test automatic language detection."""
        chunker = SentenceChunker(chunk_size=500, language="auto")

        en_chunks = chunker.chunk(english_text)
        ar_chunks = chunker.chunk(arabic_text)

        # Both should produce valid chunks
        assert len(en_chunks) >= 1
        assert len(ar_chunks) >= 1


class TestMultiSentenceChunker:
    """Tests for MultiSentenceChunker."""

    def test_fixed_sentence_count(self, english_text):
        """Test chunking with fixed sentence count."""
        chunker = MultiSentenceChunker(sentences_per_chunk=3)
        chunks = chunker.chunk(english_text)

        assert len(chunks) >= 1
        for chunk in chunks:
            assert "sentence_count" in chunk.metadata
            assert chunk.metadata["sentence_count"] <= 4  # Allow some flexibility

    def test_arabic_multi_sentence(self, arabic_text):
        """Test multi-sentence chunking for Arabic."""
        chunker = MultiSentenceChunker(sentences_per_chunk=2, language="ar")
        chunks = chunker.chunk(arabic_text)

        assert len(chunks) >= 1
