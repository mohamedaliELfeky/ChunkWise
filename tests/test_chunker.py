"""
Tests for Main Chunker Entry Point
"""

import pytest
from chunkwise import Chunker, chunk_text, chunk_arabic
from chunkwise.config import ChunkConfig, Strategy


class TestChunker:
    """Tests for main Chunker class."""

    def test_default_chunking(self, english_text):
        """Test chunking with default settings."""
        chunker = Chunker()
        chunks = chunker.chunk(english_text)

        assert len(chunks) >= 1
        assert all(hasattr(c, "content") for c in chunks)
        assert all(hasattr(c, "index") for c in chunks)

    def test_strategy_selection(self, english_text):
        """Test different strategy selections."""
        strategies = ["recursive", "sentence", "paragraph", "character", "word"]

        for strategy in strategies:
            chunker = Chunker(strategy=strategy, chunk_size=300)
            chunks = chunker.chunk(english_text)
            assert len(chunks) >= 1

    def test_callable_interface(self, english_text):
        """Test that Chunker can be called directly."""
        chunker = Chunker(chunk_size=300)
        chunks = chunker(english_text)

        assert len(chunks) >= 1

    def test_chunk_documents(self, english_text, arabic_text):
        """Test chunking multiple documents."""
        chunker = Chunker(chunk_size=300)
        batches = chunker.chunk_documents([english_text, arabic_text])

        assert len(batches) == 2
        assert all(len(batch.chunks) >= 1 for batch in batches)

    def test_config_object(self, english_text):
        """Test using ChunkConfig object."""
        config = ChunkConfig(
            strategy=Strategy.RECURSIVE,
            chunk_size=300,
            chunk_overlap=30,
        )
        chunker = Chunker(config=config)
        chunks = chunker.chunk(english_text)

        assert len(chunks) >= 1

    def test_repr(self):
        """Test string representation."""
        chunker = Chunker(strategy="recursive", chunk_size=512)
        repr_str = repr(chunker)

        assert "recursive" in repr_str
        assert "512" in repr_str


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_chunk_text(self, english_text):
        """Test chunk_text function."""
        chunks = chunk_text(english_text, chunk_size=300)

        assert len(chunks) >= 1

    def test_chunk_arabic(self, arabic_text):
        """Test chunk_arabic function."""
        chunks = chunk_arabic(arabic_text, chunk_size=200)

        assert len(chunks) >= 1
        # Should have Arabic language in metadata
        for chunk in chunks:
            if "language" in chunk.metadata:
                assert chunk.metadata["language"] in ["ar", "mixed"]
