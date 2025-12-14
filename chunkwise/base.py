"""
Base Chunker Abstract Class

Defines the interface that all chunking strategies must implement.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, AsyncIterator, Iterator
import asyncio

from chunkwise.chunk import Chunk, ChunkBatch
from chunkwise.config import ChunkConfig, Language
from chunkwise.exceptions import ChunkSizeError


class BaseChunker(ABC):
    """
    Abstract base class for all chunking strategies.

    All chunking strategies must inherit from this class and implement
    the `_chunk_text` method.

    Attributes:
        config: ChunkConfig instance with chunking parameters

    Example:
        >>> class MyChunker(BaseChunker):
        ...     def _chunk_text(self, text: str) -> List[Chunk]:
        ...         # Implementation here
        ...         pass
        ...
        >>> chunker = MyChunker(config=ChunkConfig(chunk_size=100))
        >>> chunks = chunker.chunk("Hello world")
    """

    def __init__(self, config: Optional[ChunkConfig] = None, **kwargs):
        """
        Initialize the chunker.

        Args:
            config: ChunkConfig instance. If not provided, uses defaults.
            **kwargs: Override specific config parameters
        """
        if config is None:
            config = ChunkConfig(**kwargs)
        else:
            # Apply any kwargs overrides to config
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        self.config = config
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.config.chunk_size <= 0:
            raise ChunkSizeError("chunk_size must be positive")
        if self.config.chunk_overlap < 0:
            raise ChunkSizeError("chunk_overlap cannot be negative")
        if self.config.chunk_overlap >= self.config.chunk_size:
            raise ChunkSizeError("chunk_overlap must be less than chunk_size")
        if self.config.min_chunk_size < 0:
            raise ChunkSizeError("min_chunk_size cannot be negative")

    @abstractmethod
    def _chunk_text(self, text: str) -> List[Chunk]:
        """
        Internal method to perform the actual chunking.

        This method must be implemented by all subclasses.

        Args:
            text: The text to chunk

        Returns:
            List of Chunk objects
        """
        pass

    def chunk(self, text: str) -> List[Chunk]:
        """
        Chunk the input text.

        This is the main public method for chunking text.
        It handles preprocessing and postprocessing around
        the strategy-specific _chunk_text method.

        Args:
            text: The text to chunk

        Returns:
            List of Chunk objects

        Example:
            >>> chunker = RecursiveChunker(chunk_size=100)
            >>> chunks = chunker.chunk("Your text here...")
            >>> for chunk in chunks:
            ...     print(f"Chunk {chunk.index}: {len(chunk)} chars")
        """
        if not text or not text.strip():
            return []

        # Preprocess text
        processed_text = self._preprocess(text)

        # Perform chunking
        chunks = self._chunk_text(processed_text)

        # Postprocess chunks
        chunks = self._postprocess(chunks, text)

        return chunks

    async def chunk_async(self, text: str) -> List[Chunk]:
        """
        Asynchronously chunk the input text.

        Useful for strategies that involve API calls (e.g., semantic, agentic).
        Default implementation runs synchronously but can be overridden.

        Args:
            text: The text to chunk

        Returns:
            List of Chunk objects
        """
        # Default implementation - subclasses can override for true async
        return await asyncio.to_thread(self.chunk, text)

    def chunk_documents(self, documents: List[str]) -> List[ChunkBatch]:
        """
        Chunk multiple documents.

        Args:
            documents: List of document texts

        Returns:
            List of ChunkBatch objects, one per document
        """
        batches = []
        for i, doc in enumerate(documents):
            chunks = self.chunk(doc)
            batch = ChunkBatch(
                chunks=chunks,
                source=f"document_{i}",
                total_chars=len(doc),
                metadata={"index": i},
            )
            batches.append(batch)
        return batches

    async def chunk_documents_async(self, documents: List[str]) -> List[ChunkBatch]:
        """
        Asynchronously chunk multiple documents.

        Args:
            documents: List of document texts

        Returns:
            List of ChunkBatch objects
        """
        tasks = [self.chunk_async(doc) for doc in documents]
        results = await asyncio.gather(*tasks)

        batches = []
        for i, (doc, chunks) in enumerate(zip(documents, results)):
            batch = ChunkBatch(
                chunks=chunks,
                source=f"document_{i}",
                total_chars=len(doc),
                metadata={"index": i},
            )
            batches.append(batch)
        return batches

    def chunk_iter(self, text: str) -> Iterator[Chunk]:
        """
        Iterate over chunks one at a time.

        Useful for processing very large texts without
        loading all chunks into memory.

        Args:
            text: The text to chunk

        Yields:
            Chunk objects one at a time
        """
        chunks = self.chunk(text)
        for chunk in chunks:
            yield chunk

    async def chunk_iter_async(self, text: str) -> AsyncIterator[Chunk]:
        """
        Asynchronously iterate over chunks.

        Args:
            text: The text to chunk

        Yields:
            Chunk objects one at a time
        """
        chunks = await self.chunk_async(text)
        for chunk in chunks:
            yield chunk

    def _preprocess(self, text: str) -> str:
        """
        Preprocess text before chunking.

        Override in subclasses for custom preprocessing.

        Args:
            text: Raw input text

        Returns:
            Preprocessed text
        """
        # Remove null bytes and normalize whitespace
        text = text.replace("\x00", "")

        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        return text

    def _postprocess(self, chunks: List[Chunk], original_text: str) -> List[Chunk]:
        """
        Postprocess chunks after chunking.

        Override in subclasses for custom postprocessing.

        Args:
            chunks: List of chunks from _chunk_text
            original_text: The original input text

        Returns:
            Postprocessed list of chunks
        """
        # Filter out empty chunks
        chunks = [c for c in chunks if c.content.strip()]

        # Merge small chunks if configured
        if self.config.min_chunk_size > 0:
            chunks = self._merge_small_chunks(chunks)

        # Add metadata if configured
        if self.config.include_metadata:
            chunks = self._add_metadata(chunks, original_text)

        return chunks

    def _merge_small_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Merge chunks smaller than min_chunk_size with adjacent chunks.

        Args:
            chunks: List of chunks

        Returns:
            List of chunks with small chunks merged
        """
        if not chunks:
            return chunks

        min_size = self.config.min_chunk_size
        merged = []
        buffer = None

        for chunk in chunks:
            if buffer is None:
                if len(chunk.content) < min_size:
                    buffer = chunk
                else:
                    merged.append(chunk)
            else:
                # Merge buffer with current chunk
                combined_content = buffer.content + " " + chunk.content
                combined_chunk = Chunk(
                    content=combined_content,
                    index=buffer.index,
                    start_char=buffer.start_char,
                    end_char=chunk.end_char,
                    metadata={**buffer.metadata, **chunk.metadata},
                )

                if len(combined_content) < min_size:
                    buffer = combined_chunk
                else:
                    merged.append(combined_chunk)
                    buffer = None

        # Handle remaining buffer
        if buffer is not None:
            if merged:
                # Merge with previous chunk
                last = merged[-1]
                combined = Chunk(
                    content=last.content + " " + buffer.content,
                    index=last.index,
                    start_char=last.start_char,
                    end_char=buffer.end_char,
                    metadata={**last.metadata, **buffer.metadata},
                )
                merged[-1] = combined
            else:
                merged.append(buffer)

        # Re-index chunks
        for i, chunk in enumerate(merged):
            chunk.index = i

        return merged

    def _add_metadata(self, chunks: List[Chunk], original_text: str) -> List[Chunk]:
        """
        Add metadata to chunks.

        Args:
            chunks: List of chunks
            original_text: Original input text

        Returns:
            Chunks with added metadata
        """
        from chunkwise.language.detector import detect_language

        for chunk in chunks:
            # Add language detection
            if "language" not in chunk.metadata:
                try:
                    chunk.metadata["language"] = detect_language(chunk.content)
                except Exception:
                    chunk.metadata["language"] = "unknown"

            # Add token count if configured
            if self.config.compute_tokens and "token_count" not in chunk.metadata:
                chunk.metadata["token_count"] = self._count_tokens(chunk.content)

            # Add position info
            chunk.metadata["total_chunks"] = len(chunks)
            chunk.metadata["position"] = chunk.index / max(len(chunks) - 1, 1)

        return chunks

    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        try:
            import tiktoken

            enc = tiktoken.get_encoding(self.config.tokenizer_model)
            return len(enc.encode(text))
        except Exception:
            # Fallback to simple word count
            return len(text.split())

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"{self.__class__.__name__}("
            f"chunk_size={self.config.chunk_size}, "
            f"overlap={self.config.chunk_overlap}, "
            f"language={self.config.language})"
        )
