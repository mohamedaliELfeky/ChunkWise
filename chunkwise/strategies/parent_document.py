"""
Parent Document Retriever (Small-to-Big) Strategy

Retrieves small chunks for precision but returns parent documents for context.
"""

from typing import List, Optional, Dict
from dataclasses import dataclass, field

from chunkwise.base import BaseChunker
from chunkwise.chunk import Chunk, ChunkBatch
from chunkwise.config import ChunkConfig


@dataclass
class ParentDocument:
    """Represents a parent document with its child chunks."""

    content: str
    index: int
    start_char: int
    end_char: int
    children: List[Chunk] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class ParentDocumentChunker(BaseChunker):
    """
    Parent Document Retriever pattern (Small-to-Big).

    Creates small chunks for retrieval but maintains references to larger
    parent documents. When a small chunk is retrieved, the full parent
    document can be returned to the LLM for more context.

    This pattern is useful when:
    - You need precise retrieval (small chunks)
    - But LLM needs more context (larger documents)

    Similar to LangChain's ParentDocumentRetriever.

    Example:
        >>> chunker = ParentDocumentChunker(
        ...     child_chunk_size=200,
        ...     parent_chunk_size=1000,
        ... )
        >>> chunks = chunker.chunk(document)
        >>>
        >>> # Each chunk has reference to parent
        >>> for chunk in chunks:
        ...     parent = chunk.metadata["parent_content"]
    """

    def __init__(
        self,
        config: Optional[ChunkConfig] = None,
        child_chunk_size: int = 200,
        parent_chunk_size: int = 1000,
        child_overlap: int = 20,
        parent_overlap: int = 100,
        **kwargs,
    ):
        """
        Initialize Parent Document chunker.

        Args:
            config: ChunkConfig instance
            child_chunk_size: Size of child chunks (for retrieval)
            parent_chunk_size: Size of parent chunks (for context)
            child_overlap: Overlap between child chunks
            parent_overlap: Overlap between parent chunks
            **kwargs: Override config parameters
        """
        # Use child size as main chunk size
        if config:
            config.chunk_size = child_chunk_size
            config.chunk_overlap = child_overlap
        else:
            kwargs["chunk_size"] = child_chunk_size
            kwargs["chunk_overlap"] = child_overlap

        super().__init__(config, **kwargs)
        self.child_chunk_size = child_chunk_size
        self.parent_chunk_size = parent_chunk_size
        self.child_overlap = child_overlap
        self.parent_overlap = parent_overlap

    def _chunk_text(self, text: str) -> List[Chunk]:
        """
        Create child chunks with parent references.

        Args:
            text: Input text

        Returns:
            List of child chunks with parent metadata
        """
        if not text:
            return []

        # Create parent chunks first
        parents = self._create_parents(text)

        # Create child chunks within each parent
        all_children = []
        child_index = 0

        for parent in parents:
            children = self._create_children(parent)

            for child in children:
                child.index = child_index
                # Adjust positions relative to full document
                child.start_char = parent.start_char + child.start_char
                child.end_char = parent.start_char + child.end_char

                # Add parent reference
                child.metadata.update({
                    "strategy": "parent_document",
                    "parent_index": parent.index,
                    "parent_content": parent.content,
                    "parent_start": parent.start_char,
                    "parent_end": parent.end_char,
                })

                all_children.append(child)
                child_index += 1

        return all_children

    def _create_parents(self, text: str) -> List[ParentDocument]:
        """
        Create parent document chunks.

        Args:
            text: Full document text

        Returns:
            List of ParentDocument objects
        """
        from chunkwise.strategies.recursive import RecursiveChunker

        parent_config = ChunkConfig(
            chunk_size=self.parent_chunk_size,
            chunk_overlap=self.parent_overlap,
        )
        parent_chunker = RecursiveChunker(config=parent_config)
        parent_chunks = parent_chunker._chunk_text(text)

        parents = []
        for chunk in parent_chunks:
            parents.append(
                ParentDocument(
                    content=chunk.content,
                    index=chunk.index,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    metadata=chunk.metadata,
                )
            )

        return parents

    def _create_children(self, parent: ParentDocument) -> List[Chunk]:
        """
        Create child chunks from a parent.

        Args:
            parent: Parent document

        Returns:
            List of child Chunk objects
        """
        from chunkwise.strategies.recursive import RecursiveChunker

        child_config = ChunkConfig(
            chunk_size=self.child_chunk_size,
            chunk_overlap=self.child_overlap,
        )
        child_chunker = RecursiveChunker(config=child_config)
        return child_chunker._chunk_text(parent.content)

    def get_parent(self, chunk: Chunk) -> str:
        """
        Get the parent content for a retrieved chunk.

        Args:
            chunk: Retrieved child chunk

        Returns:
            Parent document content
        """
        return chunk.metadata.get("parent_content", chunk.content)

    def expand_results(self, chunks: List[Chunk]) -> List[str]:
        """
        Expand retrieved chunks to their parent documents.

        Deduplicates parents when multiple children from same parent.

        Args:
            chunks: Retrieved child chunks

        Returns:
            List of unique parent contents
        """
        seen_parents = set()
        parents = []

        for chunk in chunks:
            parent_idx = chunk.metadata.get("parent_index")
            if parent_idx not in seen_parents:
                seen_parents.add(parent_idx)
                parents.append(self.get_parent(chunk))

        return parents


class SmallToBigChunker(ParentDocumentChunker):
    """
    Alias for ParentDocumentChunker.

    Small-to-Big retrieval pattern.
    """

    pass


class BigToSmallChunker(BaseChunker):
    """
    Big-to-Small retrieval pattern.

    First retrieves large chunks, then drills down to find
    the most relevant small section within.

    Example:
        >>> chunker = BigToSmallChunker(
        ...     big_chunk_size=2000,
        ...     small_chunk_size=200,
        ... )
    """

    def __init__(
        self,
        config: Optional[ChunkConfig] = None,
        big_chunk_size: int = 2000,
        small_chunk_size: int = 200,
        **kwargs,
    ):
        """
        Initialize Big-to-Small chunker.

        Args:
            config: ChunkConfig instance
            big_chunk_size: Size of big chunks
            small_chunk_size: Size of small chunks
            **kwargs: Override config parameters
        """
        super().__init__(config, **kwargs)
        self.big_chunk_size = big_chunk_size
        self.small_chunk_size = small_chunk_size

    def _chunk_text(self, text: str) -> List[Chunk]:
        """
        Create hierarchical big/small chunks.

        Args:
            text: Input text

        Returns:
            List of chunks with hierarchy metadata
        """
        if not text:
            return []

        from chunkwise.strategies.recursive import RecursiveChunker

        # Create big chunks
        big_config = ChunkConfig(chunk_size=self.big_chunk_size, chunk_overlap=0)
        big_chunker = RecursiveChunker(config=big_config)
        big_chunks = big_chunker._chunk_text(text)

        all_chunks = []
        chunk_index = 0

        for big_idx, big_chunk in enumerate(big_chunks):
            # Create small chunks within
            small_config = ChunkConfig(chunk_size=self.small_chunk_size, chunk_overlap=0)
            small_chunker = RecursiveChunker(config=small_config)
            small_chunks = small_chunker._chunk_text(big_chunk.content)

            for small_idx, small_chunk in enumerate(small_chunks):
                small_chunk.index = chunk_index
                small_chunk.start_char = big_chunk.start_char + small_chunk.start_char
                small_chunk.end_char = big_chunk.start_char + small_chunk.end_char
                small_chunk.metadata.update({
                    "strategy": "big_to_small",
                    "big_chunk_index": big_idx,
                    "big_chunk_content": big_chunk.content,
                    "small_chunk_index": small_idx,
                    "hierarchy_level": "small",
                })
                all_chunks.append(small_chunk)
                chunk_index += 1

        return all_chunks
