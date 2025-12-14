"""
Hierarchical Chunking Strategy

Multi-level tree structure preserving document hierarchy.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

from chunkwise.base import BaseChunker
from chunkwise.chunk import Chunk
from chunkwise.config import ChunkConfig


@dataclass
class HierarchyNode:
    """Represents a node in the document hierarchy tree."""

    content: str
    level: int
    index: int
    start_char: int
    end_char: int
    parent_index: Optional[int] = None
    children_indices: List[int] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class HierarchicalChunker(BaseChunker):
    """
    Hierarchical chunking with multi-level tree structure.

    Creates chunks at multiple levels of granularity:
    - Level 0: Full document / major sections
    - Level 1: Paragraphs
    - Level 2: Sentences

    Preserves parent-child relationships between levels.

    Use cases:
    - Complex technical documents
    - Legal contracts
    - Research papers
    - Textbooks

    Example:
        >>> chunker = HierarchicalChunker(
        ...     levels=[2000, 500, 100]  # Section, paragraph, sentence sizes
        ... )
        >>> chunks = chunker.chunk(document)
        >>>
        >>> # Navigate hierarchy
        >>> for chunk in chunks:
        ...     level = chunk.metadata["hierarchy_level"]
        ...     parent = chunk.metadata.get("parent_index")
    """

    def __init__(
        self,
        config: Optional[ChunkConfig] = None,
        levels: Optional[List[int]] = None,
        level_names: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Initialize Hierarchical chunker.

        Args:
            config: ChunkConfig instance
            levels: Chunk sizes for each level [large, medium, small]
            level_names: Names for each level ["section", "paragraph", "sentence"]
            **kwargs: Override config parameters
        """
        super().__init__(config, **kwargs)
        self.levels = levels or [2000, 500, 100]
        self.level_names = level_names or ["section", "paragraph", "sentence"]

        # Ensure we have names for all levels
        while len(self.level_names) < len(self.levels):
            self.level_names.append(f"level_{len(self.level_names)}")

    def _chunk_text(self, text: str) -> List[Chunk]:
        """
        Create hierarchical chunks.

        Args:
            text: Input text

        Returns:
            List of all chunks with hierarchy metadata
        """
        if not text:
            return []

        # Build hierarchy tree
        nodes = self._build_hierarchy(text)

        # Convert to chunks
        return self._nodes_to_chunks(nodes)

    def _build_hierarchy(self, text: str) -> List[HierarchyNode]:
        """
        Build hierarchical structure of nodes.

        Args:
            text: Input text

        Returns:
            List of HierarchyNode objects
        """
        from chunkwise.strategies.recursive import RecursiveChunker

        all_nodes = []
        node_index = 0

        # Process each level
        level_chunks: Dict[int, List[tuple]] = {}  # level -> [(content, start, end, parent_idx)]

        # Level 0: Start with full text
        level_chunks[0] = [(text, 0, len(text), None)]

        for level, chunk_size in enumerate(self.levels):
            current_level_nodes = []

            for content, start_offset, end_offset, parent_idx in level_chunks.get(level, []):
                # Create chunks at this level
                level_config = ChunkConfig(chunk_size=chunk_size, chunk_overlap=0)
                chunker = RecursiveChunker(config=level_config)
                chunks = chunker._chunk_text(content)

                for chunk in chunks:
                    node = HierarchyNode(
                        content=chunk.content,
                        level=level,
                        index=node_index,
                        start_char=start_offset + chunk.start_char,
                        end_char=start_offset + chunk.end_char,
                        parent_index=parent_idx,
                        metadata={
                            "level_name": self.level_names[level],
                        },
                    )

                    # Track for next level
                    if level + 1 < len(self.levels):
                        if level + 1 not in level_chunks:
                            level_chunks[level + 1] = []
                        level_chunks[level + 1].append(
                            (chunk.content, node.start_char, node.end_char, node_index)
                        )

                    current_level_nodes.append(node)
                    node_index += 1

            all_nodes.extend(current_level_nodes)

        # Set children indices
        self._set_children_indices(all_nodes)

        return all_nodes

    def _set_children_indices(self, nodes: List[HierarchyNode]) -> None:
        """Set children indices for all nodes."""
        for node in nodes:
            if node.parent_index is not None:
                parent = next(
                    (n for n in nodes if n.index == node.parent_index), None
                )
                if parent:
                    parent.children_indices.append(node.index)

    def _nodes_to_chunks(self, nodes: List[HierarchyNode]) -> List[Chunk]:
        """Convert HierarchyNodes to Chunks."""
        chunks = []

        for node in nodes:
            chunks.append(
                Chunk(
                    content=node.content,
                    index=node.index,
                    start_char=node.start_char,
                    end_char=node.end_char,
                    metadata={
                        "strategy": "hierarchical",
                        "hierarchy_level": node.level,
                        "level_name": node.metadata.get("level_name"),
                        "parent_index": node.parent_index,
                        "children_indices": node.children_indices,
                        "is_leaf": len(node.children_indices) == 0,
                    },
                )
            )

        return chunks

    def get_level_chunks(self, chunks: List[Chunk], level: int) -> List[Chunk]:
        """
        Get all chunks at a specific level.

        Args:
            chunks: All hierarchical chunks
            level: Level to filter (0=largest, increasing=smaller)

        Returns:
            Chunks at the specified level
        """
        return [c for c in chunks if c.metadata.get("hierarchy_level") == level]

    def get_leaf_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Get only leaf-level chunks (smallest level).

        Args:
            chunks: All hierarchical chunks

        Returns:
            Leaf chunks only
        """
        return [c for c in chunks if c.metadata.get("is_leaf", False)]

    def get_parent_chunk(
        self, chunk: Chunk, all_chunks: List[Chunk]
    ) -> Optional[Chunk]:
        """
        Get the parent chunk.

        Args:
            chunk: Child chunk
            all_chunks: All chunks

        Returns:
            Parent chunk or None
        """
        parent_idx = chunk.metadata.get("parent_index")
        if parent_idx is None:
            return None

        return next((c for c in all_chunks if c.index == parent_idx), None)

    def get_children_chunks(
        self, chunk: Chunk, all_chunks: List[Chunk]
    ) -> List[Chunk]:
        """
        Get children chunks.

        Args:
            chunk: Parent chunk
            all_chunks: All chunks

        Returns:
            List of children chunks
        """
        children_indices = chunk.metadata.get("children_indices", [])
        return [c for c in all_chunks if c.index in children_indices]

    def navigate_up(
        self, chunk: Chunk, all_chunks: List[Chunk], levels: int = 1
    ) -> Optional[Chunk]:
        """
        Navigate up the hierarchy.

        Args:
            chunk: Starting chunk
            all_chunks: All chunks
            levels: Number of levels to go up

        Returns:
            Ancestor chunk or None
        """
        current = chunk
        for _ in range(levels):
            parent = self.get_parent_chunk(current, all_chunks)
            if parent is None:
                return current
            current = parent
        return current

    def navigate_down(
        self, chunk: Chunk, all_chunks: List[Chunk]
    ) -> List[Chunk]:
        """
        Navigate down to leaf chunks.

        Args:
            chunk: Starting chunk
            all_chunks: All chunks

        Returns:
            All leaf descendants
        """
        if chunk.metadata.get("is_leaf", False):
            return [chunk]

        children = self.get_children_chunks(chunk, all_chunks)
        leaves = []

        for child in children:
            leaves.extend(self.navigate_down(child, all_chunks))

        return leaves
