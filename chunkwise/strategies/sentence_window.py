"""
Sentence Window Chunking Strategy

Retrieves small sentence-level chunks but expands context at generation time.
This decouples retrieval optimization from generation optimization.
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass

from chunkwise.base import BaseChunker
from chunkwise.chunk import Chunk
from chunkwise.config import ChunkConfig
from chunkwise.language.detector import detect_language


@dataclass
class SentenceWindow:
    """Represents a sentence with its surrounding context window."""

    sentence: str
    sentence_index: int
    window_before: List[str]
    window_after: List[str]
    start_char: int
    end_char: int


class SentenceWindowChunker(BaseChunker):
    """
    Sentence Window chunking for optimized retrieval + generation.

    This strategy stores individual sentences for retrieval but maintains
    references to surrounding sentences. At generation time, the window
    can be expanded to provide more context to the LLM.

    Benefits:
    - Better retrieval precision (small, focused chunks)
    - Better generation quality (expanded context)
    - Decouples retrieval and generation optimization

    From ARAGOG research: Sentence Window achieves highest precision
    among retrieval techniques.

    Example:
        >>> chunker = SentenceWindowChunker(window_size=2)
        >>> chunks = chunker.chunk(text)
        >>>
        >>> # Each chunk is a single sentence
        >>> # But metadata contains surrounding context
        >>> for chunk in chunks:
        ...     window = chunk.metadata["window_context"]
        ...     full_context = window["before"] + chunk.content + window["after"]
    """

    def __init__(
        self,
        config: Optional[ChunkConfig] = None,
        window_size: int = 3,
        store_window_text: bool = True,
        **kwargs,
    ):
        """
        Initialize Sentence Window chunker.

        Args:
            config: ChunkConfig instance
            window_size: Number of sentences before/after to include
            store_window_text: Store actual window text in metadata
            **kwargs: Override config parameters
        """
        super().__init__(config, **kwargs)
        self.window_size = window_size
        self.store_window_text = store_window_text

    def _chunk_text(self, text: str) -> List[Chunk]:
        """
        Split text into sentence windows.

        Args:
            text: Input text

        Returns:
            List of Chunk objects (one per sentence)
        """
        if not text:
            return []

        # Split into sentences
        sentences = self._split_sentences(text)

        if not sentences:
            return [
                Chunk(
                    content=text,
                    index=0,
                    start_char=0,
                    end_char=len(text),
                    metadata={"strategy": "sentence_window"},
                )
            ]

        # Create sentence windows
        windows = self._create_windows(sentences, text)

        # Convert to chunks
        return self._windows_to_chunks(windows)

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences based on language."""
        lang = detect_language(text)

        if lang == "ar":
            from chunkwise.language.arabic.sentence_splitter import split_arabic_sentences

            return split_arabic_sentences(text, min_length=5)
        else:
            from chunkwise.language.english.sentence_splitter import split_english_sentences

            return split_english_sentences(text, min_length=5)

    def _create_windows(
        self, sentences: List[str], text: str
    ) -> List[SentenceWindow]:
        """
        Create windows for each sentence.

        Args:
            sentences: List of sentences
            text: Original text

        Returns:
            List of SentenceWindow objects
        """
        windows = []
        char_positions = self._find_sentence_positions(sentences, text)

        for i, sentence in enumerate(sentences):
            # Get surrounding sentences
            start_idx = max(0, i - self.window_size)
            end_idx = min(len(sentences), i + self.window_size + 1)

            window_before = sentences[start_idx:i]
            window_after = sentences[i + 1 : end_idx]

            start_char, end_char = char_positions[i]

            windows.append(
                SentenceWindow(
                    sentence=sentence,
                    sentence_index=i,
                    window_before=window_before,
                    window_after=window_after,
                    start_char=start_char,
                    end_char=end_char,
                )
            )

        return windows

    def _find_sentence_positions(
        self, sentences: List[str], text: str
    ) -> List[Tuple[int, int]]:
        """Find character positions of each sentence."""
        positions = []
        current_pos = 0

        for sentence in sentences:
            # Find sentence in text
            start = text.find(sentence[:30], current_pos)
            if start < 0:
                start = current_pos
            end = start + len(sentence)
            positions.append((start, end))
            current_pos = end

        return positions

    def _windows_to_chunks(self, windows: List[SentenceWindow]) -> List[Chunk]:
        """
        Convert sentence windows to Chunk objects.

        Args:
            windows: List of SentenceWindow objects

        Returns:
            List of Chunk objects
        """
        chunks = []

        for i, window in enumerate(windows):
            metadata = {
                "strategy": "sentence_window",
                "sentence_index": window.sentence_index,
                "total_sentences": len(windows),
                "window_size": self.window_size,
                "window_indices": {
                    "before": list(
                        range(
                            max(0, window.sentence_index - self.window_size),
                            window.sentence_index,
                        )
                    ),
                    "after": list(
                        range(
                            window.sentence_index + 1,
                            min(len(windows), window.sentence_index + self.window_size + 1),
                        )
                    ),
                },
            }

            if self.store_window_text:
                metadata["window_context"] = {
                    "before": " ".join(window.window_before),
                    "after": " ".join(window.window_after),
                }
                metadata["expanded_content"] = " ".join(
                    window.window_before + [window.sentence] + window.window_after
                )

            chunks.append(
                Chunk(
                    content=window.sentence,
                    index=i,
                    start_char=window.start_char,
                    end_char=window.end_char,
                    metadata=metadata,
                )
            )

        return chunks

    def expand_chunk(self, chunk: Chunk) -> str:
        """
        Expand a chunk to include its full window context.

        Args:
            chunk: A chunk from this chunker

        Returns:
            Expanded text including window
        """
        if "expanded_content" in chunk.metadata:
            return chunk.metadata["expanded_content"]

        window = chunk.metadata.get("window_context", {})
        before = window.get("before", "")
        after = window.get("after", "")

        parts = []
        if before:
            parts.append(before)
        parts.append(chunk.content)
        if after:
            parts.append(after)

        return " ".join(parts)


class AutoMergingChunker(BaseChunker):
    """
    Auto-merging retriever pattern.

    Stores small leaf chunks but automatically merges them into
    parent chunks when multiple leaves from the same parent are retrieved.

    Similar to LlamaIndex's AutoMergingRetriever.

    Example:
        >>> chunker = AutoMergingChunker(
        ...     leaf_size=128,
        ...     parent_size=512,
        ...     merge_threshold=0.5  # Merge if 50%+ of leaves retrieved
        ... )
    """

    def __init__(
        self,
        config: Optional[ChunkConfig] = None,
        leaf_size: int = 128,
        parent_size: int = 512,
        merge_threshold: float = 0.5,
        **kwargs,
    ):
        """
        Initialize Auto-merging chunker.

        Args:
            config: ChunkConfig instance
            leaf_size: Size of leaf chunks (for retrieval)
            parent_size: Size of parent chunks (for merging)
            merge_threshold: Fraction of leaves needed to merge
            **kwargs: Override config parameters
        """
        # Override chunk_size with leaf_size
        if config:
            config.chunk_size = leaf_size
        else:
            kwargs["chunk_size"] = leaf_size

        super().__init__(config, **kwargs)
        self.leaf_size = leaf_size
        self.parent_size = parent_size
        self.merge_threshold = merge_threshold

    def _chunk_text(self, text: str) -> List[Chunk]:
        """
        Create leaf chunks with parent references.

        Args:
            text: Input text

        Returns:
            List of leaf chunks with parent metadata
        """
        if not text:
            return []

        # First create parent chunks
        from chunkwise.strategies.recursive import RecursiveChunker

        parent_config = ChunkConfig(
            chunk_size=self.parent_size,
            chunk_overlap=0,
        )
        parent_chunker = RecursiveChunker(config=parent_config)
        parent_chunks = parent_chunker._chunk_text(text)

        # Then create leaf chunks within each parent
        leaf_config = ChunkConfig(
            chunk_size=self.leaf_size,
            chunk_overlap=0,
        )
        leaf_chunker = RecursiveChunker(config=leaf_config)

        all_leaves = []
        leaf_index = 0

        for parent_idx, parent in enumerate(parent_chunks):
            # Create leaves from parent content
            leaves = leaf_chunker._chunk_text(parent.content)

            for local_idx, leaf in enumerate(leaves):
                leaf.index = leaf_index
                leaf.start_char = parent.start_char + leaf.start_char
                leaf.end_char = parent.start_char + leaf.end_char
                leaf.metadata.update({
                    "strategy": "auto_merging",
                    "parent_index": parent_idx,
                    "parent_content": parent.content,
                    "local_index": local_idx,
                    "total_leaves_in_parent": len(leaves),
                    "merge_threshold": self.merge_threshold,
                })
                all_leaves.append(leaf)
                leaf_index += 1

        return all_leaves

    def should_merge(self, retrieved_chunks: List[Chunk]) -> dict:
        """
        Determine which chunks should be merged to parents.

        Args:
            retrieved_chunks: Chunks retrieved from search

        Returns:
            Dict mapping parent_index to merged content
        """
        # Group by parent
        parent_groups = {}
        for chunk in retrieved_chunks:
            parent_idx = chunk.metadata.get("parent_index")
            if parent_idx is not None:
                if parent_idx not in parent_groups:
                    parent_groups[parent_idx] = {
                        "chunks": [],
                        "total": chunk.metadata.get("total_leaves_in_parent", 1),
                        "parent_content": chunk.metadata.get("parent_content", ""),
                    }
                parent_groups[parent_idx]["chunks"].append(chunk)

        # Determine which to merge
        merge_results = {}
        for parent_idx, group in parent_groups.items():
            ratio = len(group["chunks"]) / group["total"]
            if ratio >= self.merge_threshold:
                merge_results[parent_idx] = group["parent_content"]

        return merge_results
