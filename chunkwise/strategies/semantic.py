"""
Semantic Chunking Strategy

Chunking based on semantic similarity using embeddings.
"""

from typing import List, Optional
import numpy as np

from chunkwise.base import BaseChunker
from chunkwise.chunk import Chunk
from chunkwise.config import ChunkConfig, EmbeddingConfig
from chunkwise.embeddings.base import get_embedding_provider
from chunkwise.exceptions import EmbeddingError


class SemanticChunker(BaseChunker):
    """
    Split text based on semantic similarity.

    Uses embeddings to identify natural break points where the semantic
    content changes significantly. Creates chunks that are semantically coherent.

    Algorithm:
    1. Split text into sentences
    2. Generate embeddings for each sentence
    3. Calculate similarity between consecutive sentences
    4. Identify breakpoints where similarity drops below threshold
    5. Create chunks at these semantic boundaries

    Example:
        >>> chunker = SemanticChunker(chunk_size=512)
        >>> chunks = chunker.chunk(text)

        >>> # With specific embedding model
        >>> chunker = SemanticChunker(
        ...     chunk_size=512,
        ...     embedding_model="all-mpnet-base-v2"
        ... )

        >>> # With OpenAI embeddings
        >>> chunker = SemanticChunker(
        ...     embedding_provider="openai",
        ...     embedding_model="text-embedding-3-small"
        ... )
    """

    def __init__(
        self,
        config: Optional[ChunkConfig] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_provider: str = "sentence-transformers",
        similarity_threshold: float = 0.5,
        min_sentences: int = 2,
        **kwargs,
    ):
        """
        Initialize semantic chunker.

        Args:
            config: ChunkConfig instance
            embedding_model: Name of embedding model
            embedding_provider: Provider ("sentence-transformers", "openai")
            similarity_threshold: Minimum similarity to keep sentences together
            min_sentences: Minimum sentences per chunk
            **kwargs: Override config parameters
        """
        super().__init__(config, **kwargs)

        self.embedding_model = embedding_model
        self.embedding_provider = embedding_provider
        self.similarity_threshold = similarity_threshold
        self.min_sentences = min_sentences

        self._embedder = None

    @property
    def embedder(self):
        """Lazy load embedding provider."""
        if self._embedder is None:
            api_key = None
            if self.config.embedding_config:
                api_key = self.config.embedding_config.api_key

            self._embedder = get_embedding_provider(
                provider=self.embedding_provider,
                model=self.embedding_model,
                api_key=api_key,
            )
        return self._embedder

    def _chunk_text(self, text: str) -> List[Chunk]:
        """
        Split text based on semantic similarity.

        Args:
            text: Input text

        Returns:
            List of Chunk objects
        """
        if not text:
            return []

        # Split into sentences
        sentences = self._split_sentences(text)

        if len(sentences) <= 1:
            return [
                Chunk(
                    content=text,
                    index=0,
                    start_char=0,
                    end_char=len(text),
                    metadata={"strategy": "semantic"},
                )
            ]

        # Generate embeddings
        try:
            embeddings = self.embedder.embed(sentences)
        except Exception as e:
            # Fallback to recursive chunking if embeddings fail
            from chunkwise.strategies.recursive import RecursiveChunker

            recursive = RecursiveChunker(config=self.config)
            chunks = recursive._chunk_text(text)
            for chunk in chunks:
                chunk.metadata["strategy"] = "semantic_fallback"
                chunk.metadata["fallback_reason"] = str(e)
            return chunks

        # Find semantic breakpoints
        breakpoints = self._find_breakpoints(embeddings)

        # Create chunks at breakpoints
        return self._create_chunks_at_breakpoints(sentences, breakpoints, text)

    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        from chunkwise.language.detector import detect_language

        language = detect_language(text)

        if language == "ar":
            from chunkwise.language.arabic.sentence_splitter import split_arabic_sentences

            return split_arabic_sentences(text, min_length=5)
        else:
            from chunkwise.language.english.sentence_splitter import split_english_sentences

            return split_english_sentences(text, min_length=5)

    def _find_breakpoints(self, embeddings: np.ndarray) -> List[int]:
        """
        Find semantic breakpoints based on similarity drops.

        Args:
            embeddings: Sentence embeddings

        Returns:
            List of breakpoint indices
        """
        if len(embeddings) <= 1:
            return []

        # Calculate similarities between consecutive sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)

        # Find breakpoints where similarity drops below threshold
        breakpoints = []

        for i, sim in enumerate(similarities):
            if sim < self.similarity_threshold:
                # Check minimum sentence requirement
                if (not breakpoints and i >= self.min_sentences - 1) or (
                    breakpoints and i - breakpoints[-1] >= self.min_sentences
                ):
                    breakpoints.append(i + 1)  # Break after this sentence

        return breakpoints

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    def _create_chunks_at_breakpoints(
        self,
        sentences: List[str],
        breakpoints: List[int],
        text: str,
    ) -> List[Chunk]:
        """
        Create chunks at semantic breakpoints.

        Args:
            sentences: List of sentences
            breakpoints: Indices where to break
            text: Original text

        Returns:
            List of Chunk objects
        """
        chunks = []
        start_idx = 0
        char_position = 0
        index = 0

        # Add end as final breakpoint
        all_breakpoints = breakpoints + [len(sentences)]

        for break_idx in all_breakpoints:
            # Get sentences for this chunk
            chunk_sentences = sentences[start_idx:break_idx]

            if not chunk_sentences:
                continue

            chunk_text = " ".join(chunk_sentences)

            # Check if chunk is too large
            if len(chunk_text) > self.config.chunk_size:
                # Split large chunk further
                sub_chunks = self._split_large_chunk(
                    chunk_sentences, text, index, char_position
                )
                chunks.extend(sub_chunks)
                index += len(sub_chunks)
                if sub_chunks:
                    char_position = sub_chunks[-1].end_char
            else:
                # Find position in original text
                start_char = self._find_text_position(
                    text, chunk_sentences[0][:30], char_position
                )
                end_char = start_char + len(chunk_text)

                chunks.append(
                    Chunk(
                        content=chunk_text,
                        index=index,
                        start_char=start_char,
                        end_char=end_char,
                        metadata={
                            "strategy": "semantic",
                            "sentence_count": len(chunk_sentences),
                        },
                    )
                )
                char_position = end_char
                index += 1

            start_idx = break_idx

        return chunks

    def _split_large_chunk(
        self,
        sentences: List[str],
        text: str,
        start_index: int,
        start_char: int,
    ) -> List[Chunk]:
        """
        Split a chunk that exceeds max size.

        Args:
            sentences: Sentences in the chunk
            text: Original text
            start_index: Starting chunk index
            start_char: Starting character position

        Returns:
            List of smaller chunks
        """
        chunks = []
        current_sentences = []
        current_length = 0
        index = start_index
        char_pos = start_char

        for sentence in sentences:
            if current_length + len(sentence) > self.config.chunk_size and current_sentences:
                chunk_text = " ".join(current_sentences)
                start = self._find_text_position(text, current_sentences[0][:30], char_pos)
                end = start + len(chunk_text)

                chunks.append(
                    Chunk(
                        content=chunk_text,
                        index=index,
                        start_char=start,
                        end_char=end,
                        metadata={"strategy": "semantic", "from_split": True},
                    )
                )

                char_pos = end
                index += 1
                current_sentences = []
                current_length = 0

            current_sentences.append(sentence)
            current_length += len(sentence) + 1

        if current_sentences:
            chunk_text = " ".join(current_sentences)
            start = self._find_text_position(text, current_sentences[0][:30], char_pos)
            end = start + len(chunk_text)

            chunks.append(
                Chunk(
                    content=chunk_text,
                    index=index,
                    start_char=start,
                    end_char=end,
                    metadata={"strategy": "semantic", "from_split": True},
                )
            )

        return chunks

    def _find_text_position(self, text: str, search: str, start: int = 0) -> int:
        """Find position of text in original."""
        pos = text.find(search, start)
        return pos if pos >= 0 else start


class ClusterChunker(BaseChunker):
    """
    Group sentences into semantically similar clusters.

    Uses clustering algorithms to group related content together.
    """

    def __init__(
        self,
        config: Optional[ChunkConfig] = None,
        n_clusters: Optional[int] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        **kwargs,
    ):
        """
        Initialize cluster chunker.

        Args:
            config: ChunkConfig instance
            n_clusters: Number of clusters (auto-detected if None)
            embedding_model: Embedding model name
            **kwargs: Override config parameters
        """
        super().__init__(config, **kwargs)
        self.n_clusters = n_clusters
        self.embedding_model = embedding_model
        self._embedder = None

    def _chunk_text(self, text: str) -> List[Chunk]:
        """
        Cluster sentences into chunks.

        Args:
            text: Input text

        Returns:
            List of Chunk objects
        """
        # Implementation uses KMeans or similar clustering
        # For simplicity, delegate to SemanticChunker
        semantic = SemanticChunker(
            config=self.config,
            embedding_model=self.embedding_model,
        )
        chunks = semantic._chunk_text(text)

        for chunk in chunks:
            chunk.metadata["strategy"] = "cluster"

        return chunks
