"""
Late Chunking Strategy

Embed full document first, then chunk while preserving global context.
"""

from typing import List, Optional
import numpy as np

from chunkwise.base import BaseChunker
from chunkwise.chunk import Chunk
from chunkwise.config import ChunkConfig
from chunkwise.embeddings.base import get_embedding_provider


class LateChunker(BaseChunker):
    """
    Late chunking strategy that embeds the full document first.

    Unlike traditional chunking which splits first then embeds each chunk,
    late chunking embeds the entire document using a long-context model,
    then creates chunks that retain global context information.

    Benefits:
    - Chunks maintain awareness of full document context
    - Better retrieval performance for questions requiring context
    - Reduced information loss at chunk boundaries

    Example:
        >>> chunker = LateChunker(chunk_size=512)
        >>> chunks = chunker.chunk(text)

        >>> # Access embeddings
        >>> for chunk in chunks:
        ...     print(chunk.metadata.get("embedding"))
    """

    def __init__(
        self,
        config: Optional[ChunkConfig] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_provider: str = "sentence-transformers",
        compute_chunk_embeddings: bool = True,
        **kwargs,
    ):
        """
        Initialize late chunker.

        Args:
            config: ChunkConfig instance
            embedding_model: Embedding model name
            embedding_provider: Embedding provider
            compute_chunk_embeddings: Whether to compute embeddings for chunks
            **kwargs: Override config parameters
        """
        super().__init__(config, **kwargs)
        self.embedding_model = embedding_model
        self.embedding_provider = embedding_provider
        self.compute_chunk_embeddings = compute_chunk_embeddings
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
        Perform late chunking.

        Args:
            text: Input text

        Returns:
            List of Chunk objects with contextual embeddings
        """
        if not text:
            return []

        # Step 1: Get full document embedding
        try:
            document_embedding = self.embedder.embed_single(text)
        except Exception:
            document_embedding = None

        # Step 2: Split into sentences
        sentences = self._split_into_sentences(text)

        if not sentences:
            return [
                Chunk(
                    content=text,
                    index=0,
                    start_char=0,
                    end_char=len(text),
                    metadata={
                        "strategy": "late",
                        "document_embedding": (
                            document_embedding.tolist()
                            if document_embedding is not None
                            else None
                        ),
                    },
                )
            ]

        # Step 3: Get sentence embeddings
        try:
            sentence_embeddings = self.embedder.embed(sentences)
        except Exception:
            sentence_embeddings = None

        # Step 4: Group sentences into chunks
        chunks = self._group_sentences_with_context(
            sentences,
            sentence_embeddings,
            document_embedding,
            text,
        )

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        from chunkwise.language.detector import detect_language

        lang = detect_language(text)

        if lang == "ar":
            from chunkwise.language.arabic.sentence_splitter import split_arabic_sentences

            return split_arabic_sentences(text, min_length=5)
        else:
            from chunkwise.language.english.sentence_splitter import split_english_sentences

            return split_english_sentences(text, min_length=5)

    def _group_sentences_with_context(
        self,
        sentences: List[str],
        sentence_embeddings: Optional[np.ndarray],
        document_embedding: Optional[np.ndarray],
        text: str,
    ) -> List[Chunk]:
        """
        Group sentences into chunks while preserving context.

        Args:
            sentences: List of sentences
            sentence_embeddings: Embeddings for each sentence
            document_embedding: Full document embedding
            text: Original text

        Returns:
            List of Chunk objects
        """
        chunks = []
        current_sentences = []
        current_length = 0
        index = 0
        char_position = 0

        for i, sentence in enumerate(sentences):
            sentence_length = len(sentence)

            # Check if adding this sentence exceeds limit
            if current_length + sentence_length > self.config.chunk_size and current_sentences:
                # Create chunk
                chunk = self._create_contextual_chunk(
                    current_sentences,
                    sentence_embeddings[max(0, i - len(current_sentences)) : i]
                    if sentence_embeddings is not None
                    else None,
                    document_embedding,
                    text,
                    index,
                    char_position,
                )
                chunks.append(chunk)
                char_position = chunk.end_char
                index += 1

                # Handle overlap
                if self.config.chunk_overlap > 0:
                    overlap = self._get_overlap_sentences(
                        current_sentences, self.config.chunk_overlap
                    )
                    current_sentences = overlap
                    current_length = sum(len(s) for s in overlap)
                else:
                    current_sentences = []
                    current_length = 0

            current_sentences.append(sentence)
            current_length += sentence_length + 1

        # Last chunk
        if current_sentences:
            start_idx = len(sentences) - len(current_sentences)
            chunk = self._create_contextual_chunk(
                current_sentences,
                sentence_embeddings[start_idx:] if sentence_embeddings is not None else None,
                document_embedding,
                text,
                index,
                char_position,
            )
            chunks.append(chunk)

        return chunks

    def _create_contextual_chunk(
        self,
        sentences: List[str],
        sentence_embeddings: Optional[np.ndarray],
        document_embedding: Optional[np.ndarray],
        text: str,
        index: int,
        start_search: int,
    ) -> Chunk:
        """
        Create a chunk with contextual embedding.

        The chunk embedding combines sentence embeddings with document context.

        Args:
            sentences: Sentences in chunk
            sentence_embeddings: Embeddings for sentences
            document_embedding: Full document embedding
            text: Original text
            index: Chunk index
            start_search: Position to start searching

        Returns:
            Chunk object
        """
        chunk_text = " ".join(sentences)

        # Find position
        start_char = text.find(sentences[0][:30], start_search)
        if start_char < 0:
            start_char = start_search
        end_char = start_char + len(chunk_text)

        # Compute contextual embedding
        chunk_embedding = None
        if self.compute_chunk_embeddings:
            if sentence_embeddings is not None and len(sentence_embeddings) > 0:
                # Average sentence embeddings
                avg_embedding = np.mean(sentence_embeddings, axis=0)

                # Blend with document embedding for context
                if document_embedding is not None:
                    # 70% chunk, 30% document context
                    chunk_embedding = 0.7 * avg_embedding + 0.3 * document_embedding
                    # Normalize
                    chunk_embedding = chunk_embedding / np.linalg.norm(chunk_embedding)
                else:
                    chunk_embedding = avg_embedding
            elif document_embedding is not None:
                chunk_embedding = document_embedding

        metadata = {
            "strategy": "late",
            "sentence_count": len(sentences),
        }

        if chunk_embedding is not None:
            metadata["embedding"] = chunk_embedding.tolist()

        return Chunk(
            content=chunk_text,
            index=index,
            start_char=start_char,
            end_char=end_char,
            metadata=metadata,
        )

    def _get_overlap_sentences(
        self, sentences: List[str], target_overlap: int
    ) -> List[str]:
        """Get sentences for overlap."""
        overlap = []
        current = 0

        for s in reversed(sentences):
            if current + len(s) <= target_overlap:
                overlap.insert(0, s)
                current += len(s) + 1
            else:
                break

        return overlap
