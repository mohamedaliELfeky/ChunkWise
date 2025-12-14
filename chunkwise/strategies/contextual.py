"""
Contextual Retrieval Chunking Strategy

Anthropic's Contextual Retrieval method that prepends context to each chunk
before embedding, dramatically improving retrieval accuracy.

Reference: https://www.anthropic.com/news/contextual-retrieval
"""

from typing import List, Optional, Callable
import asyncio

from chunkwise.base import BaseChunker
from chunkwise.chunk import Chunk
from chunkwise.config import ChunkConfig
from chunkwise.llm.base import get_llm_provider
from chunkwise.exceptions import LLMError


class ContextualChunker(BaseChunker):
    """
    Anthropic's Contextual Retrieval chunking strategy.

    This strategy prepends chunk-specific explanatory context to each chunk
    before embedding. This dramatically improves retrieval accuracy by
    providing context that would otherwise be lost when chunking.

    Performance (Anthropic benchmarks):
    - Contextual Embeddings: 35% reduction in retrieval failures
    - Contextual Embeddings + BM25: 49% reduction
    - With Reranking: 67% reduction in retrieval failures

    Example:
        >>> chunker = ContextualChunker(chunk_size=512)
        >>> chunks = chunker.chunk(document)
        >>> # Each chunk now has contextualized content

    How it works:
    1. First splits document using base strategy (recursive by default)
    2. For each chunk, uses LLM to generate contextual prefix
    3. Prepends context to chunk content
    """

    CONTEXT_PROMPT = """<document>
{document}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""

    def __init__(
        self,
        config: Optional[ChunkConfig] = None,
        base_strategy: str = "recursive",
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o-mini",
        context_max_tokens: int = 100,
        batch_size: int = 10,
        **kwargs,
    ):
        """
        Initialize Contextual chunker.

        Args:
            config: ChunkConfig instance
            base_strategy: Base chunking strategy to use first
            llm_provider: LLM provider for context generation
            llm_model: LLM model name
            context_max_tokens: Max tokens for context prefix
            batch_size: Number of chunks to process in parallel
            **kwargs: Override config parameters
        """
        super().__init__(config, **kwargs)
        self.base_strategy = base_strategy
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.context_max_tokens = context_max_tokens
        self.batch_size = batch_size
        self._llm = None
        self._base_chunker = None

    @property
    def llm(self):
        """Lazy load LLM provider."""
        if self._llm is None:
            api_key = None
            if self.config.llm_config:
                api_key = self.config.llm_config.api_key

            self._llm = get_llm_provider(
                provider=self.llm_provider,
                model=self.llm_model,
                api_key=api_key,
            )
        return self._llm

    @property
    def base_chunker(self):
        """Lazy load base chunker."""
        if self._base_chunker is None:
            from chunkwise.strategies.hybrid import HybridChunker

            hybrid = HybridChunker(config=self.config, strategies=[self.base_strategy])
            self._base_chunker = hybrid._get_chunker(self.base_strategy)
        return self._base_chunker

    def _chunk_text(self, text: str) -> List[Chunk]:
        """
        Chunk text with contextual prefixes.

        Args:
            text: Input text (full document)

        Returns:
            List of contextualized Chunk objects
        """
        if not text:
            return []

        # Step 1: Split using base strategy
        base_chunks = self.base_chunker._chunk_text(text)

        if not base_chunks:
            return []

        # Step 2: Generate context for each chunk
        try:
            contextualized_chunks = self._add_context_to_chunks(base_chunks, text)
            return contextualized_chunks
        except Exception as e:
            # Fallback: return base chunks without context
            for chunk in base_chunks:
                chunk.metadata["strategy"] = "contextual_fallback"
                chunk.metadata["context_error"] = str(e)
            return base_chunks

    def _add_context_to_chunks(
        self, chunks: List[Chunk], document: str
    ) -> List[Chunk]:
        """
        Add contextual prefixes to chunks.

        Args:
            chunks: Base chunks
            document: Full document text

        Returns:
            Chunks with context prepended
        """
        # Truncate document if too long for prompt
        max_doc_length = 8000  # Keep prompt reasonable
        doc_for_prompt = document[:max_doc_length]
        if len(document) > max_doc_length:
            doc_for_prompt += "\n[Document truncated...]"

        contextualized = []

        for i, chunk in enumerate(chunks):
            # Generate context
            context = self._generate_context(doc_for_prompt, chunk.content)

            # Create contextualized content
            if context:
                contextualized_content = f"{context}\n\n{chunk.content}"
            else:
                contextualized_content = chunk.content

            contextualized.append(
                Chunk(
                    content=contextualized_content,
                    index=i,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    metadata={
                        **chunk.metadata,
                        "strategy": "contextual",
                        "original_content": chunk.content,
                        "context_prefix": context,
                        "has_context": bool(context),
                    },
                )
            )

        return contextualized

    def _generate_context(self, document: str, chunk: str) -> str:
        """
        Generate contextual prefix for a chunk.

        Args:
            document: Full document text
            chunk: Chunk content

        Returns:
            Context string to prepend
        """
        prompt = self.CONTEXT_PROMPT.format(
            document=document,
            chunk=chunk,
        )

        try:
            context = self.llm.generate(prompt)
            # Clean up context
            context = context.strip()
            # Limit length
            if len(context) > 500:
                context = context[:500] + "..."
            return context
        except Exception:
            return ""

    async def _chunk_text_async(self, text: str) -> List[Chunk]:
        """
        Async version with parallel context generation.

        Args:
            text: Input text

        Returns:
            List of contextualized chunks
        """
        if not text:
            return []

        # Step 1: Split using base strategy
        base_chunks = self.base_chunker._chunk_text(text)

        if not base_chunks:
            return []

        # Truncate document
        max_doc_length = 8000
        doc_for_prompt = text[:max_doc_length]
        if len(text) > max_doc_length:
            doc_for_prompt += "\n[Document truncated...]"

        # Step 2: Generate contexts in parallel batches
        contextualized = []

        for batch_start in range(0, len(base_chunks), self.batch_size):
            batch = base_chunks[batch_start : batch_start + self.batch_size]

            # Generate contexts in parallel
            tasks = [
                self._generate_context_async(doc_for_prompt, chunk.content)
                for chunk in batch
            ]
            contexts = await asyncio.gather(*tasks)

            # Create contextualized chunks
            for chunk, context in zip(batch, contexts):
                if context:
                    content = f"{context}\n\n{chunk.content}"
                else:
                    content = chunk.content

                contextualized.append(
                    Chunk(
                        content=content,
                        index=len(contextualized),
                        start_char=chunk.start_char,
                        end_char=chunk.end_char,
                        metadata={
                            **chunk.metadata,
                            "strategy": "contextual",
                            "original_content": chunk.content,
                            "context_prefix": context,
                            "has_context": bool(context),
                        },
                    )
                )

        return contextualized

    async def _generate_context_async(self, document: str, chunk: str) -> str:
        """Async context generation."""
        prompt = self.CONTEXT_PROMPT.format(document=document, chunk=chunk)
        try:
            context = await self.llm.generate_async(prompt)
            context = context.strip()
            if len(context) > 500:
                context = context[:500] + "..."
            return context
        except Exception:
            return ""


class ContextualBM25Chunker(ContextualChunker):
    """
    Contextual Retrieval with BM25 support.

    Combines contextual embeddings with BM25 lexical search for
    best retrieval performance (49% improvement per Anthropic).

    The chunks are prepared for both:
    1. Semantic search (via embeddings)
    2. Lexical search (via BM25)

    Example:
        >>> chunker = ContextualBM25Chunker(chunk_size=512)
        >>> chunks = chunker.chunk(document)
        >>> # Use chunks for both embedding and BM25 indexing
    """

    def __init__(
        self,
        config: Optional[ChunkConfig] = None,
        include_bm25_tokens: bool = True,
        **kwargs,
    ):
        """
        Initialize Contextual BM25 chunker.

        Args:
            config: ChunkConfig instance
            include_bm25_tokens: Include tokenized content for BM25
            **kwargs: Additional parameters
        """
        super().__init__(config, **kwargs)
        self.include_bm25_tokens = include_bm25_tokens

    def _chunk_text(self, text: str) -> List[Chunk]:
        """
        Chunk with context and BM25 preparation.

        Args:
            text: Input text

        Returns:
            List of chunks ready for embedding + BM25
        """
        chunks = super()._chunk_text(text)

        if self.include_bm25_tokens:
            for chunk in chunks:
                # Add tokenized version for BM25
                tokens = self._tokenize_for_bm25(chunk.content)
                chunk.metadata["bm25_tokens"] = tokens

        return chunks

    def _tokenize_for_bm25(self, text: str) -> List[str]:
        """
        Tokenize text for BM25 indexing.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        import re

        # Simple tokenization (can be improved with stemming, etc.)
        text = text.lower()
        # Remove punctuation and split
        tokens = re.findall(r'\b\w+\b', text)
        # Remove very short tokens
        tokens = [t for t in tokens if len(t) > 2]
        return tokens
