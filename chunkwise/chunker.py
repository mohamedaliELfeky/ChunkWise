"""
Main Chunker Entry Point

Provides a simple, unified interface to all chunking strategies.
"""

from typing import List, Optional, Union

from chunkwise.base import BaseChunker
from chunkwise.chunk import Chunk, ChunkBatch
from chunkwise.config import ChunkConfig, Strategy, Language, ArabicConfig


class Chunker:
    """
    Main entry point for ChunkWise library.

    Provides a simple, unified interface to all chunking strategies.
    Automatically selects the appropriate chunker based on configuration.

    Example:
        >>> from chunkwise import Chunker

        >>> # Simple usage with defaults (recursive strategy)
        >>> chunker = Chunker(chunk_size=512)
        >>> chunks = chunker.chunk("Your long text here...")

        >>> # Specify strategy
        >>> chunker = Chunker(strategy="sentence", chunk_size=1000)
        >>> chunks = chunker.chunk(text)

        >>> # Arabic text
        >>> chunker = Chunker(strategy="recursive", language="ar")
        >>> chunks = chunker.chunk(arabic_text)

        >>> # Semantic chunking with embeddings
        >>> chunker = Chunker(
        ...     strategy="semantic",
        ...     chunk_size=512,
        ...     embedding_model="all-MiniLM-L6-v2"
        ... )

        >>> # Advanced: using configuration object
        >>> config = ChunkConfig(
        ...     strategy="recursive",
        ...     chunk_size=512,
        ...     chunk_overlap=50,
        ...     language="auto",
        ...     arabic_config=ArabicConfig(remove_diacritics=True)
        ... )
        >>> chunker = Chunker(config=config)
    """

    # Strategy name to class mapping - 30+ strategies
    STRATEGY_MAP = {
        # === Basic Strategies ===
        Strategy.CHARACTER: "chunkwise.strategies.fixed.CharacterChunker",
        Strategy.TOKEN: "chunkwise.strategies.fixed.TokenChunker",
        Strategy.WORD: "chunkwise.strategies.fixed.WordChunker",
        Strategy.SENTENCE: "chunkwise.strategies.sentence.SentenceChunker",
        Strategy.MULTI_SENTENCE: "chunkwise.strategies.sentence.MultiSentenceChunker",
        Strategy.PARAGRAPH: "chunkwise.strategies.paragraph.ParagraphChunker",
        Strategy.RECURSIVE: "chunkwise.strategies.recursive.RecursiveChunker",
        Strategy.SLIDING_WINDOW: "chunkwise.strategies.sliding_window.SlidingWindowChunker",

        # === Document Structure ===
        Strategy.MARKDOWN: "chunkwise.strategies.document_structure.MarkdownChunker",
        Strategy.HTML: "chunkwise.strategies.document_structure.HTMLChunker",
        Strategy.CODE: "chunkwise.strategies.document_structure.CodeChunker",

        # === Format-Specific ===
        Strategy.JSON: "chunkwise.strategies.format_specific.JSONChunker",
        Strategy.LATEX: "chunkwise.strategies.format_specific.LaTeXChunker",
        Strategy.REGEX: "chunkwise.strategies.format_specific.RegexChunker",

        # === Semantic & Embedding-Based ===
        Strategy.SEMANTIC: "chunkwise.strategies.semantic.SemanticChunker",
        Strategy.CLUSTER: "chunkwise.strategies.semantic.ClusterChunker",
        Strategy.LATE: "chunkwise.strategies.late.LateChunker",

        # === LLM-Based ===
        Strategy.AGENTIC: "chunkwise.strategies.agentic.AgenticChunker",
        Strategy.PROPOSITION: "chunkwise.strategies.agentic.PropositionChunker",
        Strategy.DOCUMENT_SUMMARY: "chunkwise.strategies.document_summary.DocumentSummaryChunker",
        Strategy.KEYWORD_SUMMARY: "chunkwise.strategies.document_summary.KeywordSummaryChunker",
        Strategy.CONTEXTUAL: "chunkwise.strategies.contextual.ContextualChunker",
        Strategy.CONTEXTUAL_BM25: "chunkwise.strategies.contextual.ContextualBM25Chunker",

        # === Retrieval-Optimized ===
        Strategy.SENTENCE_WINDOW: "chunkwise.strategies.sentence_window.SentenceWindowChunker",
        Strategy.AUTO_MERGING: "chunkwise.strategies.sentence_window.AutoMergingChunker",
        Strategy.PARENT_DOCUMENT: "chunkwise.strategies.parent_document.ParentDocumentChunker",
        Strategy.SMALL_TO_BIG: "chunkwise.strategies.parent_document.SmallToBigChunker",
        Strategy.BIG_TO_SMALL: "chunkwise.strategies.parent_document.BigToSmallChunker",
        Strategy.HIERARCHICAL: "chunkwise.strategies.hierarchical.HierarchicalChunker",

        # === Hybrid/Combined ===
        Strategy.HYBRID: "chunkwise.strategies.hybrid.HybridChunker",
        Strategy.ADAPTIVE: "chunkwise.strategies.hybrid.AdaptiveChunker",
    }

    def __init__(
        self,
        config: Optional[ChunkConfig] = None,
        strategy: Union[Strategy, str] = "recursive",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        language: Union[Language, str] = "auto",
        # Convenience parameters
        embedding_model: Optional[str] = None,
        llm_model: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize Chunker.

        Args:
            config: Full configuration object (overrides other parameters)
            strategy: Chunking strategy to use
            chunk_size: Target chunk size
            chunk_overlap: Overlap between chunks
            language: Language setting ("auto", "en", "ar")
            embedding_model: Model for semantic chunking
            llm_model: Model for agentic chunking
            **kwargs: Additional configuration parameters
        """
        # Build config
        if config is not None:
            self.config = config
        else:
            self.config = ChunkConfig(
                strategy=strategy,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                language=language,
                **kwargs,
            )

        # Store convenience parameters
        self._embedding_model = embedding_model
        self._llm_model = llm_model

        # Initialize chunker
        self._chunker: Optional[BaseChunker] = None

    @property
    def chunker(self) -> BaseChunker:
        """Lazy load the appropriate chunker."""
        if self._chunker is None:
            self._chunker = self._create_chunker()
        return self._chunker

    def _create_chunker(self) -> BaseChunker:
        """
        Create the appropriate chunker based on configuration.

        Returns:
            BaseChunker instance
        """
        strategy = self.config.strategy

        # Get strategy class path
        if isinstance(strategy, str):
            strategy = Strategy(strategy)

        class_path = self.STRATEGY_MAP.get(strategy)
        if not class_path:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Import and instantiate
        module_path, class_name = class_path.rsplit(".", 1)
        module = __import__(module_path, fromlist=[class_name])
        chunker_class = getattr(module, class_name)

        # Build kwargs for specific strategies
        extra_kwargs = {}

        if strategy == Strategy.SEMANTIC and self._embedding_model:
            extra_kwargs["embedding_model"] = self._embedding_model

        if strategy in (Strategy.AGENTIC, Strategy.PROPOSITION) and self._llm_model:
            extra_kwargs["llm_model"] = self._llm_model

        return chunker_class(config=self.config, **extra_kwargs)

    def chunk(self, text: str) -> List[Chunk]:
        """
        Chunk the input text.

        Args:
            text: Text to chunk

        Returns:
            List of Chunk objects

        Example:
            >>> chunker = Chunker(chunk_size=500)
            >>> chunks = chunker.chunk("Your text here...")
            >>> for chunk in chunks:
            ...     print(f"Chunk {chunk.index}: {len(chunk)} chars")
        """
        return self.chunker.chunk(text)

    async def chunk_async(self, text: str) -> List[Chunk]:
        """
        Asynchronously chunk the input text.

        Args:
            text: Text to chunk

        Returns:
            List of Chunk objects
        """
        return await self.chunker.chunk_async(text)

    def chunk_documents(self, documents: List[str]) -> List[ChunkBatch]:
        """
        Chunk multiple documents.

        Args:
            documents: List of document texts

        Returns:
            List of ChunkBatch objects
        """
        return self.chunker.chunk_documents(documents)

    async def chunk_documents_async(self, documents: List[str]) -> List[ChunkBatch]:
        """
        Asynchronously chunk multiple documents.

        Args:
            documents: List of document texts

        Returns:
            List of ChunkBatch objects
        """
        return await self.chunker.chunk_documents_async(documents)

    def __call__(self, text: str) -> List[Chunk]:
        """
        Allow chunker to be called directly.

        Example:
            >>> chunker = Chunker()
            >>> chunks = chunker("Your text here...")
        """
        return self.chunk(text)

    def __repr__(self) -> str:
        return (
            f"Chunker(strategy='{self.config.strategy.value}', "
            f"chunk_size={self.config.chunk_size}, "
            f"language='{self.config.language.value}')"
        )


# Convenience functions
def chunk_text(
    text: str,
    strategy: str = "recursive",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    language: str = "auto",
    **kwargs,
) -> List[Chunk]:
    """
    Convenience function to chunk text in one line.

    Args:
        text: Text to chunk
        strategy: Chunking strategy
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
        language: Language setting
        **kwargs: Additional parameters

    Returns:
        List of Chunk objects

    Example:
        >>> from chunkwise import chunk_text
        >>> chunks = chunk_text("Your text here...", chunk_size=500)
    """
    chunker = Chunker(
        strategy=strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        language=language,
        **kwargs,
    )
    return chunker.chunk(text)


def chunk_arabic(
    text: str,
    strategy: str = "recursive",
    chunk_size: int = 512,
    normalize: bool = True,
    remove_diacritics: bool = True,
    **kwargs,
) -> List[Chunk]:
    """
    Convenience function for Arabic text chunking.

    Args:
        text: Arabic text to chunk
        strategy: Chunking strategy
        chunk_size: Target chunk size
        normalize: Normalize Arabic characters
        remove_diacritics: Remove Arabic diacritics
        **kwargs: Additional parameters

    Returns:
        List of Chunk objects

    Example:
        >>> from chunkwise import chunk_arabic
        >>> chunks = chunk_arabic("النص العربي هنا...")
    """
    arabic_config = ArabicConfig(
        normalize_alef=normalize,
        normalize_yaa=normalize,
        remove_diacritics=remove_diacritics,
    )

    chunker = Chunker(
        strategy=strategy,
        chunk_size=chunk_size,
        language="ar",
        arabic_config=arabic_config,
        **kwargs,
    )
    return chunker.chunk(text)
