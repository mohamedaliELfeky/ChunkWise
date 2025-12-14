"""
ChunkWise - Comprehensive Text Chunking Library for Arabic and English

A production-ready library implementing 30+ text chunking strategies
with first-class support for both Arabic and English languages.

Usage:
    from chunkwise import Chunker, ChunkConfig

    # Simple usage
    chunker = Chunker(strategy="recursive", chunk_size=512)
    chunks = chunker.chunk(text)

    # Arabic-specific
    chunker = Chunker(strategy="sentence", language="ar")
    chunks = chunker.chunk(arabic_text)

    # Semantic chunking
    chunker = Chunker(strategy="semantic", embedding_model="all-MiniLM-L6-v2")
    chunks = chunker.chunk(text)

    # Contextual Retrieval (Anthropic method)
    chunker = ContextualChunker(llm_provider="anthropic")
    chunks = chunker.chunk(text)

    # Sentence Window (ARAGOG research)
    chunker = SentenceWindowChunker(window_size=3)
    chunks = chunker.chunk(text)
"""

from chunkwise.base import BaseChunker
from chunkwise.chunk import Chunk
from chunkwise.config import ChunkConfig, ArabicConfig
from chunkwise.exceptions import (
    ChunkWiseError,
    ChunkSizeError,
    LanguageError,
    TokenizerError,
    EmbeddingError,
    LLMError,
)

# === Basic Strategies ===
from chunkwise.strategies.fixed import (
    CharacterChunker,
    TokenChunker,
    WordChunker,
)
from chunkwise.strategies.sentence import (
    SentenceChunker,
    MultiSentenceChunker,
)
from chunkwise.strategies.paragraph import ParagraphChunker
from chunkwise.strategies.recursive import RecursiveChunker
from chunkwise.strategies.sliding_window import SlidingWindowChunker

# === Document Structure ===
from chunkwise.strategies.document_structure import (
    MarkdownChunker,
    HTMLChunker,
    CodeChunker,
)

# === Format-Specific ===
from chunkwise.strategies.format_specific import (
    JSONChunker,
    LaTeXChunker,
    RegexChunker,
)

# === Semantic & Embedding-Based ===
from chunkwise.strategies.semantic import SemanticChunker, ClusterChunker
from chunkwise.strategies.late import LateChunker

# === LLM-Based ===
from chunkwise.strategies.agentic import AgenticChunker, PropositionChunker
from chunkwise.strategies.document_summary import (
    DocumentSummaryChunker,
    KeywordSummaryChunker,
)
from chunkwise.strategies.contextual import (
    ContextualChunker,
    ContextualBM25Chunker,
)

# === Retrieval-Optimized ===
from chunkwise.strategies.sentence_window import (
    SentenceWindowChunker,
    AutoMergingChunker,
)
from chunkwise.strategies.parent_document import (
    ParentDocumentChunker,
    SmallToBigChunker,
    BigToSmallChunker,
)
from chunkwise.strategies.hierarchical import HierarchicalChunker

# === Hybrid/Combined ===
from chunkwise.strategies.hybrid import HybridChunker, AdaptiveChunker

# Main entry point
from chunkwise.chunker import Chunker, chunk_text, chunk_arabic

__version__ = "1.0.0"
__author__ = "Hesham Haroon"
__email__ = "heshamharoon9@gmail.com"

__all__ = [
    # Main classes
    "Chunker",
    "chunk_text",
    "chunk_arabic",
    "BaseChunker",
    "Chunk",
    "ChunkConfig",
    "ArabicConfig",
    # Exceptions
    "ChunkWiseError",
    "ChunkSizeError",
    "LanguageError",
    "TokenizerError",
    "EmbeddingError",
    "LLMError",

    # === Basic Strategies (9) ===
    "CharacterChunker",
    "TokenChunker",
    "WordChunker",
    "SentenceChunker",
    "MultiSentenceChunker",
    "ParagraphChunker",
    "RecursiveChunker",
    "SlidingWindowChunker",

    # === Document Structure (3) ===
    "MarkdownChunker",
    "HTMLChunker",
    "CodeChunker",

    # === Format-Specific (3) ===
    "JSONChunker",
    "LaTeXChunker",
    "RegexChunker",

    # === Semantic & Embedding-Based (3) ===
    "SemanticChunker",
    "ClusterChunker",
    "LateChunker",

    # === LLM-Based (6) ===
    "AgenticChunker",
    "PropositionChunker",
    "DocumentSummaryChunker",
    "KeywordSummaryChunker",
    "ContextualChunker",
    "ContextualBM25Chunker",

    # === Retrieval-Optimized (6) ===
    "SentenceWindowChunker",
    "AutoMergingChunker",
    "ParentDocumentChunker",
    "SmallToBigChunker",
    "BigToSmallChunker",
    "HierarchicalChunker",

    # === Hybrid/Combined (2) ===
    "HybridChunker",
    "AdaptiveChunker",
]
