"""
Chunking Strategies Module

All available text chunking strategies - 30+ strategies for comprehensive RAG support.
"""

# Basic Strategies
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

# Document Structure
from chunkwise.strategies.document_structure import (
    MarkdownChunker,
    HTMLChunker,
    CodeChunker,
)

# Format-Specific
from chunkwise.strategies.format_specific import (
    JSONChunker,
    LaTeXChunker,
    RegexChunker,
)

# Semantic & Embedding-Based
from chunkwise.strategies.semantic import SemanticChunker, ClusterChunker
from chunkwise.strategies.late import LateChunker

# LLM-Based
from chunkwise.strategies.agentic import AgenticChunker, PropositionChunker
from chunkwise.strategies.document_summary import (
    DocumentSummaryChunker,
    KeywordSummaryChunker,
)
from chunkwise.strategies.contextual import (
    ContextualChunker,
    ContextualBM25Chunker,
)

# Retrieval-Optimized
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

# Hybrid/Combined
from chunkwise.strategies.hybrid import HybridChunker, AdaptiveChunker

__all__ = [
    # === Basic Strategies ===
    # Fixed-size
    "CharacterChunker",
    "TokenChunker",
    "WordChunker",
    # Sentence-based
    "SentenceChunker",
    "MultiSentenceChunker",
    # Paragraph
    "ParagraphChunker",
    # Recursive
    "RecursiveChunker",
    # Sliding window
    "SlidingWindowChunker",

    # === Document Structure ===
    "MarkdownChunker",
    "HTMLChunker",
    "CodeChunker",

    # === Format-Specific ===
    "JSONChunker",
    "LaTeXChunker",
    "RegexChunker",

    # === Semantic & Embedding-Based ===
    "SemanticChunker",
    "ClusterChunker",
    "LateChunker",

    # === LLM-Based ===
    "AgenticChunker",
    "PropositionChunker",
    "DocumentSummaryChunker",
    "KeywordSummaryChunker",
    "ContextualChunker",
    "ContextualBM25Chunker",

    # === Retrieval-Optimized ===
    "SentenceWindowChunker",
    "AutoMergingChunker",
    "ParentDocumentChunker",
    "SmallToBigChunker",
    "BigToSmallChunker",
    "HierarchicalChunker",

    # === Hybrid/Combined ===
    "HybridChunker",
    "AdaptiveChunker",
]
