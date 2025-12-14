"""
ChunkWise Configuration

Configuration classes for customizing chunking behavior.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Literal, Union
from enum import Enum


class Strategy(str, Enum):
    """Available chunking strategies - 30+ strategies for comprehensive RAG support."""

    # === Basic Strategies ===
    # Fixed-size strategies
    CHARACTER = "character"
    TOKEN = "token"
    WORD = "word"

    # Sentence-based
    SENTENCE = "sentence"
    MULTI_SENTENCE = "multi_sentence"

    # Paragraph-based
    PARAGRAPH = "paragraph"

    # Hierarchical
    RECURSIVE = "recursive"

    # Sliding window
    SLIDING_WINDOW = "sliding_window"

    # === Document Structure ===
    MARKDOWN = "markdown"
    HTML = "html"
    CODE = "code"

    # === Format-Specific ===
    JSON = "json"
    LATEX = "latex"
    REGEX = "regex"

    # === Semantic & Embedding-Based ===
    SEMANTIC = "semantic"
    CLUSTER = "cluster"
    LATE = "late"

    # === LLM-Based ===
    AGENTIC = "agentic"
    PROPOSITION = "proposition"
    DOCUMENT_SUMMARY = "document_summary"
    KEYWORD_SUMMARY = "keyword_summary"
    CONTEXTUAL = "contextual"
    CONTEXTUAL_BM25 = "contextual_bm25"

    # === Retrieval-Optimized ===
    SENTENCE_WINDOW = "sentence_window"
    AUTO_MERGING = "auto_merging"
    PARENT_DOCUMENT = "parent_document"
    SMALL_TO_BIG = "small_to_big"
    BIG_TO_SMALL = "big_to_small"
    HIERARCHICAL = "hierarchical"

    # === Hybrid/Combined ===
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"


class Language(str, Enum):
    """Supported languages."""

    AUTO = "auto"
    ENGLISH = "en"
    ARABIC = "ar"
    MULTILINGUAL = "multi"


@dataclass
class ArabicConfig:
    """
    Configuration specific to Arabic text processing.

    Attributes:
        normalize_alef: Normalize Alef variants (أ إ آ) to bare Alef (ا)
        normalize_yaa: Normalize Yaa variants (ي ى) to standard form
        normalize_taa_marbuta: Normalize Taa Marbuta (ة) to Haa (ه)
        remove_diacritics: Remove Arabic diacritics/tashkeel (ً ٌ ٍ َ ُ ِ ّ ْ)
        remove_tatweel: Remove Tatweel/Kashida (ـ)
        use_camel_tokenizer: Use CAMeL Tools for morphological tokenization
        sentence_end_markers: Custom sentence-ending punctuation marks
    """

    normalize_alef: bool = True
    normalize_yaa: bool = True
    normalize_taa_marbuta: bool = False
    remove_diacritics: bool = True
    remove_tatweel: bool = True
    use_camel_tokenizer: bool = False
    sentence_end_markers: List[str] = field(
        default_factory=lambda: [".", "؟", "!", "؛", "۔", "。"]
    )


@dataclass
class EnglishConfig:
    """
    Configuration specific to English text processing.

    Attributes:
        use_spacy: Use spaCy for sentence detection
        sentence_end_markers: Custom sentence-ending punctuation marks
        preserve_case: Preserve original case (don't lowercase)
    """

    use_spacy: bool = False
    sentence_end_markers: List[str] = field(default_factory=lambda: [".", "?", "!"])
    preserve_case: bool = True


@dataclass
class EmbeddingConfig:
    """
    Configuration for embedding-based operations.

    Attributes:
        model: Embedding model name or path
        provider: Embedding provider (sentence-transformers, openai, cohere)
        api_key: API key for cloud providers
        batch_size: Batch size for embedding generation
        normalize: Normalize embeddings to unit length
    """

    model: str = "all-MiniLM-L6-v2"
    provider: Literal["sentence-transformers", "openai", "cohere"] = "sentence-transformers"
    api_key: Optional[str] = None
    batch_size: int = 32
    normalize: bool = True


@dataclass
class LLMConfig:
    """
    Configuration for LLM-based operations.

    Attributes:
        model: LLM model name
        provider: LLM provider (openai, anthropic, ollama)
        api_key: API key for cloud providers
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        base_url: Base URL for API (useful for Ollama)
    """

    model: str = "gpt-4o-mini"
    provider: Literal["openai", "anthropic", "ollama"] = "openai"
    api_key: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 4096
    base_url: Optional[str] = None


@dataclass
class ChunkConfig:
    """
    Main configuration for chunking operations.

    Attributes:
        strategy: Chunking strategy to use
        chunk_size: Target chunk size (interpretation depends on strategy)
        chunk_overlap: Number of overlapping units between chunks
        language: Language setting (auto-detect or specific)
        min_chunk_size: Minimum chunk size (chunks smaller than this may be merged)
        max_chunk_size: Maximum chunk size (hard limit)

        # Strategy-specific settings
        separators: List of separators for recursive chunking
        sentences_per_chunk: Number of sentences per chunk (for sentence strategies)

        # Language-specific configs
        arabic_config: Arabic-specific configuration
        english_config: English-specific configuration

        # Advanced configs
        embedding_config: Configuration for semantic chunking
        llm_config: Configuration for agentic chunking

        # Metadata
        include_metadata: Include metadata in chunks
        compute_tokens: Compute token counts for chunks
    """

    # Core settings
    strategy: Union[Strategy, str] = Strategy.RECURSIVE
    chunk_size: int = 512
    chunk_overlap: int = 50
    language: Union[Language, str] = Language.AUTO

    # Size constraints
    min_chunk_size: int = 100
    max_chunk_size: Optional[int] = None

    # Recursive chunking separators (language-aware defaults set in __post_init__)
    separators: Optional[List[str]] = None

    # Sentence chunking
    sentences_per_chunk: int = 3

    # Language configs
    arabic_config: ArabicConfig = field(default_factory=ArabicConfig)
    english_config: EnglishConfig = field(default_factory=EnglishConfig)

    # Advanced configs
    embedding_config: Optional[EmbeddingConfig] = None
    llm_config: Optional[LLMConfig] = None

    # Metadata options
    include_metadata: bool = True
    compute_tokens: bool = True

    # Tokenizer settings
    tokenizer: str = "tiktoken"  # tiktoken, simple, arabic, huggingface
    tokenizer_model: str = "cl100k_base"  # For tiktoken

    def __post_init__(self):
        """Set defaults based on language and strategy."""
        # Convert string to enum if needed
        if isinstance(self.strategy, str):
            self.strategy = Strategy(self.strategy)
        if isinstance(self.language, str):
            self.language = Language(self.language)

        # Set default separators based on language
        if self.separators is None:
            if self.language == Language.ARABIC:
                self.separators = self._arabic_separators()
            elif self.language == Language.ENGLISH:
                self.separators = self._english_separators()
            else:
                self.separators = self._multilingual_separators()

        # Set max chunk size if not specified
        if self.max_chunk_size is None:
            self.max_chunk_size = self.chunk_size * 2

        # Initialize embedding config for semantic strategy
        if self.strategy == Strategy.SEMANTIC and self.embedding_config is None:
            self.embedding_config = EmbeddingConfig()

        # Initialize LLM config for agentic strategy
        if self.strategy in (Strategy.AGENTIC, Strategy.PROPOSITION) and self.llm_config is None:
            self.llm_config = LLMConfig()

    @staticmethod
    def _english_separators() -> List[str]:
        """Default separators for English text."""
        return [
            "\n\n",  # Paragraphs
            "\n",  # Lines
            ". ",  # Sentences
            "? ",
            "! ",
            "; ",  # Clauses
            ", ",  # Phrases
            " ",  # Words
            "",  # Characters (fallback)
        ]

    @staticmethod
    def _arabic_separators() -> List[str]:
        """Default separators for Arabic text."""
        return [
            "\n\n",  # Paragraphs
            "\n",  # Lines
            "۔ ",  # Arabic full stop
            "؟ ",  # Arabic question mark
            "! ",  # Exclamation
            "؛ ",  # Arabic semicolon
            "، ",  # Arabic comma
            ". ",  # Western period (common in Arabic)
            " ",  # Words
            "",  # Characters (fallback)
        ]

    @staticmethod
    def _multilingual_separators() -> List[str]:
        """Default separators for mixed/multilingual text."""
        return [
            "\n\n",
            "\n",
            "۔ ",
            "؟ ",
            "؛ ",
            "، ",
            ". ",
            "? ",
            "! ",
            "; ",
            ", ",
            " ",
            "",
        ]

    def for_arabic(self) -> "ChunkConfig":
        """Return a copy configured for Arabic text."""
        import copy

        config = copy.deepcopy(self)
        config.language = Language.ARABIC
        config.separators = self._arabic_separators()
        return config

    def for_english(self) -> "ChunkConfig":
        """Return a copy configured for English text."""
        import copy

        config = copy.deepcopy(self)
        config.language = Language.ENGLISH
        config.separators = self._english_separators()
        return config

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "strategy": self.strategy.value if isinstance(self.strategy, Strategy) else self.strategy,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "language": self.language.value if isinstance(self.language, Language) else self.language,
            "min_chunk_size": self.min_chunk_size,
            "max_chunk_size": self.max_chunk_size,
            "separators": self.separators,
            "sentences_per_chunk": self.sentences_per_chunk,
        }
