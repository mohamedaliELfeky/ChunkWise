"""
ChunkWise Exceptions

Custom exception classes for the ChunkWise library.
"""


class ChunkWiseError(Exception):
    """Base exception for all ChunkWise errors."""

    def __init__(self, message: str = "An error occurred in ChunkWise"):
        self.message = message
        super().__init__(self.message)


class ChunkSizeError(ChunkWiseError):
    """Raised when chunk size configuration is invalid."""

    def __init__(self, message: str = "Invalid chunk size configuration"):
        super().__init__(message)


class LanguageError(ChunkWiseError):
    """Raised when language detection or processing fails."""

    def __init__(self, message: str = "Language detection or processing error"):
        super().__init__(message)


class TokenizerError(ChunkWiseError):
    """Raised when tokenization fails."""

    def __init__(self, message: str = "Tokenization error"):
        super().__init__(message)


class EmbeddingError(ChunkWiseError):
    """Raised when embedding generation fails."""

    def __init__(self, message: str = "Embedding generation error"):
        super().__init__(message)


class LLMError(ChunkWiseError):
    """Raised when LLM-based operations fail."""

    def __init__(self, message: str = "LLM operation error"):
        super().__init__(message)


class ConfigurationError(ChunkWiseError):
    """Raised when configuration is invalid."""

    def __init__(self, message: str = "Invalid configuration"):
        super().__init__(message)


class StrategyError(ChunkWiseError):
    """Raised when a chunking strategy encounters an error."""

    def __init__(self, message: str = "Chunking strategy error"):
        super().__init__(message)


class DocumentParsingError(ChunkWiseError):
    """Raised when document parsing fails."""

    def __init__(self, message: str = "Document parsing error"):
        super().__init__(message)
