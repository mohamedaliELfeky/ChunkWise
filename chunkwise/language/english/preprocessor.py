"""
English Text Preprocessor

Functions for normalizing and cleaning English text.
"""

import re
from typing import Optional
from dataclasses import dataclass


# Unicode quotes and dashes
UNICODE_QUOTES = {
    """: '"',
    """: '"',
    "'": "'",
    "'": "'",
    "„": '"',
    "«": '"',
    "»": '"',
}

UNICODE_DASHES = {
    "–": "-",  # En dash
    "—": "-",  # Em dash
    "−": "-",  # Minus sign
}

# Common contractions
CONTRACTIONS = {
    "won't": "will not",
    "can't": "cannot",
    "n't": " not",
    "'re": " are",
    "'s": " is",
    "'d": " would",
    "'ll": " will",
    "'ve": " have",
    "'m": " am",
}


@dataclass
class EnglishPreprocessorConfig:
    """Configuration for English preprocessing."""

    normalize_quotes: bool = True
    normalize_dashes: bool = True
    expand_contractions: bool = False
    lowercase: bool = False
    remove_extra_whitespace: bool = True


class EnglishPreprocessor:
    """
    English text preprocessor.

    Example:
        >>> preprocessor = EnglishPreprocessor()
        >>> preprocessor.preprocess("Hello   World")
        'Hello World'
    """

    def __init__(self, config: Optional[EnglishPreprocessorConfig] = None):
        """
        Initialize preprocessor.

        Args:
            config: Preprocessing configuration
        """
        self.config = config or EnglishPreprocessorConfig()

    def preprocess(self, text: str) -> str:
        """
        Apply all configured preprocessing steps.

        Args:
            text: Input text

        Returns:
            Preprocessed text
        """
        if not text:
            return text

        if self.config.normalize_quotes:
            text = normalize_quotes(text)

        if self.config.normalize_dashes:
            text = normalize_dashes(text)

        if self.config.expand_contractions:
            text = expand_contractions(text)

        if self.config.lowercase:
            text = text.lower()

        if self.config.remove_extra_whitespace:
            text = normalize_whitespace(text)

        return text


def normalize_quotes(text: str) -> str:
    """
    Normalize Unicode quotes to ASCII.

    Args:
        text: Input text

    Returns:
        Text with normalized quotes
    """
    for unicode_char, ascii_char in UNICODE_QUOTES.items():
        text = text.replace(unicode_char, ascii_char)
    return text


def normalize_dashes(text: str) -> str:
    """
    Normalize Unicode dashes to ASCII hyphen.

    Args:
        text: Input text

    Returns:
        Text with normalized dashes
    """
    for unicode_char, ascii_char in UNICODE_DASHES.items():
        text = text.replace(unicode_char, ascii_char)
    return text


def expand_contractions(text: str) -> str:
    """
    Expand English contractions.

    Args:
        text: Input text

    Returns:
        Text with expanded contractions

    Example:
        >>> expand_contractions("I won't do it")
        "I will not do it"
    """
    # Sort by length (longest first) to avoid partial replacements
    for contraction, expansion in sorted(
        CONTRACTIONS.items(), key=lambda x: len(x[0]), reverse=True
    ):
        # Case-insensitive replacement
        pattern = re.compile(re.escape(contraction), re.IGNORECASE)
        text = pattern.sub(expansion, text)
    return text


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text.

    Args:
        text: Input text

    Returns:
        Text with normalized whitespace
    """
    # Replace multiple spaces with single space
    text = re.sub(r" +", " ", text)

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Replace multiple newlines with double newline
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def normalize_english(
    text: str,
    normalize_quotes_flag: bool = True,
    normalize_dashes_flag: bool = True,
    lowercase: bool = False,
) -> str:
    """
    Convenience function for English text normalization.

    Args:
        text: Input text
        normalize_quotes_flag: Normalize Unicode quotes
        normalize_dashes_flag: Normalize Unicode dashes
        lowercase: Convert to lowercase

    Returns:
        Normalized text
    """
    config = EnglishPreprocessorConfig(
        normalize_quotes=normalize_quotes_flag,
        normalize_dashes=normalize_dashes_flag,
        lowercase=lowercase,
    )
    preprocessor = EnglishPreprocessor(config)
    return preprocessor.preprocess(text)
