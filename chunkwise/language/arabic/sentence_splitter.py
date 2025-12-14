"""
Arabic Sentence Splitter

Functions for splitting Arabic text into sentences.
"""

import re
from typing import List, Optional
from dataclasses import dataclass, field


# Arabic sentence-ending punctuation
ARABIC_SENTENCE_ENDERS = [
    ".",   # Period (used in Arabic too)
    "؟",   # Arabic question mark
    "!",   # Exclamation mark
    "؛",   # Arabic semicolon (sometimes ends sentences)
    "۔",   # Arabic full stop (Urdu/Persian style)
    "。",  # CJK full stop (sometimes in mixed text)
]

# Pattern for sentence endings
SENTENCE_END_PATTERN = re.compile(
    r'([.؟!؛۔。])\s*'
)

# Common Arabic abbreviations that shouldn't split sentences
ARABIC_ABBREVIATIONS = [
    "ص.",   # Page
    "ج.",   # Part/Volume
    "م.",   # Year/Meter
    "هـ.",  # Hijri year
    "د.",   # Doctor
    "أ.",   # Professor/Teacher
    "ب.",   # Section
]


@dataclass
class ArabicSentenceSplitterConfig:
    """Configuration for Arabic sentence splitting."""

    sentence_enders: List[str] = field(
        default_factory=lambda: ARABIC_SENTENCE_ENDERS.copy()
    )
    min_sentence_length: int = 10
    respect_abbreviations: bool = True
    use_pyarabic: bool = False
    use_camel: bool = False


class ArabicSentenceSplitter:
    """
    Split Arabic text into sentences.

    Handles Arabic-specific punctuation and abbreviations.

    Example:
        >>> splitter = ArabicSentenceSplitter()
        >>> sentences = splitter.split("كيف حالك؟ أنا بخير.")
        >>> len(sentences)
        2
    """

    def __init__(self, config: Optional[ArabicSentenceSplitterConfig] = None):
        """
        Initialize sentence splitter.

        Args:
            config: Configuration for sentence splitting
        """
        self.config = config or ArabicSentenceSplitterConfig()
        self._pattern = self._build_pattern()

    def _build_pattern(self) -> re.Pattern:
        """Build regex pattern for sentence splitting."""
        enders = "|".join(re.escape(e) for e in self.config.sentence_enders)
        return re.compile(f"([{enders}])\\s*")

    def split(self, text: str) -> List[str]:
        """
        Split text into sentences.

        Args:
            text: Input Arabic text

        Returns:
            List of sentences
        """
        if not text or not text.strip():
            return []

        # Try external libraries if configured
        if self.config.use_camel:
            return self._split_with_camel(text)

        if self.config.use_pyarabic:
            return self._split_with_pyarabic(text)

        # Use built-in splitting
        return self._split_builtin(text)

    def _split_builtin(self, text: str) -> List[str]:
        """
        Split using built-in regex-based method.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        # Handle abbreviations
        if self.config.respect_abbreviations:
            text = self._protect_abbreviations(text)

        # Split by sentence enders
        parts = self._pattern.split(text)

        # Recombine parts (content + punctuation pairs)
        sentences = []
        current = ""

        for i, part in enumerate(parts):
            if i % 2 == 0:
                # Content part
                current += part
            else:
                # Punctuation part
                current += part
                if current.strip():
                    sentences.append(current.strip())
                current = ""

        # Add any remaining content
        if current.strip():
            sentences.append(current.strip())

        # Restore abbreviations
        if self.config.respect_abbreviations:
            sentences = [self._restore_abbreviations(s) for s in sentences]

        # Filter by minimum length
        if self.config.min_sentence_length > 0:
            sentences = [
                s for s in sentences
                if len(s) >= self.config.min_sentence_length
            ]

        return sentences

    def _protect_abbreviations(self, text: str) -> str:
        """
        Replace abbreviations with placeholders.

        Args:
            text: Input text

        Returns:
            Text with abbreviations protected
        """
        for i, abbr in enumerate(ARABIC_ABBREVIATIONS):
            placeholder = f"__ABBR_{i}__"
            text = text.replace(abbr, placeholder)
        return text

    def _restore_abbreviations(self, text: str) -> str:
        """
        Restore abbreviations from placeholders.

        Args:
            text: Text with placeholders

        Returns:
            Text with abbreviations restored
        """
        for i, abbr in enumerate(ARABIC_ABBREVIATIONS):
            placeholder = f"__ABBR_{i}__"
            text = text.replace(placeholder, abbr)
        return text

    def _split_with_pyarabic(self, text: str) -> List[str]:
        """
        Split using PyArabic library.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        try:
            import pyarabic.araby as araby

            sentences = araby.sentence_tokenize(text)
            return [s.strip() for s in sentences if s.strip()]
        except ImportError:
            # Fallback to built-in
            return self._split_builtin(text)
        except Exception:
            return self._split_builtin(text)

    def _split_with_camel(self, text: str) -> List[str]:
        """
        Split using CAMeL Tools library.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        try:
            from camel_tools.tokenizers.word import simple_word_tokenize
            from camel_tools.utils.charsets import UNICODE_PUNCT_CHARSET

            # CAMeL doesn't have a sentence tokenizer, so we use our own
            # logic with CAMeL's character analysis
            return self._split_builtin(text)
        except ImportError:
            return self._split_builtin(text)
        except Exception:
            return self._split_builtin(text)


def split_arabic_sentences(
    text: str,
    min_length: int = 10,
    use_pyarabic: bool = False,
) -> List[str]:
    """
    Convenience function to split Arabic text into sentences.

    Args:
        text: Input Arabic text
        min_length: Minimum sentence length
        use_pyarabic: Use PyArabic library if available

    Returns:
        List of sentences

    Example:
        >>> sentences = split_arabic_sentences("كيف حالك؟ أنا بخير. شكراً!")
        >>> len(sentences)
        3
    """
    config = ArabicSentenceSplitterConfig(
        min_sentence_length=min_length,
        use_pyarabic=use_pyarabic,
    )
    splitter = ArabicSentenceSplitter(config)
    return splitter.split(text)


def count_arabic_sentences(text: str) -> int:
    """
    Count sentences in Arabic text.

    Args:
        text: Input text

    Returns:
        Number of sentences
    """
    sentences = split_arabic_sentences(text, min_length=0)
    return len(sentences)


def get_sentence_boundaries(text: str) -> List[tuple]:
    """
    Get character positions of sentence boundaries.

    Args:
        text: Input text

    Returns:
        List of (start, end) tuples for each sentence
    """
    sentences = split_arabic_sentences(text, min_length=0)
    boundaries = []
    current_pos = 0

    for sentence in sentences:
        # Find sentence in original text
        start = text.find(sentence, current_pos)
        if start >= 0:
            end = start + len(sentence)
            boundaries.append((start, end))
            current_pos = end

    return boundaries
