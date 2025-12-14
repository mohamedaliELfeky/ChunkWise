"""
English Sentence Splitter

Functions for splitting English text into sentences.
"""

import re
from typing import List, Optional
from dataclasses import dataclass, field


# Common abbreviations that don't end sentences
ABBREVIATIONS = [
    "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.",
    "vs.", "etc.", "e.g.", "i.e.", "al.", "et.",
    "Inc.", "Ltd.", "Corp.", "Co.",
    "Jan.", "Feb.", "Mar.", "Apr.", "Jun.", "Jul.", "Aug.", "Sep.", "Sept.", "Oct.", "Nov.", "Dec.",
    "Mon.", "Tue.", "Wed.", "Thu.", "Fri.", "Sat.", "Sun.",
    "St.", "Ave.", "Blvd.", "Rd.",
    "Fig.", "Vol.", "No.", "pp.", "p.",
    "U.S.", "U.K.", "U.N.",
]

# Pattern for sentence endings
SENTENCE_END_PATTERN = re.compile(
    r'([.!?])\s+'
)


@dataclass
class EnglishSentenceSplitterConfig:
    """Configuration for English sentence splitting."""

    sentence_enders: List[str] = field(default_factory=lambda: [".", "!", "?"])
    min_sentence_length: int = 10
    respect_abbreviations: bool = True
    use_spacy: bool = False


class EnglishSentenceSplitter:
    """
    Split English text into sentences.

    Example:
        >>> splitter = EnglishSentenceSplitter()
        >>> sentences = splitter.split("Hello world. How are you?")
        >>> len(sentences)
        2
    """

    def __init__(self, config: Optional[EnglishSentenceSplitterConfig] = None):
        """
        Initialize sentence splitter.

        Args:
            config: Configuration for sentence splitting
        """
        self.config = config or EnglishSentenceSplitterConfig()
        self._spacy_nlp = None

    def split(self, text: str) -> List[str]:
        """
        Split text into sentences.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        if not text or not text.strip():
            return []

        if self.config.use_spacy:
            return self._split_with_spacy(text)

        return self._split_builtin(text)

    def _split_builtin(self, text: str) -> List[str]:
        """
        Split using built-in regex-based method.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        # Protect abbreviations
        if self.config.respect_abbreviations:
            text = self._protect_abbreviations(text)

        # Split by sentence enders
        parts = SENTENCE_END_PATTERN.split(text)

        # Recombine parts
        sentences = []
        current = ""

        for i, part in enumerate(parts):
            if i % 2 == 0:
                current += part
            else:
                current += part
                if current.strip():
                    sentences.append(current.strip())
                current = ""

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
        """Replace abbreviations with placeholders."""
        for i, abbr in enumerate(ABBREVIATIONS):
            placeholder = f"__ABBR_{i}__"
            text = text.replace(abbr, placeholder)
        return text

    def _restore_abbreviations(self, text: str) -> str:
        """Restore abbreviations from placeholders."""
        for i, abbr in enumerate(ABBREVIATIONS):
            placeholder = f"__ABBR_{i}__"
            text = text.replace(placeholder, abbr)
        return text

    def _split_with_spacy(self, text: str) -> List[str]:
        """
        Split using spaCy.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        try:
            import spacy

            if self._spacy_nlp is None:
                try:
                    self._spacy_nlp = spacy.load("en_core_web_sm")
                except OSError:
                    # Model not installed, use blank
                    self._spacy_nlp = spacy.blank("en")
                    self._spacy_nlp.add_pipe("sentencizer")

            doc = self._spacy_nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents]

            if self.config.min_sentence_length > 0:
                sentences = [
                    s for s in sentences
                    if len(s) >= self.config.min_sentence_length
                ]

            return sentences
        except ImportError:
            return self._split_builtin(text)
        except Exception:
            return self._split_builtin(text)


def split_english_sentences(
    text: str,
    min_length: int = 10,
    use_spacy: bool = False,
) -> List[str]:
    """
    Convenience function to split English text into sentences.

    Args:
        text: Input text
        min_length: Minimum sentence length
        use_spacy: Use spaCy if available

    Returns:
        List of sentences

    Example:
        >>> sentences = split_english_sentences("Hello world. How are you? I'm fine!")
        >>> len(sentences)
        3
    """
    config = EnglishSentenceSplitterConfig(
        min_sentence_length=min_length,
        use_spacy=use_spacy,
    )
    splitter = EnglishSentenceSplitter(config)
    return splitter.split(text)


def count_english_sentences(text: str) -> int:
    """
    Count sentences in English text.

    Args:
        text: Input text

    Returns:
        Number of sentences
    """
    sentences = split_english_sentences(text, min_length=0)
    return len(sentences)
