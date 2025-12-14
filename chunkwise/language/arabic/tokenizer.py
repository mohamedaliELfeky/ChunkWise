"""
Arabic Tokenizer

Word tokenization for Arabic text with support for morphological analysis.
"""

import re
from typing import List, Optional, Tuple
from dataclasses import dataclass


# Simple Arabic word pattern
ARABIC_WORD_PATTERN = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+|[a-zA-Z]+|[\d٠-٩]+"
)

# Arabic punctuation for splitting
ARABIC_PUNCTUATION = set("،؛؟!.,:;()[]{}«»\"'")


@dataclass
class ArabicTokenizerConfig:
    """Configuration for Arabic tokenization."""

    use_camel: bool = False
    use_pyarabic: bool = False
    normalize_before_tokenize: bool = True
    keep_punctuation: bool = False
    split_clitics: bool = False


class ArabicTokenizer:
    """
    Arabic word tokenizer.

    Supports multiple backends:
    - Built-in regex-based tokenization
    - PyArabic tokenization
    - CAMeL Tools morphological tokenization

    Example:
        >>> tokenizer = ArabicTokenizer()
        >>> tokens = tokenizer.tokenize("مرحبا بالعالم")
        >>> tokens
        ['مرحبا', 'بالعالم']
    """

    def __init__(self, config: Optional[ArabicTokenizerConfig] = None):
        """
        Initialize tokenizer.

        Args:
            config: Tokenization configuration
        """
        self.config = config or ArabicTokenizerConfig()
        self._camel_analyzer = None

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.

        Args:
            text: Input Arabic text

        Returns:
            List of tokens
        """
        if not text or not text.strip():
            return []

        # Normalize if configured
        if self.config.normalize_before_tokenize:
            from chunkwise.language.arabic.preprocessor import normalize_arabic

            text = normalize_arabic(text)

        # Use appropriate backend
        if self.config.use_camel and self.config.split_clitics:
            return self._tokenize_camel_morphological(text)

        if self.config.use_camel:
            return self._tokenize_camel_simple(text)

        if self.config.use_pyarabic:
            return self._tokenize_pyarabic(text)

        return self._tokenize_builtin(text)

    def _tokenize_builtin(self, text: str) -> List[str]:
        """
        Tokenize using built-in regex.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        tokens = ARABIC_WORD_PATTERN.findall(text)

        if self.config.keep_punctuation:
            # Add punctuation as separate tokens
            all_tokens = []
            current_pos = 0
            for token in tokens:
                # Find token in text
                token_pos = text.find(token, current_pos)
                # Add any punctuation between tokens
                between = text[current_pos:token_pos]
                for char in between:
                    if char in ARABIC_PUNCTUATION:
                        all_tokens.append(char)
                all_tokens.append(token)
                current_pos = token_pos + len(token)
            tokens = all_tokens

        return tokens

    def _tokenize_pyarabic(self, text: str) -> List[str]:
        """
        Tokenize using PyArabic.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        try:
            import pyarabic.araby as araby

            tokens = araby.tokenize(text)
            if not self.config.keep_punctuation:
                tokens = [t for t in tokens if t not in ARABIC_PUNCTUATION]
            return tokens
        except ImportError:
            return self._tokenize_builtin(text)
        except Exception:
            return self._tokenize_builtin(text)

    def _tokenize_camel_simple(self, text: str) -> List[str]:
        """
        Tokenize using CAMeL Tools simple tokenizer.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        try:
            from camel_tools.tokenizers.word import simple_word_tokenize

            tokens = simple_word_tokenize(text)
            if not self.config.keep_punctuation:
                tokens = [t for t in tokens if t not in ARABIC_PUNCTUATION]
            return tokens
        except ImportError:
            return self._tokenize_builtin(text)
        except Exception:
            return self._tokenize_builtin(text)

    def _tokenize_camel_morphological(self, text: str) -> List[str]:
        """
        Tokenize using CAMeL Tools morphological analyzer.

        This splits clitics (attached prepositions, conjunctions, etc.)
        from words.

        Args:
            text: Input text

        Returns:
            List of tokens with clitics split
        """
        try:
            from camel_tools.morphology.database import MorphologyDB
            from camel_tools.morphology.analyzer import Analyzer
            from camel_tools.tokenizers.morphological import MorphologicalTokenizer

            # Initialize analyzer if needed
            if self._camel_analyzer is None:
                db = MorphologyDB.builtin_db()
                self._camel_analyzer = Analyzer(db)

            tokenizer = MorphologicalTokenizer(self._camel_analyzer, scheme="atbtok")
            tokens = tokenizer.tokenize(text)

            if not self.config.keep_punctuation:
                tokens = [t for t in tokens if t not in ARABIC_PUNCTUATION]

            return tokens
        except ImportError:
            return self._tokenize_builtin(text)
        except Exception:
            return self._tokenize_builtin(text)

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Input text

        Returns:
            Number of tokens
        """
        return len(self.tokenize(text))

    def tokenize_with_positions(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Tokenize and return positions.

        Args:
            text: Input text

        Returns:
            List of (token, start, end) tuples
        """
        tokens = self.tokenize(text)
        result = []
        current_pos = 0

        for token in tokens:
            start = text.find(token, current_pos)
            if start >= 0:
                end = start + len(token)
                result.append((token, start, end))
                current_pos = end

        return result


def tokenize_arabic(
    text: str,
    split_clitics: bool = False,
    use_camel: bool = False,
) -> List[str]:
    """
    Convenience function to tokenize Arabic text.

    Args:
        text: Input Arabic text
        split_clitics: Whether to split clitics from words
        use_camel: Use CAMeL Tools if available

    Returns:
        List of tokens

    Example:
        >>> tokenize_arabic("مرحبا بالعالم")
        ['مرحبا', 'بالعالم']
        >>> tokenize_arabic("والكتاب", split_clitics=True, use_camel=True)
        ['و', 'ال', 'كتاب']  # If CAMeL is available
    """
    config = ArabicTokenizerConfig(
        use_camel=use_camel,
        split_clitics=split_clitics,
    )
    tokenizer = ArabicTokenizer(config)
    return tokenizer.tokenize(text)


def count_arabic_words(text: str) -> int:
    """
    Count Arabic words in text.

    Args:
        text: Input text

    Returns:
        Number of words
    """
    tokenizer = ArabicTokenizer()
    return tokenizer.count_tokens(text)
