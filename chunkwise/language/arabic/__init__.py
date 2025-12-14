"""
Arabic Language Support

Preprocessing, sentence splitting, and tokenization for Arabic text.
"""

from chunkwise.language.arabic.preprocessor import (
    ArabicPreprocessor,
    normalize_arabic,
    remove_diacritics,
    remove_tatweel,
    normalize_alef,
    normalize_yaa,
)
from chunkwise.language.arabic.sentence_splitter import (
    ArabicSentenceSplitter,
    split_arabic_sentences,
)
from chunkwise.language.arabic.tokenizer import (
    ArabicTokenizer,
    tokenize_arabic,
)

__all__ = [
    # Preprocessor
    "ArabicPreprocessor",
    "normalize_arabic",
    "remove_diacritics",
    "remove_tatweel",
    "normalize_alef",
    "normalize_yaa",
    # Sentence splitter
    "ArabicSentenceSplitter",
    "split_arabic_sentences",
    # Tokenizer
    "ArabicTokenizer",
    "tokenize_arabic",
]
