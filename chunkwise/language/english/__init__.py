"""
English Language Support

Preprocessing and sentence splitting for English text.
"""

from chunkwise.language.english.preprocessor import (
    EnglishPreprocessor,
    normalize_english,
)
from chunkwise.language.english.sentence_splitter import (
    EnglishSentenceSplitter,
    split_english_sentences,
)

__all__ = [
    "EnglishPreprocessor",
    "normalize_english",
    "EnglishSentenceSplitter",
    "split_english_sentences",
]
