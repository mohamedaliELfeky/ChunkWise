"""
Language Support Module

Language detection and language-specific processing for Arabic and English.
"""

from chunkwise.language.detector import detect_language, is_arabic, is_english, is_mixed

__all__ = [
    "detect_language",
    "is_arabic",
    "is_english",
    "is_mixed",
]
