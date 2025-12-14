"""
Language Detection

Functions for detecting text language with support for Arabic, English,
and mixed-language text.
"""

import re
from typing import Optional, Tuple


# Arabic Unicode range
ARABIC_PATTERN = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")

# Basic Latin (English) pattern
LATIN_PATTERN = re.compile(r"[a-zA-Z]")


def detect_language(text: str) -> str:
    """
    Detect the primary language of text.

    Args:
        text: Input text

    Returns:
        Language code: "ar" (Arabic), "en" (English), "mixed", or "unknown"

    Example:
        >>> detect_language("Hello world")
        'en'
        >>> detect_language("مرحبا بالعالم")
        'ar'
        >>> detect_language("Hello مرحبا")
        'mixed'
    """
    if not text or not text.strip():
        return "unknown"

    arabic_count = len(ARABIC_PATTERN.findall(text))
    latin_count = len(LATIN_PATTERN.findall(text))
    total = arabic_count + latin_count

    if total == 0:
        # Try langdetect for other languages
        try:
            from langdetect import detect

            return detect(text)
        except Exception:
            return "unknown"

    arabic_ratio = arabic_count / total
    latin_ratio = latin_count / total

    # Mixed if both have significant presence
    if arabic_ratio > 0.2 and latin_ratio > 0.2:
        return "mixed"
    elif arabic_ratio > latin_ratio:
        return "ar"
    else:
        return "en"


def detect_language_detailed(text: str) -> dict:
    """
    Get detailed language detection results.

    Args:
        text: Input text

    Returns:
        Dictionary with language stats
    """
    if not text:
        return {
            "language": "unknown",
            "arabic_chars": 0,
            "latin_chars": 0,
            "arabic_ratio": 0.0,
            "latin_ratio": 0.0,
            "confidence": 0.0,
        }

    arabic_count = len(ARABIC_PATTERN.findall(text))
    latin_count = len(LATIN_PATTERN.findall(text))
    total = arabic_count + latin_count

    if total == 0:
        return {
            "language": "unknown",
            "arabic_chars": 0,
            "latin_chars": 0,
            "arabic_ratio": 0.0,
            "latin_ratio": 0.0,
            "confidence": 0.0,
        }

    arabic_ratio = arabic_count / total
    latin_ratio = latin_count / total

    # Determine language and confidence
    if arabic_ratio > 0.8:
        language = "ar"
        confidence = arabic_ratio
    elif latin_ratio > 0.8:
        language = "en"
        confidence = latin_ratio
    elif arabic_ratio > 0.2 and latin_ratio > 0.2:
        language = "mixed"
        confidence = min(arabic_ratio, latin_ratio) * 2  # Higher when more balanced
    elif arabic_ratio > latin_ratio:
        language = "ar"
        confidence = arabic_ratio
    else:
        language = "en"
        confidence = latin_ratio

    return {
        "language": language,
        "arabic_chars": arabic_count,
        "latin_chars": latin_count,
        "arabic_ratio": arabic_ratio,
        "latin_ratio": latin_ratio,
        "confidence": confidence,
    }


def is_arabic(text: str, threshold: float = 0.5) -> bool:
    """
    Check if text is primarily Arabic.

    Args:
        text: Input text
        threshold: Minimum ratio of Arabic characters

    Returns:
        True if text is primarily Arabic
    """
    if not text:
        return False

    arabic_count = len(ARABIC_PATTERN.findall(text))
    latin_count = len(LATIN_PATTERN.findall(text))
    total = arabic_count + latin_count

    if total == 0:
        return False

    return (arabic_count / total) >= threshold


def is_english(text: str, threshold: float = 0.5) -> bool:
    """
    Check if text is primarily English/Latin.

    Args:
        text: Input text
        threshold: Minimum ratio of Latin characters

    Returns:
        True if text is primarily English
    """
    if not text:
        return False

    arabic_count = len(ARABIC_PATTERN.findall(text))
    latin_count = len(LATIN_PATTERN.findall(text))
    total = arabic_count + latin_count

    if total == 0:
        return False

    return (latin_count / total) >= threshold


def is_mixed(text: str, threshold: float = 0.2) -> bool:
    """
    Check if text contains significant amounts of both Arabic and English.

    Args:
        text: Input text
        threshold: Minimum ratio for each language to be considered mixed

    Returns:
        True if text is mixed language
    """
    if not text:
        return False

    arabic_count = len(ARABIC_PATTERN.findall(text))
    latin_count = len(LATIN_PATTERN.findall(text))
    total = arabic_count + latin_count

    if total == 0:
        return False

    arabic_ratio = arabic_count / total
    latin_ratio = latin_count / total

    return arabic_ratio >= threshold and latin_ratio >= threshold


def split_by_language(text: str) -> list:
    """
    Split text into segments by language.

    Args:
        text: Input text

    Returns:
        List of (text, language) tuples

    Example:
        >>> split_by_language("Hello مرحبا World")
        [("Hello ", "en"), ("مرحبا", "ar"), (" World", "en")]
    """
    if not text:
        return []

    segments = []
    current_segment = ""
    current_lang = None

    for char in text:
        if ARABIC_PATTERN.match(char):
            char_lang = "ar"
        elif LATIN_PATTERN.match(char):
            char_lang = "en"
        else:
            # Neutral character (space, punctuation, etc.)
            char_lang = current_lang  # Keep current language

        if char_lang != current_lang and current_segment:
            if current_lang is not None:
                segments.append((current_segment, current_lang))
            current_segment = char
            current_lang = char_lang
        else:
            current_segment += char
            if current_lang is None:
                current_lang = char_lang

    if current_segment and current_lang:
        segments.append((current_segment, current_lang))

    return segments


def get_language_boundaries(text: str) -> list:
    """
    Get character positions where language changes.

    Args:
        text: Input text

    Returns:
        List of (position, from_lang, to_lang) tuples
    """
    if not text:
        return []

    boundaries = []
    prev_lang = None

    for i, char in enumerate(text):
        if ARABIC_PATTERN.match(char):
            char_lang = "ar"
        elif LATIN_PATTERN.match(char):
            char_lang = "en"
        else:
            continue  # Skip neutral characters

        if prev_lang is not None and char_lang != prev_lang:
            boundaries.append((i, prev_lang, char_lang))

        prev_lang = char_lang

    return boundaries
