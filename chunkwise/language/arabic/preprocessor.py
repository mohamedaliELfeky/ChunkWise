"""
Arabic Text Preprocessor

Functions for normalizing and cleaning Arabic text.
"""

import re
from typing import Optional
from dataclasses import dataclass


# Arabic diacritics (tashkeel)
ARABIC_DIACRITICS = re.compile(
    r"[\u064B-\u065F\u0670]"  # Fathatan, Dammatan, Kasratan, Fatha, Damma, Kasra, Shadda, Sukun, etc.
)

# Tatweel/Kashida (elongation character)
TATWEEL = "\u0640"

# Alef variants
ALEF_VARIANTS = {
    "\u0622": "\u0627",  # Alef with Madda -> Alef
    "\u0623": "\u0627",  # Alef with Hamza Above -> Alef
    "\u0625": "\u0627",  # Alef with Hamza Below -> Alef
    "\u0671": "\u0627",  # Alef Wasla -> Alef
}

# Yaa variants
YAA_VARIANTS = {
    "\u0649": "\u064A",  # Alef Maksura -> Yaa
}

# Taa Marbuta
TAA_MARBUTA = "\u0629"
HAA = "\u0647"

# Common Arabic punctuation
ARABIC_PUNCTUATION = {
    "،": ",",  # Arabic comma
    "؛": ";",  # Arabic semicolon
    "؟": "?",  # Arabic question mark
}


@dataclass
class ArabicPreprocessorConfig:
    """Configuration for Arabic preprocessing."""

    normalize_alef: bool = True
    normalize_yaa: bool = True
    normalize_taa_marbuta: bool = False
    remove_diacritics: bool = True
    remove_tatweel: bool = True
    normalize_punctuation: bool = False


class ArabicPreprocessor:
    """
    Arabic text preprocessor with configurable normalization options.

    Example:
        >>> preprocessor = ArabicPreprocessor()
        >>> preprocessor.preprocess("مُحَمَّد")  # With diacritics
        'محمد'
        >>> preprocessor.preprocess("أحمد إبراهيم")  # Alef variants
        'احمد ابراهيم'
    """

    def __init__(self, config: Optional[ArabicPreprocessorConfig] = None):
        """
        Initialize preprocessor.

        Args:
            config: Preprocessing configuration
        """
        self.config = config or ArabicPreprocessorConfig()

    def preprocess(self, text: str) -> str:
        """
        Apply all configured preprocessing steps.

        Args:
            text: Input Arabic text

        Returns:
            Preprocessed text
        """
        if not text:
            return text

        # Apply each step based on config
        if self.config.remove_diacritics:
            text = remove_diacritics(text)

        if self.config.remove_tatweel:
            text = remove_tatweel(text)

        if self.config.normalize_alef:
            text = normalize_alef(text)

        if self.config.normalize_yaa:
            text = normalize_yaa(text)

        if self.config.normalize_taa_marbuta:
            text = normalize_taa_marbuta(text)

        if self.config.normalize_punctuation:
            text = normalize_punctuation(text)

        return text


def remove_diacritics(text: str) -> str:
    """
    Remove Arabic diacritics (tashkeel) from text.

    Diacritics include: Fathatan (ً), Dammatan (ٌ), Kasratan (ٍ),
    Fatha (َ), Damma (ُ), Kasra (ِ), Shadda (ّ), Sukun (ْ)

    Args:
        text: Input text

    Returns:
        Text with diacritics removed

    Example:
        >>> remove_diacritics("مُحَمَّد")
        'محمد'
    """
    return ARABIC_DIACRITICS.sub("", text)


def remove_tatweel(text: str) -> str:
    """
    Remove Tatweel/Kashida (elongation character) from text.

    The Tatweel character (ـ) is used to stretch words for
    aesthetic purposes.

    Args:
        text: Input text

    Returns:
        Text with Tatweel removed

    Example:
        >>> remove_tatweel("مـــحـــمـــد")
        'محمد'
    """
    return text.replace(TATWEEL, "")


def normalize_alef(text: str) -> str:
    """
    Normalize Alef variants to bare Alef (ا).

    Normalizes: أ إ آ ٱ -> ا

    Args:
        text: Input text

    Returns:
        Text with normalized Alef

    Example:
        >>> normalize_alef("أحمد إبراهيم آمنة")
        'احمد ابراهيم امنة'
    """
    for variant, normalized in ALEF_VARIANTS.items():
        text = text.replace(variant, normalized)
    return text


def normalize_yaa(text: str) -> str:
    """
    Normalize Yaa variants (ي ى) to standard form.

    Normalizes: ى (Alef Maksura) -> ي

    Args:
        text: Input text

    Returns:
        Text with normalized Yaa

    Example:
        >>> normalize_yaa("على مصطفى")
        'علي مصطفي'
    """
    for variant, normalized in YAA_VARIANTS.items():
        text = text.replace(variant, normalized)
    return text


def normalize_taa_marbuta(text: str) -> str:
    """
    Normalize Taa Marbuta (ة) to Haa (ه).

    Note: This is lossy and changes meaning. Use with caution.

    Args:
        text: Input text

    Returns:
        Text with Taa Marbuta converted to Haa

    Example:
        >>> normalize_taa_marbuta("مدرسة")
        'مدرسه'
    """
    return text.replace(TAA_MARBUTA, HAA)


def normalize_punctuation(text: str) -> str:
    """
    Normalize Arabic punctuation to Western equivalents.

    Args:
        text: Input text

    Returns:
        Text with normalized punctuation

    Example:
        >>> normalize_punctuation("كيف حالك؟")
        'كيف حالك?'
    """
    for arabic, western in ARABIC_PUNCTUATION.items():
        text = text.replace(arabic, western)
    return text


def normalize_arabic(
    text: str,
    normalize_alef_variant: bool = True,
    normalize_yaa_variant: bool = True,
    remove_diacritics_flag: bool = True,
    remove_tatweel_flag: bool = True,
) -> str:
    """
    Convenience function to apply common Arabic normalization.

    Args:
        text: Input text
        normalize_alef_variant: Normalize Alef variants
        normalize_yaa_variant: Normalize Yaa variants
        remove_diacritics_flag: Remove diacritics
        remove_tatweel_flag: Remove Tatweel

    Returns:
        Normalized text

    Example:
        >>> normalize_arabic("مُحَمَّد أحْمَد")
        'محمد احمد'
    """
    if remove_diacritics_flag:
        text = remove_diacritics(text)

    if remove_tatweel_flag:
        text = remove_tatweel(text)

    if normalize_alef_variant:
        text = normalize_alef(text)

    if normalize_yaa_variant:
        text = normalize_yaa(text)

    return text


def is_arabic_char(char: str) -> bool:
    """
    Check if a character is Arabic.

    Args:
        char: Single character

    Returns:
        True if character is Arabic
    """
    if not char:
        return False

    code = ord(char)
    return (
        0x0600 <= code <= 0x06FF  # Arabic
        or 0x0750 <= code <= 0x077F  # Arabic Supplement
        or 0x08A0 <= code <= 0x08FF  # Arabic Extended-A
        or 0xFB50 <= code <= 0xFDFF  # Arabic Presentation Forms-A
        or 0xFE70 <= code <= 0xFEFF  # Arabic Presentation Forms-B
    )


def strip_non_arabic(text: str, keep_spaces: bool = True) -> str:
    """
    Remove non-Arabic characters from text.

    Args:
        text: Input text
        keep_spaces: Whether to keep spaces

    Returns:
        Text with only Arabic characters (and optionally spaces)
    """
    result = []
    for char in text:
        if is_arabic_char(char) or (keep_spaces and char.isspace()):
            result.append(char)
    return "".join(result)


def get_arabic_char_count(text: str) -> int:
    """
    Count Arabic characters in text.

    Args:
        text: Input text

    Returns:
        Number of Arabic characters
    """
    return sum(1 for char in text if is_arabic_char(char))
