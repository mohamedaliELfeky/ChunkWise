"""
Tests for Arabic Language Support
"""

import pytest
from chunkwise.language.arabic.preprocessor import (
    ArabicPreprocessor,
    normalize_arabic,
    remove_diacritics,
    normalize_alef,
)
from chunkwise.language.arabic.sentence_splitter import (
    ArabicSentenceSplitter,
    split_arabic_sentences,
)
from chunkwise.language.arabic.tokenizer import (
    ArabicTokenizer,
    tokenize_arabic,
)
from chunkwise.language.detector import (
    detect_language,
    is_arabic,
    is_english,
    is_mixed,
)


class TestArabicPreprocessor:
    """Tests for Arabic text preprocessing."""

    def test_remove_diacritics(self):
        """Test diacritics removal."""
        text_with_diacritics = "مُحَمَّد"
        result = remove_diacritics(text_with_diacritics)
        assert result == "محمد"

    def test_normalize_alef(self):
        """Test Alef normalization."""
        text = "أحمد إبراهيم"
        result = normalize_alef(text)
        assert result == "احمد ابراهيم"

    def test_full_normalization(self):
        """Test full Arabic normalization."""
        text = "مُحَمَّد أحْمَد"
        result = normalize_arabic(text)
        assert "ُ" not in result
        assert "َ" not in result

    def test_preprocessor_class(self):
        """Test ArabicPreprocessor class."""
        preprocessor = ArabicPreprocessor()
        text = "مُحَمَّد"
        result = preprocessor.preprocess(text)
        assert result == "محمد"


class TestArabicSentenceSplitter:
    """Tests for Arabic sentence splitting."""

    def test_basic_splitting(self):
        """Test basic sentence splitting."""
        text = "كيف حالك؟ أنا بخير. شكراً!"
        sentences = split_arabic_sentences(text, min_length=0)
        assert len(sentences) == 3

    def test_arabic_question_mark(self):
        """Test Arabic question mark handling."""
        text = "ما اسمك؟"
        sentences = split_arabic_sentences(text, min_length=0)
        assert len(sentences) == 1

    def test_splitter_class(self):
        """Test ArabicSentenceSplitter class."""
        splitter = ArabicSentenceSplitter()
        text = "هذه جملة. وهذه جملة أخرى."
        sentences = splitter.split(text)
        assert len(sentences) >= 1


class TestArabicTokenizer:
    """Tests for Arabic tokenization."""

    def test_basic_tokenization(self):
        """Test basic tokenization."""
        text = "مرحبا بالعالم"
        tokens = tokenize_arabic(text)
        assert len(tokens) == 2

    def test_tokenizer_class(self):
        """Test ArabicTokenizer class."""
        tokenizer = ArabicTokenizer()
        tokens = tokenizer.tokenize("مرحبا بالعالم")
        assert len(tokens) >= 2

    def test_token_count(self):
        """Test token counting."""
        tokenizer = ArabicTokenizer()
        count = tokenizer.count_tokens("واحد اثنان ثلاثة")
        assert count == 3


class TestLanguageDetection:
    """Tests for language detection."""

    def test_detect_english(self):
        """Test English detection."""
        text = "Hello world, this is a test."
        lang = detect_language(text)
        assert lang == "en"

    def test_detect_arabic(self):
        """Test Arabic detection."""
        text = "مرحبا بالعالم"
        lang = detect_language(text)
        assert lang == "ar"

    def test_detect_mixed(self):
        """Test mixed language detection."""
        text = "Hello مرحبا World"
        lang = detect_language(text)
        assert lang == "mixed"

    def test_is_arabic_function(self):
        """Test is_arabic function."""
        assert is_arabic("مرحبا بالعالم")
        assert not is_arabic("Hello world")

    def test_is_english_function(self):
        """Test is_english function."""
        assert is_english("Hello world")
        assert not is_english("مرحبا بالعالم")

    def test_is_mixed_function(self):
        """Test is_mixed function."""
        assert is_mixed("Hello مرحبا")
        assert not is_mixed("Hello world")
        assert not is_mixed("مرحبا بالعالم")
