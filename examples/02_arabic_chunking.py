"""
ChunkWise Arabic Text Chunking Examples

This file demonstrates Arabic text processing with ChunkWise.
"""

from chunkwise import Chunker, chunk_arabic
from chunkwise.config import ChunkConfig, ArabicConfig
from chunkwise.language.arabic.preprocessor import (
    normalize_arabic,
    remove_diacritics,
    normalize_alef,
)
from chunkwise.language.arabic.sentence_splitter import split_arabic_sentences
from chunkwise.language.detector import detect_language

# Sample Arabic texts
ARABIC_TEXT = """
الذكاء الاصطناعي هو فرع من علوم الحاسوب يهدف إلى إنشاء آلات ذكية تعمل وتتفاعل مثل البشر. بعض الأنشطة التي صممت لها أجهزة الكمبيوتر ذات الذكاء الاصطناعي تشمل التعرف على الكلام والتعلم والتخطيط وحل المشكلات.

يعتبر الذكاء الاصطناعي مجالاً متعدد التخصصات، حيث يجمع بين علوم الحاسوب والرياضيات وعلم النفس واللغويات. منذ بدايته، أثار الذكاء الاصطناعي جدلاً فلسفياً حول طبيعة العقل البشري.

تطبيقات الذكاء الاصطناعي متعددة ومتنوعة، وتشمل محركات البحث المتقدمة وأنظمة التوصية والتعرف على الكلام والسيارات ذاتية القيادة والأدوات الإبداعية.
"""

ARABIC_WITH_DIACRITICS = """
مُحَمَّدٌ رَسُولُ اللهِ صَلَّى اللهُ عَلَيْهِ وَسَلَّمَ. الإِسْلامُ دِينُ السَّلامِ وَالرَّحْمَةِ.
"""

MIXED_TEXT = """
Welcome to ChunkWise! مرحباً بكم في تشنك وايز

This library supports both Arabic العربية and English الإنجليزية.
تدعم هذه المكتبة كلاً من العربية والإنجليزية.
"""


def example_1_basic_arabic_chunking():
    """Basic Arabic text chunking."""
    print("=" * 50)
    print("Example 1: Basic Arabic Chunking")
    print("=" * 50)

    chunker = Chunker(chunk_size=150, language="ar")
    chunks = chunker.chunk(ARABIC_TEXT)

    print(f"Created {len(chunks)} chunks:\n")
    for chunk in chunks:
        print(f"Chunk {chunk.index}:")
        print(f"  {chunk.content[:70]}...")
        print()


def example_2_convenience_function():
    """Using the chunk_arabic convenience function."""
    print("=" * 50)
    print("Example 2: chunk_arabic() Function")
    print("=" * 50)

    chunks = chunk_arabic(ARABIC_TEXT, chunk_size=150)

    print(f"Created {len(chunks)} chunks using chunk_arabic()")
    print()


def example_3_diacritics_handling():
    """Handling Arabic diacritics (tashkeel)."""
    print("=" * 50)
    print("Example 3: Diacritics Handling")
    print("=" * 50)

    print("Original text with diacritics:")
    print(f"  {ARABIC_WITH_DIACRITICS.strip()}")
    print()

    # Remove diacritics
    clean_text = remove_diacritics(ARABIC_WITH_DIACRITICS)
    print("After removing diacritics:")
    print(f"  {clean_text.strip()}")
    print()


def example_4_normalization():
    """Arabic character normalization."""
    print("=" * 50)
    print("Example 4: Arabic Normalization")
    print("=" * 50)

    # Text with different Alef variants
    text = "أحمد وإبراهيم وآمنة"
    print(f"Original: {text}")

    normalized = normalize_alef(text)
    print(f"Alef normalized: {normalized}")

    full_normalized = normalize_arabic(text)
    print(f"Fully normalized: {full_normalized}")
    print()


def example_5_sentence_splitting():
    """Arabic sentence splitting."""
    print("=" * 50)
    print("Example 5: Arabic Sentence Splitting")
    print("=" * 50)

    text = "كيف حالك؟ أنا بخير. شكراً لك!"
    sentences = split_arabic_sentences(text, min_length=0)

    print(f"Text: {text}")
    print(f"\nSplit into {len(sentences)} sentences:")
    for i, sentence in enumerate(sentences):
        print(f"  {i + 1}. {sentence}")
    print()


def example_6_language_detection():
    """Automatic language detection."""
    print("=" * 50)
    print("Example 6: Language Detection")
    print("=" * 50)

    texts = [
        ("Hello world", "English text"),
        ("مرحبا بالعالم", "Arabic text"),
        ("Hello مرحبا world", "Mixed text"),
    ]

    for text, description in texts:
        lang = detect_language(text)
        print(f"{description}: '{text}' -> {lang}")
    print()


def example_7_custom_arabic_config():
    """Using custom Arabic configuration."""
    print("=" * 50)
    print("Example 7: Custom Arabic Config")
    print("=" * 50)

    # Create custom Arabic config
    arabic_config = ArabicConfig(
        normalize_alef=True,
        normalize_yaa=True,
        remove_diacritics=True,
        remove_tatweel=True,
    )

    config = ChunkConfig(
        strategy="recursive",
        chunk_size=150,
        language="ar",
        arabic_config=arabic_config,
    )

    chunker = Chunker(config=config)
    chunks = chunker.chunk(ARABIC_TEXT)

    print(f"Created {len(chunks)} chunks with custom config")
    print()


def example_8_mixed_language():
    """Handling mixed Arabic-English text."""
    print("=" * 50)
    print("Example 8: Mixed Language Text")
    print("=" * 50)

    # Auto-detect language
    chunker = Chunker(chunk_size=100, language="auto")
    chunks = chunker.chunk(MIXED_TEXT)

    print(f"Mixed text chunked into {len(chunks)} chunks:\n")
    for chunk in chunks:
        lang = chunk.metadata.get("language", "unknown")
        print(f"Chunk {chunk.index} (detected: {lang}):")
        print(f"  {chunk.content}")
        print()


def example_9_sentence_chunking_arabic():
    """Sentence-based chunking for Arabic."""
    print("=" * 50)
    print("Example 9: Arabic Sentence Chunking")
    print("=" * 50)

    chunker = Chunker(strategy="sentence", chunk_size=200, language="ar")
    chunks = chunker.chunk(ARABIC_TEXT)

    print(f"Created {len(chunks)} sentence-based chunks:\n")
    for chunk in chunks:
        sentence_count = chunk.metadata.get("sentence_count", "N/A")
        print(f"Chunk {chunk.index} ({sentence_count} sentences):")
        print(f"  {chunk.content[:60]}...")
        print()


if __name__ == "__main__":
    example_1_basic_arabic_chunking()
    example_2_convenience_function()
    example_3_diacritics_handling()
    example_4_normalization()
    example_5_sentence_splitting()
    example_6_language_detection()
    example_7_custom_arabic_config()
    example_8_mixed_language()
    example_9_sentence_chunking_arabic()

    print("All Arabic examples completed!")
