"""
ChunkWise Mixed Language Examples

This file demonstrates handling mixed Arabic-English text.
"""

from chunkwise import Chunker
from chunkwise.language.detector import (
    detect_language,
    split_by_language,
    get_language_boundaries,
)
from chunkwise.config import ChunkConfig, ArabicConfig

# Mixed language texts
MIXED_TEXTS = {
    "tech_blog": """
    # Welcome to Tech Blog مرحباً بكم في مدونة التقنية

    Today we'll discuss AI الذكاء الاصطناعي and its applications.
    سنناقش اليوم تطبيقات الذكاء الاصطناعي المختلفة.

    ## Machine Learning التعلم الآلي

    Machine learning is a subset of AI that enables computers to learn from data.
    التعلم الآلي هو فرع من الذكاء الاصطناعي يمكّن الحواسيب من التعلم من البيانات.

    ### Key Concepts المفاهيم الأساسية

    1. Supervised Learning - التعلم الخاضع للإشراف
    2. Unsupervised Learning - التعلم غير الخاضع للإشراف
    3. Reinforcement Learning - التعلم المعزز
    """,

    "news_article": """
    Breaking News أخبار عاجلة

    The technology conference المؤتمر التقني was held in Dubai دبي today.
    حضر المؤتمر أكثر من 5000 شخص من مختلف أنحاء العالم.

    "This is a milestone for the region," said the CEO.
    قال الرئيس التنفيذي: "هذا إنجاز تاريخي للمنطقة".
    """,

    "product_description": """
    iPhone 15 Pro آيفون 15 برو

    The most powerful iPhone ever أقوى آيفون على الإطلاق

    Features المميزات:
    - A17 Pro chip معالج A17 برو
    - Titanium design تصميم من التيتانيوم
    - 5x optical zoom عدسة تقريب 5x
    """
}


def example_1_detect_mixed_language():
    """Detect language in mixed text."""
    print("=" * 50)
    print("Example 1: Language Detection")
    print("=" * 50)

    for name, text in MIXED_TEXTS.items():
        lang = detect_language(text)
        print(f"\n{name}: detected as '{lang}'")

    # Test specific phrases
    phrases = [
        "Hello world",
        "مرحبا بالعالم",
        "Hello مرحبا World",
        "The الذكاء AI الاصطناعي",
    ]

    print("\nPhrase-level detection:")
    for phrase in phrases:
        lang = detect_language(phrase)
        print(f"  '{phrase}' -> {lang}")

    print()


def example_2_split_by_language():
    """Split text into language segments."""
    print("=" * 50)
    print("Example 2: Split by Language")
    print("=" * 50)

    text = "Hello مرحبا World عالم!"
    segments = split_by_language(text)

    print(f"Text: '{text}'")
    print("\nSegments:")
    for segment, lang in segments:
        print(f"  '{segment}' -> {lang}")

    print()


def example_3_language_boundaries():
    """Find language transition points."""
    print("=" * 50)
    print("Example 3: Language Boundaries")
    print("=" * 50)

    text = "Hello مرحبا World عالم!"
    boundaries = get_language_boundaries(text)

    print(f"Text: '{text}'")
    print("\nLanguage transitions:")
    for pos, from_lang, to_lang in boundaries:
        print(f"  Position {pos}: {from_lang} -> {to_lang}")

    print()


def example_4_chunk_mixed_text():
    """Chunk mixed language text."""
    print("=" * 50)
    print("Example 4: Chunking Mixed Text")
    print("=" * 50)

    # Use auto language detection
    chunker = Chunker(
        strategy="recursive",
        chunk_size=200,
        language="auto",
    )

    for name, text in MIXED_TEXTS.items():
        print(f"\n{name}:")
        chunks = chunker.chunk(text)
        print(f"  Created {len(chunks)} chunks")

        for chunk in chunks[:2]:  # Show first 2
            lang = chunk.metadata.get("language", "unknown")
            preview = chunk.content[:50].replace("\n", " ")
            print(f"    Chunk {chunk.index} ({lang}): {preview}...")

    print()


def example_5_sentence_chunking_mixed():
    """Sentence-based chunking for mixed text."""
    print("=" * 50)
    print("Example 5: Sentence Chunking (Mixed)")
    print("=" * 50)

    text = MIXED_TEXTS["news_article"]

    chunker = Chunker(
        strategy="sentence",
        chunk_size=300,
        language="auto",
    )
    chunks = chunker.chunk(text)

    print(f"Created {len(chunks)} sentence-based chunks:\n")
    for chunk in chunks:
        lang = chunk.metadata.get("language", "unknown")
        print(f"Chunk {chunk.index} (detected: {lang}):")
        print(f"  {chunk.content[:80]}...")
        print()


def example_6_arabic_normalization_in_mixed():
    """Apply Arabic normalization in mixed text."""
    print("=" * 50)
    print("Example 6: Arabic Normalization in Mixed Text")
    print("=" * 50)

    from chunkwise.language.arabic.preprocessor import normalize_arabic

    text = "The مُحَمَّد name أحْمَد is common"

    print(f"Original: {text}")

    # Normalize only Arabic parts
    segments = split_by_language(text)
    normalized_parts = []

    for segment, lang in segments:
        if lang == "ar":
            normalized_parts.append(normalize_arabic(segment))
        else:
            normalized_parts.append(segment)

    normalized = "".join(normalized_parts)
    print(f"Normalized: {normalized}")

    print()


def example_7_custom_config_for_mixed():
    """Custom configuration for mixed language handling."""
    print("=" * 50)
    print("Example 7: Custom Config for Mixed Text")
    print("=" * 50)

    # Configure Arabic-specific settings
    arabic_config = ArabicConfig(
        normalize_alef=True,
        normalize_yaa=True,
        remove_diacritics=True,
    )

    config = ChunkConfig(
        strategy="recursive",
        chunk_size=200,
        language="auto",  # Auto-detect
        arabic_config=arabic_config,
    )

    chunker = Chunker(config=config)
    chunks = chunker.chunk(MIXED_TEXTS["tech_blog"])

    print(f"Created {len(chunks)} chunks with custom config\n")
    for chunk in chunks[:3]:
        lang = chunk.metadata.get("language", "unknown")
        print(f"Chunk {chunk.index} ({lang}): {len(chunk)} chars")

    print()


def example_8_markdown_mixed():
    """Markdown chunking for mixed language."""
    print("=" * 50)
    print("Example 8: Markdown Chunking (Mixed)")
    print("=" * 50)

    chunker = Chunker(
        strategy="markdown",
        chunk_size=300,
    )
    chunks = chunker.chunk(MIXED_TEXTS["tech_blog"])

    print(f"Created {len(chunks)} markdown-aware chunks:\n")
    for chunk in chunks:
        # Check if chunk starts with header
        is_header = chunk.content.strip().startswith("#")
        title = chunk.metadata.get("section_title", "N/A")
        print(f"Chunk {chunk.index}: section='{title}', has_header={is_header}")

    print()


if __name__ == "__main__":
    example_1_detect_mixed_language()
    example_2_split_by_language()
    example_3_language_boundaries()
    example_4_chunk_mixed_text()
    example_5_sentence_chunking_mixed()
    example_6_arabic_normalization_in_mixed()
    example_7_custom_config_for_mixed()
    example_8_markdown_mixed()

    print("All mixed language examples completed!")
