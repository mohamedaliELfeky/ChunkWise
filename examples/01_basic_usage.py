"""
ChunkWise Basic Usage Examples

This file demonstrates the basic usage of the ChunkWise library.
"""

from chunkwise import Chunker, chunk_text

# Sample text
TEXT = """
Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans. AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals.

The term "artificial intelligence" had previously been used to describe machines that mimic and display "human" cognitive skills that are associated with the human mind, such as "learning" and "problem-solving". This definition has since been rejected by major AI researchers who now describe AI in terms of rationality and acting rationally.

AI applications include advanced web search engines, recommendation systems, understanding human speech, self-driving cars, generative or creative tools, automated decision-making, and competing at the highest level in strategic game systems.
"""


def example_1_simple_chunking():
    """Simple chunking with defaults."""
    print("=" * 50)
    print("Example 1: Simple Chunking")
    print("=" * 50)

    # Create chunker with default settings (recursive strategy)
    chunker = Chunker(chunk_size=200)
    chunks = chunker.chunk(TEXT)

    print(f"Created {len(chunks)} chunks:\n")
    for chunk in chunks:
        print(f"Chunk {chunk.index}:")
        print(f"  Length: {len(chunk)} chars")
        print(f"  Content: {chunk.content[:80]}...")
        print()


def example_2_one_liner():
    """One-line chunking using convenience function."""
    print("=" * 50)
    print("Example 2: One-Line Chunking")
    print("=" * 50)

    # Use the convenience function
    chunks = chunk_text(TEXT, chunk_size=200)

    print(f"Created {len(chunks)} chunks using chunk_text()")
    print()


def example_3_different_strategies():
    """Using different chunking strategies."""
    print("=" * 50)
    print("Example 3: Different Strategies")
    print("=" * 50)

    strategies = ["character", "word", "sentence", "recursive"]

    for strategy in strategies:
        chunker = Chunker(strategy=strategy, chunk_size=200)
        chunks = chunker.chunk(TEXT)
        print(f"{strategy}: {len(chunks)} chunks")

    print()


def example_4_with_overlap():
    """Chunking with overlap."""
    print("=" * 50)
    print("Example 4: Chunking with Overlap")
    print("=" * 50)

    # Create chunks with 50 character overlap
    chunker = Chunker(chunk_size=200, chunk_overlap=50)
    chunks = chunker.chunk(TEXT)

    print(f"Created {len(chunks)} chunks with overlap:\n")

    if len(chunks) > 1:
        print("Overlap between chunk 0 and 1:")
        print(f"  End of chunk 0: ...{chunks[0].content[-60:]}")
        print(f"  Start of chunk 1: {chunks[1].content[:60]}...")
    print()


def example_5_chunk_metadata():
    """Accessing chunk metadata."""
    print("=" * 50)
    print("Example 5: Chunk Metadata")
    print("=" * 50)

    chunker = Chunker(chunk_size=200)
    chunks = chunker.chunk(TEXT)

    print("Chunk metadata:\n")
    for chunk in chunks[:2]:  # Show first 2 chunks
        print(f"Chunk {chunk.index}:")
        print(f"  Start position: {chunk.start_char}")
        print(f"  End position: {chunk.end_char}")
        print(f"  Token count: {chunk.metadata.get('token_count', 'N/A')}")
        print(f"  Language: {chunk.metadata.get('language', 'N/A')}")
        print()


def example_6_multiple_documents():
    """Chunking multiple documents."""
    print("=" * 50)
    print("Example 6: Multiple Documents")
    print("=" * 50)

    documents = [
        "First document. It has multiple sentences. This is interesting.",
        "Second document with different content. More text here.",
        "Third document about another topic entirely."
    ]

    chunker = Chunker(chunk_size=100)
    batches = chunker.chunk_documents(documents)

    print(f"Processed {len(batches)} documents:\n")
    for i, batch in enumerate(batches):
        print(f"Document {i}: {len(batch.chunks)} chunks")

    print()


def example_7_callable_interface():
    """Using Chunker as a callable."""
    print("=" * 50)
    print("Example 7: Callable Interface")
    print("=" * 50)

    chunker = Chunker(chunk_size=200)

    # Call chunker directly like a function
    chunks = chunker(TEXT)

    print(f"Called chunker directly: {len(chunks)} chunks")
    print()


if __name__ == "__main__":
    example_1_simple_chunking()
    example_2_one_liner()
    example_3_different_strategies()
    example_4_with_overlap()
    example_5_chunk_metadata()
    example_6_multiple_documents()
    example_7_callable_interface()

    print("All examples completed!")
