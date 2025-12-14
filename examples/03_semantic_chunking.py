"""
ChunkWise Semantic Chunking Examples

This file demonstrates semantic chunking using embeddings.
Requires: pip install sentence-transformers
"""

from chunkwise import Chunker
from chunkwise.config import ChunkConfig, EmbeddingConfig

# Sample text with multiple topics
TEXT = """
Machine learning is a subset of artificial intelligence that focuses on building applications that learn from data and improve their accuracy over time without being programmed to do so. In machine learning, algorithms are trained to find patterns and correlations in large data sets and to make the best decisions based on that analysis.

Deep learning is a subfield of machine learning that uses neural networks with multiple layers. These deep neural networks attempt to simulate the behavior of the human brain, allowing them to learn from large amounts of unstructured data. Deep learning is particularly effective at tasks like image recognition and natural language processing.

Natural language processing (NLP) is a branch of artificial intelligence that helps computers understand, interpret, and manipulate human language. NLP combines computational linguistics with statistical, machine learning, and deep learning models to process human language in the form of text or voice data.

Computer vision is a field of artificial intelligence that trains computers to interpret and understand the visual world. Using digital images from cameras and videos and deep learning models, machines can accurately identify and classify objects and then react to what they see.
"""


def example_1_basic_semantic_chunking():
    """Basic semantic chunking."""
    print("=" * 50)
    print("Example 1: Basic Semantic Chunking")
    print("=" * 50)

    try:
        chunker = Chunker(
            strategy="semantic",
            chunk_size=500,
            embedding_model="all-MiniLM-L6-v2",
        )
        chunks = chunker.chunk(TEXT)

        print(f"Created {len(chunks)} semantic chunks:\n")
        for chunk in chunks:
            print(f"Chunk {chunk.index}:")
            print(f"  Sentences: {chunk.metadata.get('sentence_count', 'N/A')}")
            print(f"  Content: {chunk.content[:80]}...")
            print()

    except Exception as e:
        print(f"Note: Semantic chunking requires sentence-transformers.")
        print(f"Install with: pip install sentence-transformers")
        print(f"Error: {e}")
    print()


def example_2_custom_threshold():
    """Semantic chunking with custom similarity threshold."""
    print("=" * 50)
    print("Example 2: Custom Similarity Threshold")
    print("=" * 50)

    try:
        from chunkwise.strategies.semantic import SemanticChunker

        # Lower threshold = more chunks (split on smaller similarity drops)
        chunker_low = SemanticChunker(
            chunk_size=500,
            similarity_threshold=0.3,  # More aggressive splitting
        )

        # Higher threshold = fewer chunks (only split on major topic changes)
        chunker_high = SemanticChunker(
            chunk_size=500,
            similarity_threshold=0.7,  # Less aggressive splitting
        )

        chunks_low = chunker_low.chunk(TEXT)
        chunks_high = chunker_high.chunk(TEXT)

        print(f"Low threshold (0.3): {len(chunks_low)} chunks")
        print(f"High threshold (0.7): {len(chunks_high)} chunks")

    except Exception as e:
        print(f"Note: Requires sentence-transformers. Error: {e}")
    print()


def example_3_different_embedding_models():
    """Using different embedding models."""
    print("=" * 50)
    print("Example 3: Different Embedding Models")
    print("=" * 50)

    models = [
        ("all-MiniLM-L6-v2", "Fast, good quality"),
        ("all-mpnet-base-v2", "Higher quality"),
    ]

    for model, description in models:
        try:
            chunker = Chunker(
                strategy="semantic",
                chunk_size=500,
                embedding_model=model,
            )
            chunks = chunker.chunk(TEXT)
            print(f"{model} ({description}): {len(chunks)} chunks")
        except Exception as e:
            print(f"{model}: Not available - {e}")

    print()


def example_4_with_embeddings_config():
    """Using EmbeddingConfig for advanced settings."""
    print("=" * 50)
    print("Example 4: Advanced Embedding Config")
    print("=" * 50)

    try:
        embedding_config = EmbeddingConfig(
            model="all-MiniLM-L6-v2",
            provider="sentence-transformers",
            batch_size=32,
            normalize=True,
        )

        config = ChunkConfig(
            strategy="semantic",
            chunk_size=500,
            embedding_config=embedding_config,
        )

        chunker = Chunker(config=config)
        chunks = chunker.chunk(TEXT)

        print(f"Created {len(chunks)} chunks with custom embedding config")

    except Exception as e:
        print(f"Note: Requires sentence-transformers. Error: {e}")
    print()


def example_5_late_chunking():
    """Late chunking (embed full document first)."""
    print("=" * 50)
    print("Example 5: Late Chunking")
    print("=" * 50)

    try:
        chunker = Chunker(
            strategy="late",
            chunk_size=500,
            embedding_model="all-MiniLM-L6-v2",
        )
        chunks = chunker.chunk(TEXT)

        print(f"Created {len(chunks)} chunks with late chunking:\n")
        for chunk in chunks:
            has_embedding = "embedding" in chunk.metadata
            print(f"Chunk {chunk.index}: {len(chunk)} chars, has_embedding={has_embedding}")

    except Exception as e:
        print(f"Note: Requires sentence-transformers. Error: {e}")
    print()


def example_6_cluster_based_chunking():
    """Cluster-based semantic chunking."""
    print("=" * 50)
    print("Example 6: Cluster-Based Chunking")
    print("=" * 50)

    try:
        from chunkwise.strategies.semantic import ClusterChunker

        chunker = ClusterChunker(
            chunk_size=500,
            embedding_model="all-MiniLM-L6-v2",
        )
        chunks = chunker.chunk(TEXT)

        print(f"Created {len(chunks)} cluster-based chunks")

    except Exception as e:
        print(f"Note: Requires sentence-transformers. Error: {e}")
    print()


if __name__ == "__main__":
    print("Semantic Chunking Examples")
    print("Requires: pip install sentence-transformers\n")

    example_1_basic_semantic_chunking()
    example_2_custom_threshold()
    example_3_different_embedding_models()
    example_4_with_embeddings_config()
    example_5_late_chunking()
    example_6_cluster_based_chunking()

    print("All semantic chunking examples completed!")
