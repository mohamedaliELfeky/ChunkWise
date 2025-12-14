"""
ChunkWise RAG Pipeline Example

This file demonstrates using ChunkWise in a Retrieval-Augmented Generation pipeline.
"""

from typing import List, Optional
from chunkwise import Chunker
from chunkwise.chunk import Chunk

# Sample documents for RAG
DOCUMENTS = [
    """
    Python is a high-level, interpreted programming language known for its readability and versatility.
    Created by Guido van Rossum and first released in 1991, Python has become one of the most popular
    programming languages in the world. It supports multiple programming paradigms including procedural,
    object-oriented, and functional programming. Python's extensive standard library and large ecosystem
    of third-party packages make it suitable for a wide range of applications.
    """,
    """
    Machine learning is transforming how we interact with technology. Deep learning models can now
    recognize images, understand speech, and generate human-like text. These advances are powered by
    neural networks with millions or billions of parameters, trained on vast amounts of data.
    Popular frameworks like TensorFlow and PyTorch have made it easier for developers to build and
    deploy machine learning models.
    """,
    """
    Cloud computing has revolutionized how businesses deploy and scale applications. Services like
    AWS, Google Cloud, and Azure provide on-demand computing resources, storage, and databases.
    Companies can now launch applications globally within minutes, paying only for the resources
    they use. This has enabled startups to compete with established enterprises and has accelerated
    the pace of innovation.
    """,
]


class SimpleVectorStore:
    """A simple in-memory vector store for demonstration."""

    def __init__(self):
        self.chunks: List[Chunk] = []
        self.embeddings: List[List[float]] = []

    def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]):
        """Add chunks with their embeddings."""
        self.chunks.extend(chunks)
        self.embeddings.extend(embeddings)

    def search(self, query_embedding: List[float], top_k: int = 3) -> List[Chunk]:
        """Search for most similar chunks."""
        import numpy as np

        if not self.embeddings:
            return []

        # Calculate cosine similarity
        query = np.array(query_embedding)
        similarities = []

        for emb in self.embeddings:
            emb_arr = np.array(emb)
            sim = np.dot(query, emb_arr) / (np.linalg.norm(query) * np.linalg.norm(emb_arr))
            similarities.append(sim)

        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        return [self.chunks[i] for i in top_indices]


def example_rag_pipeline():
    """Complete RAG pipeline example."""
    print("=" * 60)
    print("RAG Pipeline with ChunkWise")
    print("=" * 60)

    try:
        from chunkwise.embeddings.sentence_transformers_embed import (
            SentenceTransformersEmbedding,
        )

        # Step 1: Initialize components
        print("\n1. Initializing components...")
        chunker = Chunker(
            strategy="semantic",
            chunk_size=300,
            chunk_overlap=50,
        )
        embedder = SentenceTransformersEmbedding(model_name="all-MiniLM-L6-v2")
        vector_store = SimpleVectorStore()

        # Step 2: Process documents
        print("2. Processing documents...")
        all_chunks = []
        for i, doc in enumerate(DOCUMENTS):
            chunks = chunker.chunk(doc)
            for chunk in chunks:
                chunk.metadata["source_doc"] = i
            all_chunks.extend(chunks)
            print(f"   Document {i + 1}: {len(chunks)} chunks")

        # Step 3: Generate embeddings
        print("3. Generating embeddings...")
        texts = [chunk.content for chunk in all_chunks]
        embeddings = embedder.embed(texts)

        # Step 4: Store in vector store
        print("4. Storing in vector store...")
        vector_store.add_chunks(all_chunks, embeddings.tolist())
        print(f"   Total chunks stored: {len(all_chunks)}")

        # Step 5: Query the system
        print("\n5. Querying the system...")
        queries = [
            "What is Python programming language?",
            "How does machine learning work?",
            "What is cloud computing?",
        ]

        for query in queries:
            print(f"\n   Query: '{query}'")
            query_embedding = embedder.embed_single(query)
            results = vector_store.search(query_embedding.tolist(), top_k=2)

            print("   Results:")
            for j, result in enumerate(results):
                source = result.metadata.get("source_doc", "unknown")
                print(f"     {j + 1}. [Doc {source + 1}] {result.content[:80]}...")

    except ImportError:
        print("\nNote: This example requires sentence-transformers")
        print("Install with: pip install sentence-transformers")

    print("\n" + "=" * 60)


def example_different_strategies_for_rag():
    """Compare different chunking strategies for RAG."""
    print("=" * 60)
    print("Comparing Chunking Strategies for RAG")
    print("=" * 60)

    strategies = ["recursive", "sentence", "semantic"]
    doc = DOCUMENTS[0]  # Use first document

    for strategy in strategies:
        try:
            chunker = Chunker(strategy=strategy, chunk_size=200)
            chunks = chunker.chunk(doc)

            avg_size = sum(len(c.content) for c in chunks) / len(chunks)
            print(f"\n{strategy.upper()}:")
            print(f"  Chunks: {len(chunks)}")
            print(f"  Avg size: {avg_size:.0f} chars")

            # Show chunk quality indicators
            for i, chunk in enumerate(chunks[:2]):
                # Check if chunk is a complete thought
                ends_properly = chunk.content.rstrip()[-1] in ".!?"
                print(f"  Chunk {i}: {len(chunk)} chars, ends_properly={ends_properly}")

        except Exception as e:
            print(f"\n{strategy.upper()}: Error - {e}")

    print("\n" + "=" * 60)


def example_batch_processing():
    """Batch processing multiple documents."""
    print("=" * 60)
    print("Batch Processing Documents")
    print("=" * 60)

    chunker = Chunker(strategy="recursive", chunk_size=300)

    # Process all documents at once
    batches = chunker.chunk_documents(DOCUMENTS)

    print(f"\nProcessed {len(batches)} documents:")
    for i, batch in enumerate(batches):
        print(f"\n  Document {i + 1}:")
        print(f"    Chunks: {len(batch.chunks)}")
        print(f"    Total chars: {batch.total_chars}")

        for chunk in batch.chunks:
            print(f"    - Chunk {chunk.index}: {len(chunk)} chars")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    example_rag_pipeline()
    example_different_strategies_for_rag()
    example_batch_processing()

    print("\nAll RAG pipeline examples completed!")
