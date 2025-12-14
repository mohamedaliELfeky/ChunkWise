"""
ChunkWise Agentic Chunking Examples

This file demonstrates LLM-powered chunking strategies.
Requires: pip install openai (or anthropic for Claude)
"""

import os
from chunkwise import Chunker
from chunkwise.config import ChunkConfig, LLMConfig

# Sample text
TEXT = """
The history of computing is rich and varied. In the early days, computers were massive machines that filled entire rooms. The ENIAC, completed in 1945, was one of the first general-purpose electronic computers. It weighed about 30 tons and consumed enormous amounts of electricity.

The invention of the transistor in 1947 changed everything. Transistors were smaller, more reliable, and consumed less power than vacuum tubes. This led to the development of smaller and more powerful computers throughout the 1950s and 1960s.

The microprocessor revolution began in the 1970s with the Intel 4004, the first commercially available microprocessor. This tiny chip contained all the components of a computer's central processing unit. It paved the way for personal computers.

The 1980s saw the rise of personal computing. Companies like Apple and IBM brought computers into homes and offices. The graphical user interface made computers accessible to non-technical users.

The internet transformed computing in the 1990s. What started as a military and academic network became a global communication platform. The World Wide Web made information accessible to anyone with a computer and an internet connection.

Today, we live in an era of cloud computing, artificial intelligence, and mobile devices. Computers are everywhere, from our pockets to our homes to our cars. The future promises even more integration of computing into every aspect of our lives.
"""


def example_1_basic_agentic_chunking():
    """Basic LLM-powered chunking."""
    print("=" * 50)
    print("Example 1: Basic Agentic Chunking")
    print("=" * 50)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Note: Set OPENAI_API_KEY environment variable")
        print("Skipping this example...")
        return

    try:
        chunker = Chunker(
            strategy="agentic",
            chunk_size=300,
            llm_model="gpt-4o-mini",
        )
        chunks = chunker.chunk(TEXT)

        print(f"Created {len(chunks)} AI-determined chunks:\n")
        for chunk in chunks:
            print(f"Chunk {chunk.index}:")
            print(f"  {chunk.content[:80]}...")
            print()

    except Exception as e:
        print(f"Error: {e}")
    print()


def example_2_proposition_chunking():
    """Extract atomic propositions from text."""
    print("=" * 50)
    print("Example 2: Proposition Chunking")
    print("=" * 50)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Note: Set OPENAI_API_KEY environment variable")
        print("Skipping this example...")
        return

    try:
        chunker = Chunker(
            strategy="proposition",
            llm_model="gpt-4o-mini",
        )

        # Use a shorter text for propositions
        short_text = TEXT[:500]
        chunks = chunker.chunk(short_text)

        print(f"Extracted {len(chunks)} proposition chunks:\n")
        for chunk in chunks[:5]:  # Show first 5
            print(f"  - {chunk.content[:100]}...")

    except Exception as e:
        print(f"Error: {e}")
    print()


def example_3_with_anthropic():
    """Using Anthropic's Claude for agentic chunking."""
    print("=" * 50)
    print("Example 3: Agentic Chunking with Claude")
    print("=" * 50)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Note: Set ANTHROPIC_API_KEY environment variable")
        print("Skipping this example...")
        return

    try:
        from chunkwise.strategies.agentic import AgenticChunker

        chunker = AgenticChunker(
            chunk_size=300,
            llm_provider="anthropic",
            llm_model="claude-3-haiku-20240307",
        )
        chunks = chunker.chunk(TEXT)

        print(f"Created {len(chunks)} chunks using Claude")

    except Exception as e:
        print(f"Error: {e}")
    print()


def example_4_custom_llm_config():
    """Using custom LLM configuration."""
    print("=" * 50)
    print("Example 4: Custom LLM Config")
    print("=" * 50)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Note: Set OPENAI_API_KEY environment variable")
        print("Skipping this example...")
        return

    try:
        llm_config = LLMConfig(
            model="gpt-4o-mini",
            provider="openai",
            temperature=0.0,  # Deterministic output
            max_tokens=4096,
        )

        config = ChunkConfig(
            strategy="agentic",
            chunk_size=300,
            llm_config=llm_config,
        )

        chunker = Chunker(config=config)
        chunks = chunker.chunk(TEXT)

        print(f"Created {len(chunks)} chunks with custom LLM config")

    except Exception as e:
        print(f"Error: {e}")
    print()


def example_5_hybrid_with_agentic():
    """Combining agentic with other strategies."""
    print("=" * 50)
    print("Example 5: Hybrid with Agentic Fallback")
    print("=" * 50)

    try:
        # Try agentic first, fall back to recursive
        chunker = Chunker(
            strategy="hybrid",
            chunk_size=300,
        )

        # Configure hybrid mode (this will use fallback)
        from chunkwise.strategies.hybrid import HybridChunker

        hybrid = HybridChunker(
            chunk_size=300,
            strategies=["agentic", "recursive"],
            mode="fallback",
        )
        chunks = hybrid.chunk(TEXT)

        print(f"Created {len(chunks)} chunks with hybrid strategy")
        for chunk in chunks[:2]:
            strategy = chunk.metadata.get("strategy", "unknown")
            print(f"  Chunk {chunk.index}: strategy={strategy}")

    except Exception as e:
        print(f"Error: {e}")
    print()


if __name__ == "__main__":
    print("Agentic Chunking Examples")
    print("Requires: pip install openai (or anthropic)\n")

    example_1_basic_agentic_chunking()
    example_2_proposition_chunking()
    example_3_with_anthropic()
    example_4_custom_llm_config()
    example_5_hybrid_with_agentic()

    print("All agentic chunking examples completed!")
