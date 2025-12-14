"""
Chunk Data Model

Defines the Chunk dataclass that represents a single chunk of text
with associated metadata.
"""

from dataclasses import dataclass, field
from typing import Any, Optional, List
import json


@dataclass
class Chunk:
    """
    Represents a single chunk of text with metadata.

    Attributes:
        content: The actual text content of the chunk
        index: The position of this chunk in the sequence (0-indexed)
        start_char: Starting character position in the original text
        end_char: Ending character position in the original text
        metadata: Additional metadata about the chunk

    Example:
        >>> chunk = Chunk(
        ...     content="Hello world",
        ...     index=0,
        ...     start_char=0,
        ...     end_char=11,
        ...     metadata={"language": "en", "tokens": 2}
        ... )
        >>> print(chunk)
        Chunk(index=0, chars=11, content='Hello world...')
    """

    content: str
    index: int
    start_char: int
    end_char: int
    metadata: dict = field(default_factory=dict)

    @property
    def length(self) -> int:
        """Return the character length of the chunk content."""
        return len(self.content)

    @property
    def token_count(self) -> Optional[int]:
        """Return the token count if available in metadata."""
        return self.metadata.get("token_count")

    @property
    def language(self) -> Optional[str]:
        """Return the detected language if available in metadata."""
        return self.metadata.get("language")

    @property
    def embedding(self) -> Optional[List[float]]:
        """Return the embedding vector if available in metadata."""
        return self.metadata.get("embedding")

    def to_dict(self) -> dict:
        """Convert chunk to dictionary representation."""
        return {
            "content": self.content,
            "index": self.index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "length": self.length,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Convert chunk to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        """Create a Chunk instance from a dictionary."""
        return cls(
            content=data["content"],
            index=data["index"],
            start_char=data["start_char"],
            end_char=data["end_char"],
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "Chunk":
        """Create a Chunk instance from a JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def __repr__(self) -> str:
        """Return a concise string representation."""
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        preview = preview.replace("\n", "\\n")
        return f"Chunk(index={self.index}, chars={self.length}, content='{preview}')"

    def __str__(self) -> str:
        """Return the chunk content as string."""
        return self.content

    def __len__(self) -> int:
        """Return the length of the chunk content."""
        return self.length

    def __eq__(self, other: Any) -> bool:
        """Check equality based on content and position."""
        if not isinstance(other, Chunk):
            return False
        return (
            self.content == other.content
            and self.index == other.index
            and self.start_char == other.start_char
            and self.end_char == other.end_char
        )

    def __hash__(self) -> int:
        """Return hash based on content and position."""
        return hash((self.content, self.index, self.start_char, self.end_char))


@dataclass
class ChunkBatch:
    """
    Represents a batch of chunks from a single document.

    Attributes:
        chunks: List of Chunk objects
        source: Optional source identifier (filename, URL, etc.)
        total_chars: Total characters in the original document
        metadata: Additional metadata about the batch
    """

    chunks: List[Chunk]
    source: Optional[str] = None
    total_chars: int = 0
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        """Return the number of chunks in the batch."""
        return len(self.chunks)

    def __iter__(self):
        """Iterate over chunks."""
        return iter(self.chunks)

    def __getitem__(self, index: int) -> Chunk:
        """Get chunk by index."""
        return self.chunks[index]

    @property
    def total_tokens(self) -> Optional[int]:
        """Return total token count if available."""
        token_counts = [c.token_count for c in self.chunks if c.token_count is not None]
        return sum(token_counts) if token_counts else None

    def to_texts(self) -> List[str]:
        """Return list of chunk contents as strings."""
        return [chunk.content for chunk in self.chunks]

    def to_dict(self) -> dict:
        """Convert batch to dictionary representation."""
        return {
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "source": self.source,
            "total_chars": self.total_chars,
            "metadata": self.metadata,
        }
