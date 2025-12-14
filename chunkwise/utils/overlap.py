"""
Overlap Utilities

Functions for calculating and managing chunk overlap.
"""

from typing import List, Optional, Callable
from chunkwise.chunk import Chunk


def calculate_overlap(
    chunk_size: int,
    overlap_ratio: Optional[float] = None,
    overlap_tokens: Optional[int] = None,
) -> int:
    """
    Calculate overlap size based on ratio or absolute tokens.

    Args:
        chunk_size: Size of each chunk
        overlap_ratio: Overlap as ratio of chunk_size (0.0 to 0.5)
        overlap_tokens: Absolute number of overlap tokens

    Returns:
        Calculated overlap size
    """
    if overlap_tokens is not None:
        return min(overlap_tokens, chunk_size // 2)

    if overlap_ratio is not None:
        ratio = min(max(overlap_ratio, 0.0), 0.5)
        return int(chunk_size * ratio)

    return 0


def get_overlap_text(
    text: str,
    position: int,
    overlap_size: int,
    direction: str = "before",
) -> str:
    """
    Get overlap text from a position.

    Args:
        text: Full text
        position: Position to get overlap from
        overlap_size: Size of overlap
        direction: "before" for text before position, "after" for text after

    Returns:
        Overlap text
    """
    if direction == "before":
        start = max(0, position - overlap_size)
        return text[start:position]
    else:
        end = min(len(text), position + overlap_size)
        return text[position:end]


def add_overlap_to_chunks(
    chunks: List[Chunk],
    original_text: str,
    overlap_size: int,
    overlap_position: str = "both",
) -> List[Chunk]:
    """
    Add overlap to existing chunks by extending their content.

    Args:
        chunks: List of chunks
        original_text: Original full text
        overlap_size: Size of overlap to add
        overlap_position: "start", "end", or "both"

    Returns:
        Chunks with overlap added
    """
    if not chunks or overlap_size <= 0:
        return chunks

    result = []

    for i, chunk in enumerate(chunks):
        new_content = chunk.content
        new_start = chunk.start_char
        new_end = chunk.end_char

        # Add overlap from previous chunk (prepend)
        if overlap_position in ("start", "both") and i > 0:
            overlap_start = max(0, chunk.start_char - overlap_size)
            if overlap_start < chunk.start_char:
                prefix = original_text[overlap_start:chunk.start_char]
                new_content = prefix + new_content
                new_start = overlap_start

        # Add overlap from next chunk (append)
        if overlap_position in ("end", "both") and i < len(chunks) - 1:
            overlap_end = min(len(original_text), chunk.end_char + overlap_size)
            if overlap_end > chunk.end_char:
                suffix = original_text[chunk.end_char:overlap_end]
                new_content = new_content + suffix
                new_end = overlap_end

        result.append(
            Chunk(
                content=new_content,
                index=chunk.index,
                start_char=new_start,
                end_char=new_end,
                metadata={
                    **chunk.metadata,
                    "has_overlap": True,
                    "overlap_size": overlap_size,
                },
            )
        )

    return result


def sliding_window_positions(
    text_length: int,
    window_size: int,
    step_size: int,
) -> List[tuple]:
    """
    Calculate start/end positions for sliding window chunking.

    Args:
        text_length: Length of text
        window_size: Size of each window
        step_size: Step between windows (window_size - overlap)

    Returns:
        List of (start, end) tuples
    """
    if text_length <= 0 or window_size <= 0 or step_size <= 0:
        return []

    positions = []
    start = 0

    while start < text_length:
        end = min(start + window_size, text_length)
        positions.append((start, end))

        if end >= text_length:
            break

        start += step_size

    return positions


def calculate_overlap_ratio(chunk1: Chunk, chunk2: Chunk) -> float:
    """
    Calculate the overlap ratio between two chunks.

    Args:
        chunk1: First chunk
        chunk2: Second chunk

    Returns:
        Overlap ratio (0.0 to 1.0)
    """
    # Character position overlap
    overlap_start = max(chunk1.start_char, chunk2.start_char)
    overlap_end = min(chunk1.end_char, chunk2.end_char)

    if overlap_start >= overlap_end:
        return 0.0

    overlap_length = overlap_end - overlap_start
    min_length = min(len(chunk1), len(chunk2))

    return overlap_length / max(min_length, 1)


def merge_overlapping_chunks(
    chunks: List[Chunk],
    min_overlap_ratio: float = 0.5,
) -> List[Chunk]:
    """
    Merge chunks that have significant overlap.

    Args:
        chunks: List of chunks
        min_overlap_ratio: Minimum overlap ratio to trigger merge

    Returns:
        List of merged chunks
    """
    if not chunks:
        return chunks

    # Sort by start position
    sorted_chunks = sorted(chunks, key=lambda c: c.start_char)

    merged = [sorted_chunks[0]]

    for chunk in sorted_chunks[1:]:
        last = merged[-1]
        overlap = calculate_overlap_ratio(last, chunk)

        if overlap >= min_overlap_ratio:
            # Merge chunks
            new_content = last.content
            if chunk.end_char > last.end_char:
                # Add non-overlapping part from chunk
                new_content += chunk.content[last.end_char - chunk.start_char:]

            merged[-1] = Chunk(
                content=new_content,
                index=last.index,
                start_char=last.start_char,
                end_char=max(last.end_char, chunk.end_char),
                metadata={**last.metadata, "merged": True},
            )
        else:
            chunk.index = len(merged)
            merged.append(chunk)

    return merged
