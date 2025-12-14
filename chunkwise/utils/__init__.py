"""
ChunkWise Utilities

Helper functions for text processing, overlap calculation, and metrics.
"""

from chunkwise.utils.text import (
    normalize_whitespace,
    clean_text,
    split_by_separators,
    merge_splits,
)
from chunkwise.utils.overlap import (
    calculate_overlap,
    add_overlap_to_chunks,
    get_overlap_text,
)

__all__ = [
    "normalize_whitespace",
    "clean_text",
    "split_by_separators",
    "merge_splits",
    "calculate_overlap",
    "add_overlap_to_chunks",
    "get_overlap_text",
]
