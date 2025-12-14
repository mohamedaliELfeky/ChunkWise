"""
Text Utilities

Common text processing functions used across chunking strategies.
"""

import re
from typing import List, Optional, Callable


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text.

    - Replaces multiple spaces with single space
    - Preserves newlines but normalizes multiple newlines
    - Strips leading/trailing whitespace

    Args:
        text: Input text

    Returns:
        Text with normalized whitespace
    """
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Replace multiple spaces (but not newlines) with single space
    text = re.sub(r"[^\S\n]+", " ", text)

    # Replace 3+ newlines with 2 newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def clean_text(text: str, remove_urls: bool = False, remove_emails: bool = False) -> str:
    """
    Clean text by removing unwanted elements.

    Args:
        text: Input text
        remove_urls: Remove URLs from text
        remove_emails: Remove email addresses from text

    Returns:
        Cleaned text
    """
    if remove_urls:
        # Remove URLs
        text = re.sub(r"https?://\S+|www\.\S+", "", text)

    if remove_emails:
        # Remove email addresses
        text = re.sub(r"\S+@\S+\.\S+", "", text)

    # Remove null bytes
    text = text.replace("\x00", "")

    # Remove other control characters except newlines and tabs
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    return text


def split_by_separators(
    text: str,
    separators: List[str],
    keep_separator: bool = True,
    separator_position: str = "end",
) -> List[str]:
    """
    Split text by a list of separators in order of priority.

    Tries each separator in order until one produces splits.

    Args:
        text: Text to split
        separators: List of separators to try, in order of preference
        keep_separator: Whether to keep the separator in the splits
        separator_position: Where to attach separator: "start", "end", or "none"

    Returns:
        List of text splits
    """
    if not text:
        return []

    if not separators:
        return [text]

    for separator in separators:
        if not separator:
            # Empty separator means split into characters
            return list(text)

        if separator in text:
            if keep_separator and separator_position != "none":
                # Split and keep separator
                if separator_position == "end":
                    # Separator at end of each chunk
                    pattern = f"({re.escape(separator)})"
                    parts = re.split(pattern, text)
                    # Combine parts with their separators
                    result = []
                    for i in range(0, len(parts) - 1, 2):
                        if i + 1 < len(parts):
                            result.append(parts[i] + parts[i + 1])
                        else:
                            result.append(parts[i])
                    if len(parts) % 2 == 1 and parts[-1]:
                        result.append(parts[-1])
                    return [r for r in result if r]
                else:
                    # Separator at start of each chunk
                    pattern = f"({re.escape(separator)})"
                    parts = re.split(pattern, text)
                    result = []
                    for i, part in enumerate(parts):
                        if i == 0 and part:
                            result.append(part)
                        elif i > 0 and i % 2 == 1:
                            # This is a separator
                            if i + 1 < len(parts):
                                result.append(part + parts[i + 1])
                    return [r for r in result if r]
            else:
                # Simple split without keeping separator
                return [s for s in text.split(separator) if s]

    # No separator found, return as-is
    return [text]


def merge_splits(
    splits: List[str],
    max_size: int,
    overlap: int = 0,
    length_function: Optional[Callable[[str], int]] = None,
    separator: str = " ",
) -> List[str]:
    """
    Merge small splits into chunks of target size.

    Args:
        splits: List of text splits to merge
        max_size: Maximum size for each merged chunk
        overlap: Number of characters/tokens to overlap
        length_function: Function to calculate length (default: len)
        separator: String to join splits with

    Returns:
        List of merged chunks
    """
    if not splits:
        return []

    if length_function is None:
        length_function = len

    chunks = []
    current_chunk: List[str] = []
    current_length = 0

    for split in splits:
        split_length = length_function(split)

        # If single split exceeds max_size, add it as-is
        if split_length > max_size:
            if current_chunk:
                chunks.append(separator.join(current_chunk))
                current_chunk = []
                current_length = 0
            chunks.append(split)
            continue

        # Check if adding this split would exceed max_size
        test_length = current_length + split_length
        if current_chunk:
            test_length += length_function(separator)

        if test_length <= max_size:
            current_chunk.append(split)
            current_length = test_length
        else:
            # Save current chunk and start new one
            if current_chunk:
                chunks.append(separator.join(current_chunk))

            # Handle overlap
            if overlap > 0 and current_chunk:
                # Get overlap from end of previous chunk
                overlap_splits = _get_overlap_splits(
                    current_chunk, overlap, length_function, separator
                )
                current_chunk = overlap_splits + [split]
                current_length = sum(length_function(s) for s in current_chunk)
                current_length += length_function(separator) * (len(current_chunk) - 1)
            else:
                current_chunk = [split]
                current_length = split_length

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(separator.join(current_chunk))

    return chunks


def _get_overlap_splits(
    splits: List[str],
    overlap: int,
    length_function: Callable[[str], int],
    separator: str,
) -> List[str]:
    """
    Get splits from the end that fit within overlap size.

    Args:
        splits: Current chunk splits
        overlap: Target overlap size
        length_function: Function to calculate length
        separator: Separator between splits

    Returns:
        List of splits for overlap
    """
    if not splits:
        return []

    result = []
    current_length = 0

    for split in reversed(splits):
        split_length = length_function(split)
        test_length = current_length + split_length
        if result:
            test_length += length_function(separator)

        if test_length <= overlap:
            result.insert(0, split)
            current_length = test_length
        else:
            break

    return result


def find_breakpoint(
    text: str,
    max_position: int,
    break_chars: str = " \n\t.,;:!?",
) -> int:
    """
    Find a good breakpoint in text before max_position.

    Looks for word boundaries, sentence boundaries, etc.

    Args:
        text: Text to search
        max_position: Maximum position for break
        break_chars: Characters that are good breakpoints

    Returns:
        Position of best breakpoint
    """
    if max_position >= len(text):
        return len(text)

    # Search backwards for a good break character
    for i in range(max_position, max(0, max_position - 100), -1):
        if text[i] in break_chars:
            return i + 1

    # No good breakpoint found, break at max_position
    return max_position


def count_words(text: str) -> int:
    """
    Count words in text.

    Args:
        text: Input text

    Returns:
        Word count
    """
    return len(text.split())


def count_sentences(text: str, sentence_enders: str = ".!?؟۔") -> int:
    """
    Count sentences in text.

    Args:
        text: Input text
        sentence_enders: Characters that end sentences

    Returns:
        Sentence count
    """
    pattern = f"[{re.escape(sentence_enders)}]"
    sentences = re.split(pattern, text)
    return len([s for s in sentences if s.strip()])


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text to max_length, adding suffix if truncated.

    Args:
        text: Input text
        max_length: Maximum length
        suffix: Suffix to add if truncated

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text

    truncate_at = max_length - len(suffix)
    # Try to truncate at word boundary
    breakpoint = find_breakpoint(text, truncate_at)
    return text[:breakpoint].rstrip() + suffix
