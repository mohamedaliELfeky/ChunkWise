"""
Format-Specific Chunking Strategies

Chunkers for JSON, LaTeX, Regex patterns, and other specific formats.
"""

import re
import json
from typing import List, Optional, Pattern

from chunkwise.base import BaseChunker
from chunkwise.chunk import Chunk
from chunkwise.config import ChunkConfig


class JSONChunker(BaseChunker):
    """
    JSON-aware chunking that preserves structure.

    Splits JSON documents while keeping objects intact.

    Example:
        >>> chunker = JSONChunker(chunk_size=1000)
        >>> chunks = chunker.chunk(json_string)
    """

    def __init__(
        self,
        config: Optional[ChunkConfig] = None,
        max_depth: int = 2,
        **kwargs,
    ):
        """
        Initialize JSON chunker.

        Args:
            config: ChunkConfig instance
            max_depth: Maximum depth to split at
            **kwargs: Override config parameters
        """
        super().__init__(config, **kwargs)
        self.max_depth = max_depth

    def _chunk_text(self, text: str) -> List[Chunk]:
        """
        Chunk JSON text.

        Args:
            text: JSON string

        Returns:
            List of Chunk objects
        """
        if not text:
            return []

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Not valid JSON, fall back to recursive
            from chunkwise.strategies.recursive import RecursiveChunker

            return RecursiveChunker(config=self.config)._chunk_text(text)

        # Extract chunks from JSON structure
        json_chunks = self._extract_json_chunks(data, depth=0)

        # Convert to Chunk objects
        chunks = []
        char_pos = 0

        for i, json_chunk in enumerate(json_chunks):
            content = json.dumps(json_chunk, ensure_ascii=False, indent=2)

            chunks.append(
                Chunk(
                    content=content,
                    index=i,
                    start_char=char_pos,
                    end_char=char_pos + len(content),
                    metadata={
                        "strategy": "json",
                        "json_type": type(json_chunk).__name__,
                    },
                )
            )
            char_pos += len(content) + 1

        return chunks

    def _extract_json_chunks(self, data, depth: int = 0) -> List:
        """
        Recursively extract chunks from JSON.

        Args:
            data: JSON data (dict, list, or primitive)
            depth: Current depth

        Returns:
            List of JSON chunks
        """
        chunks = []

        if isinstance(data, dict):
            if depth >= self.max_depth:
                # Return whole object
                chunks.append(data)
            else:
                # Split by keys
                for key, value in data.items():
                    if isinstance(value, (dict, list)) and depth < self.max_depth:
                        sub_chunks = self._extract_json_chunks(value, depth + 1)
                        for sub in sub_chunks:
                            chunks.append({key: sub})
                    else:
                        chunks.append({key: value})

        elif isinstance(data, list):
            if depth >= self.max_depth:
                chunks.append(data)
            else:
                # Split by items
                for item in data:
                    if isinstance(item, (dict, list)):
                        sub_chunks = self._extract_json_chunks(item, depth + 1)
                        chunks.extend(sub_chunks)
                    else:
                        chunks.append(item)
        else:
            chunks.append(data)

        # Merge small chunks
        return self._merge_small_json_chunks(chunks)

    def _merge_small_json_chunks(self, chunks: List) -> List:
        """Merge chunks that are too small."""
        merged = []
        current_batch = []
        current_size = 0

        for chunk in chunks:
            chunk_str = json.dumps(chunk, ensure_ascii=False)
            chunk_size = len(chunk_str)

            if current_size + chunk_size <= self.config.chunk_size:
                current_batch.append(chunk)
                current_size += chunk_size
            else:
                if current_batch:
                    if len(current_batch) == 1:
                        merged.append(current_batch[0])
                    else:
                        merged.append(current_batch)
                current_batch = [chunk]
                current_size = chunk_size

        if current_batch:
            if len(current_batch) == 1:
                merged.append(current_batch[0])
            else:
                merged.append(current_batch)

        return merged


class LaTeXChunker(BaseChunker):
    """
    LaTeX-aware chunking that respects document structure.

    Splits on sections, subsections, chapters, etc.

    Example:
        >>> chunker = LaTeXChunker(chunk_size=2000)
        >>> chunks = chunker.chunk(latex_source)
    """

    # LaTeX section patterns
    SECTION_PATTERNS = [
        r"\\chapter\{([^}]*)\}",
        r"\\section\{([^}]*)\}",
        r"\\subsection\{([^}]*)\}",
        r"\\subsubsection\{([^}]*)\}",
        r"\\paragraph\{([^}]*)\}",
    ]

    def __init__(
        self,
        config: Optional[ChunkConfig] = None,
        include_preamble: bool = False,
        **kwargs,
    ):
        """
        Initialize LaTeX chunker.

        Args:
            config: ChunkConfig instance
            include_preamble: Include document preamble in first chunk
            **kwargs: Override config parameters
        """
        super().__init__(config, **kwargs)
        self.include_preamble = include_preamble
        self._combined_pattern = re.compile(
            "|".join(f"({p})" for p in self.SECTION_PATTERNS),
            re.MULTILINE,
        )

    def _chunk_text(self, text: str) -> List[Chunk]:
        """
        Chunk LaTeX text by sections.

        Args:
            text: LaTeX source

        Returns:
            List of Chunk objects
        """
        if not text:
            return []

        # Find all section boundaries
        sections = self._find_sections(text)

        if not sections:
            # No sections found, use recursive
            from chunkwise.strategies.recursive import RecursiveChunker

            return RecursiveChunker(config=self.config)._chunk_text(text)

        return self._sections_to_chunks(sections, text)

    def _find_sections(self, text: str) -> List[dict]:
        """
        Find LaTeX sections.

        Args:
            text: LaTeX source

        Returns:
            List of section dicts
        """
        sections = []
        matches = list(self._combined_pattern.finditer(text))

        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

            # Determine section type and title
            section_type = "section"
            title = ""
            for j, pattern in enumerate(self.SECTION_PATTERNS):
                m = re.match(pattern, match.group())
                if m:
                    types = ["chapter", "section", "subsection", "subsubsection", "paragraph"]
                    section_type = types[j] if j < len(types) else "section"
                    title = m.group(1) if m.groups() else ""
                    break

            sections.append({
                "type": section_type,
                "title": title,
                "content": text[start:end],
                "start": start,
                "end": end,
            })

        # Handle content before first section
        if matches and matches[0].start() > 0:
            preamble = text[: matches[0].start()]
            if self.include_preamble and preamble.strip():
                sections.insert(0, {
                    "type": "preamble",
                    "title": "Preamble",
                    "content": preamble,
                    "start": 0,
                    "end": matches[0].start(),
                })

        return sections

    def _sections_to_chunks(self, sections: List[dict], text: str) -> List[Chunk]:
        """Convert sections to chunks."""
        chunks = []

        for i, section in enumerate(sections):
            content = section["content"]

            # Split large sections
            if len(content) > self.config.chunk_size:
                from chunkwise.strategies.recursive import RecursiveChunker

                sub_chunker = RecursiveChunker(config=self.config)
                sub_chunks = sub_chunker._chunk_text(content)

                for j, sub in enumerate(sub_chunks):
                    sub.index = len(chunks)
                    sub.start_char = section["start"] + sub.start_char
                    sub.end_char = section["start"] + sub.end_char
                    sub.metadata.update({
                        "strategy": "latex",
                        "section_type": section["type"],
                        "section_title": section["title"],
                        "is_split": True,
                    })
                    chunks.append(sub)
            else:
                chunks.append(
                    Chunk(
                        content=content,
                        index=len(chunks),
                        start_char=section["start"],
                        end_char=section["end"],
                        metadata={
                            "strategy": "latex",
                            "section_type": section["type"],
                            "section_title": section["title"],
                        },
                    )
                )

        return chunks


class RegexChunker(BaseChunker):
    """
    Regex-based chunking using custom patterns.

    Split text based on user-defined regex patterns.

    Example:
        >>> # Split on timestamps
        >>> chunker = RegexChunker(
        ...     pattern=r"\\d{4}-\\d{2}-\\d{2}",
        ...     chunk_size=500,
        ... )
        >>> chunks = chunker.chunk(log_file)
    """

    def __init__(
        self,
        config: Optional[ChunkConfig] = None,
        pattern: str = r"\n\n+",
        keep_separator: bool = True,
        separator_position: str = "start",
        **kwargs,
    ):
        """
        Initialize Regex chunker.

        Args:
            config: ChunkConfig instance
            pattern: Regex pattern to split on
            keep_separator: Whether to keep the separator
            separator_position: "start" or "end" - where to attach separator
            **kwargs: Override config parameters
        """
        super().__init__(config, **kwargs)
        self.pattern = pattern
        self.keep_separator = keep_separator
        self.separator_position = separator_position
        self._compiled_pattern = re.compile(pattern)

    def _chunk_text(self, text: str) -> List[Chunk]:
        """
        Chunk text using regex pattern.

        Args:
            text: Input text

        Returns:
            List of Chunk objects
        """
        if not text:
            return []

        # Split by pattern
        if self.keep_separator:
            parts = self._split_with_separator(text)
        else:
            parts = self._compiled_pattern.split(text)

        # Filter empty parts
        parts = [p for p in parts if p.strip()]

        if not parts:
            return [
                Chunk(
                    content=text,
                    index=0,
                    start_char=0,
                    end_char=len(text),
                    metadata={"strategy": "regex"},
                )
            ]

        # Merge small parts and split large ones
        return self._process_parts(parts, text)

    def _split_with_separator(self, text: str) -> List[str]:
        """Split keeping separator attached to chunks."""
        parts = []
        last_end = 0

        for match in self._compiled_pattern.finditer(text):
            if self.separator_position == "end":
                # Attach separator to end of previous chunk
                if last_end < match.end():
                    parts.append(text[last_end:match.end()])
                last_end = match.end()
            else:
                # Attach separator to start of next chunk
                if last_end < match.start():
                    parts.append(text[last_end:match.start()])
                last_end = match.start()

        # Remaining text
        if last_end < len(text):
            parts.append(text[last_end:])

        return parts

    def _process_parts(self, parts: List[str], text: str) -> List[Chunk]:
        """Process parts into properly sized chunks."""
        chunks = []
        current_content = ""
        current_start = 0
        char_pos = 0

        for part in parts:
            if len(current_content) + len(part) <= self.config.chunk_size:
                if not current_content:
                    current_start = char_pos
                current_content += part
            else:
                if current_content:
                    chunks.append(
                        Chunk(
                            content=current_content,
                            index=len(chunks),
                            start_char=current_start,
                            end_char=current_start + len(current_content),
                            metadata={"strategy": "regex", "pattern": self.pattern},
                        )
                    )
                current_content = part
                current_start = char_pos

            char_pos = text.find(part, char_pos) + len(part)

        # Last chunk
        if current_content:
            chunks.append(
                Chunk(
                    content=current_content,
                    index=len(chunks),
                    start_char=current_start,
                    end_char=current_start + len(current_content),
                    metadata={"strategy": "regex", "pattern": self.pattern},
                )
            )

        return chunks
