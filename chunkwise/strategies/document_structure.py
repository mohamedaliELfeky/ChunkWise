"""
Document Structure-Based Chunking Strategies

Chunkers that respect document structure like Markdown, HTML, and code.
"""

import re
from typing import List, Optional, Tuple
from dataclasses import dataclass

from chunkwise.base import BaseChunker
from chunkwise.chunk import Chunk
from chunkwise.config import ChunkConfig


@dataclass
class Section:
    """Represents a document section."""

    title: str
    content: str
    level: int
    start_char: int
    end_char: int


class MarkdownChunker(BaseChunker):
    """
    Split Markdown documents respecting header hierarchy.

    Creates chunks that preserve document structure by keeping content
    under headers together. Falls back to other splitting methods for
    large sections.

    Example:
        >>> chunker = MarkdownChunker(chunk_size=1000)
        >>> chunks = chunker.chunk(markdown_text)

        >>> # Include header in each chunk
        >>> chunker = MarkdownChunker(chunk_size=1000, include_header=True)
    """

    # Markdown header pattern
    HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    def __init__(
        self,
        config: Optional[ChunkConfig] = None,
        include_header: bool = True,
        max_header_level: int = 6,
        **kwargs,
    ):
        """
        Initialize Markdown chunker.

        Args:
            config: ChunkConfig instance
            include_header: Include section header in each chunk
            max_header_level: Maximum header level to split on (1-6)
            **kwargs: Override config parameters
        """
        super().__init__(config, **kwargs)
        self.include_header = include_header
        self.max_header_level = max_header_level

    def _chunk_text(self, text: str) -> List[Chunk]:
        """
        Split Markdown by sections.

        Args:
            text: Markdown text

        Returns:
            List of Chunk objects
        """
        if not text:
            return []

        # Parse sections
        sections = self._parse_sections(text)

        if not sections:
            # No headers found, use recursive chunking
            from chunkwise.strategies.recursive import RecursiveChunker

            recursive = RecursiveChunker(config=self.config)
            return recursive._chunk_text(text)

        # Process sections into chunks
        return self._process_sections(sections, text)

    def _parse_sections(self, text: str) -> List[Section]:
        """
        Parse Markdown into sections.

        Args:
            text: Markdown text

        Returns:
            List of Section objects
        """
        sections = []
        matches = list(self.HEADER_PATTERN.finditer(text))

        if not matches:
            return []

        for i, match in enumerate(matches):
            level = len(match.group(1))
            title = match.group(2).strip()
            start = match.start()

            # Content ends at next header or end of text
            if i + 1 < len(matches):
                end = matches[i + 1].start()
            else:
                end = len(text)

            content = text[start:end]

            sections.append(
                Section(
                    title=title,
                    content=content,
                    level=level,
                    start_char=start,
                    end_char=end,
                )
            )

        return sections

    def _process_sections(self, sections: List[Section], text: str) -> List[Chunk]:
        """
        Process sections into chunks.

        Args:
            sections: List of sections
            text: Original text

        Returns:
            List of Chunk objects
        """
        chunks = []
        index = 0

        for section in sections:
            if len(section.content) <= self.config.chunk_size:
                # Section fits in one chunk
                chunks.append(
                    Chunk(
                        content=section.content.strip(),
                        index=index,
                        start_char=section.start_char,
                        end_char=section.end_char,
                        metadata={
                            "strategy": "markdown",
                            "section_title": section.title,
                            "header_level": section.level,
                        },
                    )
                )
                index += 1
            else:
                # Split large section
                header = f"{'#' * section.level} {section.title}\n\n"
                body = section.content[len(header) - 2 :]  # Remove header

                # Use recursive chunking for body
                from chunkwise.strategies.recursive import RecursiveChunker

                body_config = ChunkConfig(
                    chunk_size=self.config.chunk_size - len(header) if self.include_header else self.config.chunk_size,
                    chunk_overlap=self.config.chunk_overlap,
                )
                recursive = RecursiveChunker(config=body_config)
                body_chunks = recursive._chunk_text(body)

                for chunk in body_chunks:
                    content = (header + chunk.content) if self.include_header else chunk.content
                    chunks.append(
                        Chunk(
                            content=content.strip(),
                            index=index,
                            start_char=section.start_char + chunk.start_char,
                            end_char=section.start_char + chunk.end_char,
                            metadata={
                                "strategy": "markdown",
                                "section_title": section.title,
                                "header_level": section.level,
                                "split_section": True,
                            },
                        )
                    )
                    index += 1

        return chunks


class HTMLChunker(BaseChunker):
    """
    Split HTML documents respecting tag structure.

    Uses BeautifulSoup to parse HTML and create semantically meaningful chunks.

    Example:
        >>> chunker = HTMLChunker(chunk_size=1000)
        >>> chunks = chunker.chunk(html_text)
    """

    # Tags that typically contain main content
    CONTENT_TAGS = ["p", "div", "section", "article", "li", "td", "th"]

    # Tags to split on
    SPLIT_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6", "section", "article"]

    def __init__(
        self,
        config: Optional[ChunkConfig] = None,
        extract_text: bool = True,
        **kwargs,
    ):
        """
        Initialize HTML chunker.

        Args:
            config: ChunkConfig instance
            extract_text: If True, extract text only; if False, keep HTML
            **kwargs: Override config parameters
        """
        super().__init__(config, **kwargs)
        self.extract_text = extract_text

    def _chunk_text(self, text: str) -> List[Chunk]:
        """
        Split HTML by structure.

        Args:
            text: HTML text

        Returns:
            List of Chunk objects
        """
        if not text:
            return []

        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(text, "html.parser")
        except ImportError:
            # BeautifulSoup not available, fall back to regex
            return self._chunk_with_regex(text)

        # Extract content
        content_parts = self._extract_content(soup)

        if not content_parts:
            # No structured content found
            plain_text = soup.get_text(separator=" ", strip=True)
            from chunkwise.strategies.recursive import RecursiveChunker

            recursive = RecursiveChunker(config=self.config)
            return recursive._chunk_text(plain_text)

        return self._process_content_parts(content_parts, text)

    def _extract_content(self, soup) -> List[Tuple[str, str, int]]:
        """
        Extract content from HTML.

        Args:
            soup: BeautifulSoup object

        Returns:
            List of (tag_name, content, position) tuples
        """
        parts = []

        for tag in soup.find_all(self.CONTENT_TAGS + self.SPLIT_TAGS):
            if self.extract_text:
                content = tag.get_text(separator=" ", strip=True)
            else:
                content = str(tag)

            if content:
                parts.append((tag.name, content, 0))

        return parts

    def _process_content_parts(
        self, parts: List[Tuple[str, str, int]], text: str
    ) -> List[Chunk]:
        """
        Process content parts into chunks.

        Args:
            parts: List of content parts
            text: Original text

        Returns:
            List of Chunk objects
        """
        chunks = []
        current_content = []
        current_length = 0
        index = 0

        for tag_name, content, _ in parts:
            content_length = len(content)

            if current_length + content_length > self.config.chunk_size and current_content:
                chunk_text = "\n".join(current_content)
                chunks.append(
                    Chunk(
                        content=chunk_text,
                        index=index,
                        start_char=0,
                        end_char=len(chunk_text),
                        metadata={"strategy": "html"},
                    )
                )
                index += 1
                current_content = []
                current_length = 0

            current_content.append(content)
            current_length += content_length + 1

        if current_content:
            chunk_text = "\n".join(current_content)
            chunks.append(
                Chunk(
                    content=chunk_text,
                    index=index,
                    start_char=0,
                    end_char=len(chunk_text),
                    metadata={"strategy": "html"},
                )
            )

        return chunks

    def _chunk_with_regex(self, text: str) -> List[Chunk]:
        """
        Fallback chunking using regex.

        Args:
            text: HTML text

        Returns:
            List of Chunk objects
        """
        # Simple tag removal
        clean_text = re.sub(r"<[^>]+>", " ", text)
        clean_text = re.sub(r"\s+", " ", clean_text).strip()

        from chunkwise.strategies.recursive import RecursiveChunker

        recursive = RecursiveChunker(config=self.config)
        return recursive._chunk_text(clean_text)


class CodeChunker(BaseChunker):
    """
    Split code files respecting function and class boundaries.

    Example:
        >>> chunker = CodeChunker(chunk_size=2000, language="python")
        >>> chunks = chunker.chunk(python_code)
    """

    # Language-specific patterns
    PATTERNS = {
        "python": {
            "function": re.compile(r"^(async\s+)?def\s+\w+", re.MULTILINE),
            "class": re.compile(r"^class\s+\w+", re.MULTILINE),
        },
        "javascript": {
            "function": re.compile(
                r"^(async\s+)?function\s+\w+|^(const|let|var)\s+\w+\s*=\s*(async\s+)?\(",
                re.MULTILINE,
            ),
            "class": re.compile(r"^class\s+\w+", re.MULTILINE),
        },
        "typescript": {
            "function": re.compile(
                r"^(async\s+)?function\s+\w+|^(const|let|var)\s+\w+\s*=\s*(async\s+)?\(",
                re.MULTILINE,
            ),
            "class": re.compile(r"^(export\s+)?(abstract\s+)?class\s+\w+", re.MULTILINE),
        },
    }

    def __init__(
        self,
        config: Optional[ChunkConfig] = None,
        code_language: str = "python",
        **kwargs,
    ):
        """
        Initialize code chunker.

        Args:
            config: ChunkConfig instance
            code_language: Programming language ("python", "javascript", etc.)
            **kwargs: Override config parameters
        """
        super().__init__(config, **kwargs)
        self.code_language = code_language.lower()

    def _chunk_text(self, text: str) -> List[Chunk]:
        """
        Split code by function/class boundaries.

        Args:
            text: Source code

        Returns:
            List of Chunk objects
        """
        if not text:
            return []

        patterns = self.PATTERNS.get(self.code_language, self.PATTERNS["python"])

        # Find all function and class definitions
        boundaries = self._find_boundaries(text, patterns)

        if not boundaries:
            # No boundaries found, use line-based chunking
            return self._chunk_by_lines(text)

        return self._chunk_by_boundaries(text, boundaries)

    def _find_boundaries(self, text: str, patterns: dict) -> List[int]:
        """
        Find code boundaries.

        Args:
            text: Source code
            patterns: Regex patterns

        Returns:
            List of boundary positions
        """
        boundaries = set([0])

        for pattern in patterns.values():
            for match in pattern.finditer(text):
                boundaries.add(match.start())

        boundaries.add(len(text))
        return sorted(boundaries)

    def _chunk_by_boundaries(self, text: str, boundaries: List[int]) -> List[Chunk]:
        """
        Create chunks at code boundaries.

        Args:
            text: Source code
            boundaries: List of boundary positions

        Returns:
            List of Chunk objects
        """
        chunks = []
        current_content = ""
        current_start = 0
        index = 0

        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            segment = text[start:end]

            if len(current_content) + len(segment) <= self.config.chunk_size:
                if not current_content:
                    current_start = start
                current_content += segment
            else:
                if current_content:
                    chunks.append(
                        Chunk(
                            content=current_content.strip(),
                            index=index,
                            start_char=current_start,
                            end_char=current_start + len(current_content),
                            metadata={"strategy": "code", "language": self.code_language},
                        )
                    )
                    index += 1

                current_content = segment
                current_start = start

        if current_content:
            chunks.append(
                Chunk(
                    content=current_content.strip(),
                    index=index,
                    start_char=current_start,
                    end_char=current_start + len(current_content),
                    metadata={"strategy": "code", "language": self.code_language},
                )
            )

        return chunks

    def _chunk_by_lines(self, text: str) -> List[Chunk]:
        """
        Fallback line-based chunking.

        Args:
            text: Source code

        Returns:
            List of Chunk objects
        """
        lines = text.split("\n")
        chunks = []
        current_lines = []
        current_length = 0
        index = 0
        char_position = 0

        for line in lines:
            line_length = len(line) + 1  # +1 for newline

            if current_length + line_length > self.config.chunk_size and current_lines:
                chunk_text = "\n".join(current_lines)
                chunks.append(
                    Chunk(
                        content=chunk_text,
                        index=index,
                        start_char=char_position,
                        end_char=char_position + len(chunk_text),
                        metadata={"strategy": "code", "language": self.code_language},
                    )
                )
                char_position += len(chunk_text) + 1
                index += 1
                current_lines = []
                current_length = 0

            current_lines.append(line)
            current_length += line_length

        if current_lines:
            chunk_text = "\n".join(current_lines)
            chunks.append(
                Chunk(
                    content=chunk_text,
                    index=index,
                    start_char=char_position,
                    end_char=char_position + len(chunk_text),
                    metadata={"strategy": "code", "language": self.code_language},
                )
            )

        return chunks
