"""
Document Summary Index Strategy

Index summaries for retrieval, return full documents for generation.
"""

from typing import List, Optional

from chunkwise.base import BaseChunker
from chunkwise.chunk import Chunk
from chunkwise.config import ChunkConfig
from chunkwise.llm.base import get_llm_provider
from chunkwise.exceptions import LLMError


class DocumentSummaryChunker(BaseChunker):
    """
    Document Summary Index chunking strategy.

    Creates summary chunks for efficient retrieval while maintaining
    references to full documents for comprehensive generation.

    How it works:
    1. Splits document into sections
    2. Generates summary for each section using LLM
    3. Returns summaries as chunks (for embedding/retrieval)
    4. Stores full section content in metadata (for generation)

    Benefits:
    - Efficient retrieval through concise summaries
    - Full context available for generation
    - Works well with long documents

    Example:
        >>> chunker = DocumentSummaryChunker(
        ...     summary_max_length=200,
        ...     section_size=2000,
        ... )
        >>> chunks = chunker.chunk(document)
        >>>
        >>> # Retrieve using summaries
        >>> # Generate using full content from metadata
        >>> full_content = chunk.metadata["full_content"]
    """

    SUMMARY_PROMPT = """Please provide a concise summary of the following text.
The summary should capture the key points and main ideas.
Keep the summary under {max_length} words.

Text:
{text}

Summary:"""

    def __init__(
        self,
        config: Optional[ChunkConfig] = None,
        section_size: int = 2000,
        summary_max_length: int = 200,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o-mini",
        **kwargs,
    ):
        """
        Initialize Document Summary chunker.

        Args:
            config: ChunkConfig instance
            section_size: Size of sections to summarize
            summary_max_length: Max words in summary
            llm_provider: LLM provider for summarization
            llm_model: LLM model name
            **kwargs: Override config parameters
        """
        super().__init__(config, **kwargs)
        self.section_size = section_size
        self.summary_max_length = summary_max_length
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self._llm = None

    @property
    def llm(self):
        """Lazy load LLM."""
        if self._llm is None:
            api_key = None
            if self.config.llm_config:
                api_key = self.config.llm_config.api_key

            self._llm = get_llm_provider(
                provider=self.llm_provider,
                model=self.llm_model,
                api_key=api_key,
            )
        return self._llm

    def _chunk_text(self, text: str) -> List[Chunk]:
        """
        Create summary chunks.

        Args:
            text: Input text

        Returns:
            List of summary chunks with full content in metadata
        """
        if not text:
            return []

        # Split into sections
        sections = self._create_sections(text)

        # Generate summaries
        chunks = []
        for i, section in enumerate(sections):
            try:
                summary = self._generate_summary(section["content"])
            except Exception:
                # Fallback: use first sentences as summary
                summary = self._extract_summary(section["content"])

            chunks.append(
                Chunk(
                    content=summary,
                    index=i,
                    start_char=section["start"],
                    end_char=section["end"],
                    metadata={
                        "strategy": "document_summary",
                        "full_content": section["content"],
                        "is_summary": True,
                        "original_length": len(section["content"]),
                        "summary_length": len(summary),
                    },
                )
            )

        return chunks

    def _create_sections(self, text: str) -> List[dict]:
        """
        Split document into sections.

        Args:
            text: Full document text

        Returns:
            List of section dicts with content, start, end
        """
        from chunkwise.strategies.recursive import RecursiveChunker

        section_config = ChunkConfig(
            chunk_size=self.section_size,
            chunk_overlap=0,
        )
        chunker = RecursiveChunker(config=section_config)
        chunks = chunker._chunk_text(text)

        return [
            {
                "content": chunk.content,
                "start": chunk.start_char,
                "end": chunk.end_char,
            }
            for chunk in chunks
        ]

    def _generate_summary(self, text: str) -> str:
        """
        Generate summary using LLM.

        Args:
            text: Text to summarize

        Returns:
            Summary string
        """
        prompt = self.SUMMARY_PROMPT.format(
            max_length=self.summary_max_length,
            text=text[:4000],  # Limit for API
        )

        summary = self.llm.generate(prompt)
        return summary.strip()

    def _extract_summary(self, text: str, num_sentences: int = 3) -> str:
        """
        Extract first sentences as fallback summary.

        Args:
            text: Text to extract from
            num_sentences: Number of sentences

        Returns:
            Extracted summary
        """
        from chunkwise.language.english.sentence_splitter import split_english_sentences

        sentences = split_english_sentences(text, min_length=0)
        return " ".join(sentences[:num_sentences])

    def get_full_content(self, chunk: Chunk) -> str:
        """
        Get full content from a summary chunk.

        Args:
            chunk: Summary chunk

        Returns:
            Full section content
        """
        return chunk.metadata.get("full_content", chunk.content)

    def expand_results(self, chunks: List[Chunk]) -> List[str]:
        """
        Expand summary chunks to full content.

        Args:
            chunks: Retrieved summary chunks

        Returns:
            List of full content strings
        """
        return [self.get_full_content(c) for c in chunks]


class KeywordSummaryChunker(DocumentSummaryChunker):
    """
    Document Summary with keyword extraction.

    Adds keyword extraction for better retrieval.
    """

    KEYWORD_PROMPT = """Extract 5-10 important keywords from the following text.
Return only the keywords, separated by commas.

Text:
{text}

Keywords:"""

    def __init__(self, *args, **kwargs):
        """Initialize with keyword extraction enabled."""
        super().__init__(*args, **kwargs)

    def _chunk_text(self, text: str) -> List[Chunk]:
        """Create summary chunks with keywords."""
        chunks = super()._chunk_text(text)

        # Add keywords to each chunk
        for chunk in chunks:
            full_content = chunk.metadata.get("full_content", chunk.content)
            try:
                keywords = self._extract_keywords(full_content)
            except Exception:
                keywords = []

            chunk.metadata["keywords"] = keywords

            # Optionally append keywords to content for better retrieval
            if keywords:
                chunk.content = f"{chunk.content}\n\nKeywords: {', '.join(keywords)}"

        return chunks

    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords using LLM.

        Args:
            text: Input text

        Returns:
            List of keywords
        """
        prompt = self.KEYWORD_PROMPT.format(text=text[:2000])
        response = self.llm.generate(prompt)

        # Parse comma-separated keywords
        keywords = [kw.strip() for kw in response.split(",")]
        return [kw for kw in keywords if kw]
