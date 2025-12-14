"""
Agentic Chunking Strategies

LLM-powered chunking that uses AI to determine optimal break points.
"""

from typing import List, Optional
import json
import re

from chunkwise.base import BaseChunker
from chunkwise.chunk import Chunk
from chunkwise.config import ChunkConfig, LLMConfig
from chunkwise.llm.base import get_llm_provider
from chunkwise.exceptions import LLMError


class AgenticChunker(BaseChunker):
    """
    LLM-powered chunking that determines optimal break points.

    Uses an LLM to analyze text and identify semantically meaningful
    boundaries for chunking. The AI considers context, topic changes,
    and narrative flow.

    Example:
        >>> chunker = AgenticChunker(chunk_size=500)
        >>> chunks = chunker.chunk(text)

        >>> # With specific LLM
        >>> chunker = AgenticChunker(
        ...     llm_provider="anthropic",
        ...     llm_model="claude-3-sonnet-20240229"
        ... )
    """

    CHUNKING_PROMPT = """Analyze the following text and identify optimal points to split it into chunks.
Consider:
1. Topic changes
2. Paragraph boundaries
3. Semantic coherence
4. Each chunk should be roughly {target_size} characters

Text to analyze:
{text}

Return a JSON array of objects with "start" and "end" character positions for each chunk.
Example: [{{"start": 0, "end": 500}}, {{"start": 500, "end": 1000}}]

Respond with ONLY the JSON array, no other text."""

    def __init__(
        self,
        config: Optional[ChunkConfig] = None,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o-mini",
        **kwargs,
    ):
        """
        Initialize agentic chunker.

        Args:
            config: ChunkConfig instance
            llm_provider: LLM provider name
            llm_model: LLM model name
            **kwargs: Override config parameters
        """
        super().__init__(config, **kwargs)
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self._llm = None

    @property
    def llm(self):
        """Lazy load LLM provider."""
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
        Use LLM to determine chunk boundaries.

        Args:
            text: Input text

        Returns:
            List of Chunk objects
        """
        if not text:
            return []

        # For very short texts, return as-is
        if len(text) <= self.config.chunk_size:
            return [
                Chunk(
                    content=text,
                    index=0,
                    start_char=0,
                    end_char=len(text),
                    metadata={"strategy": "agentic"},
                )
            ]

        # For very long texts, process in windows
        if len(text) > 10000:
            return self._chunk_large_text(text)

        try:
            # Get LLM suggestions
            boundaries = self._get_llm_boundaries(text)
            return self._create_chunks_from_boundaries(text, boundaries)
        except Exception as e:
            # Fallback to recursive chunking
            from chunkwise.strategies.recursive import RecursiveChunker

            recursive = RecursiveChunker(config=self.config)
            chunks = recursive._chunk_text(text)
            for chunk in chunks:
                chunk.metadata["strategy"] = "agentic_fallback"
                chunk.metadata["fallback_reason"] = str(e)
            return chunks

    def _get_llm_boundaries(self, text: str) -> List[dict]:
        """
        Get chunk boundaries from LLM.

        Args:
            text: Input text

        Returns:
            List of {start, end} dictionaries
        """
        prompt = self.CHUNKING_PROMPT.format(
            target_size=self.config.chunk_size,
            text=text[:8000],  # Limit for API
        )

        response = self.llm.generate(prompt)

        # Parse JSON response
        try:
            # Extract JSON from response
            json_match = re.search(r"\[.*\]", response, re.DOTALL)
            if json_match:
                boundaries = json.loads(json_match.group())
                return boundaries
            else:
                raise ValueError("No JSON array found in response")
        except json.JSONDecodeError as e:
            raise LLMError(f"Failed to parse LLM response: {e}")

    def _create_chunks_from_boundaries(
        self, text: str, boundaries: List[dict]
    ) -> List[Chunk]:
        """
        Create chunks from LLM-suggested boundaries.

        Args:
            text: Original text
            boundaries: List of boundary dictionaries

        Returns:
            List of Chunk objects
        """
        chunks = []

        for index, boundary in enumerate(boundaries):
            start = boundary.get("start", 0)
            end = boundary.get("end", len(text))

            # Validate boundaries
            start = max(0, min(start, len(text)))
            end = max(start, min(end, len(text)))

            chunk_text = text[start:end]

            if chunk_text.strip():
                chunks.append(
                    Chunk(
                        content=chunk_text.strip(),
                        index=index,
                        start_char=start,
                        end_char=end,
                        metadata={"strategy": "agentic"},
                    )
                )

        # Ensure we have at least one chunk
        if not chunks:
            chunks.append(
                Chunk(
                    content=text,
                    index=0,
                    start_char=0,
                    end_char=len(text),
                    metadata={"strategy": "agentic"},
                )
            )

        return chunks

    def _chunk_large_text(self, text: str) -> List[Chunk]:
        """
        Handle very large texts by processing in windows.

        Args:
            text: Large text

        Returns:
            List of Chunk objects
        """
        # Use recursive chunking for initial split
        from chunkwise.strategies.recursive import RecursiveChunker

        initial_config = ChunkConfig(
            chunk_size=5000,
            chunk_overlap=500,
        )
        recursive = RecursiveChunker(config=initial_config)
        initial_chunks = recursive._chunk_text(text)

        # Process each initial chunk with LLM
        all_chunks = []
        index = 0

        for chunk in initial_chunks:
            try:
                sub_chunks = self._chunk_text(chunk.content)
                for sub in sub_chunks:
                    # Adjust positions
                    sub.index = index
                    sub.start_char += chunk.start_char
                    sub.end_char += chunk.start_char
                    all_chunks.append(sub)
                    index += 1
            except Exception:
                # Keep original chunk
                chunk.index = index
                chunk.metadata["strategy"] = "agentic_partial"
                all_chunks.append(chunk)
                index += 1

        return all_chunks


class PropositionChunker(BaseChunker):
    """
    Break text into atomic propositions.

    Uses LLM to extract self-contained, atomic facts from text.
    Each proposition is a single, verifiable claim.

    Example:
        >>> chunker = PropositionChunker()
        >>> chunks = chunker.chunk(text)
        >>> # Returns one chunk per proposition/fact
    """

    PROPOSITION_PROMPT = """Extract atomic propositions from the following text.
Each proposition should be:
1. A single, self-contained fact
2. Verifiable independently
3. Complete without needing other propositions

Text:
{text}

Return a JSON array of proposition strings.
Example: ["The sky is blue.", "Water is H2O."]

Respond with ONLY the JSON array."""

    def __init__(
        self,
        config: Optional[ChunkConfig] = None,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o-mini",
        group_propositions: bool = True,
        propositions_per_chunk: int = 5,
        **kwargs,
    ):
        """
        Initialize proposition chunker.

        Args:
            config: ChunkConfig instance
            llm_provider: LLM provider name
            llm_model: LLM model name
            group_propositions: Whether to group propositions into chunks
            propositions_per_chunk: How many propositions per chunk
            **kwargs: Override config parameters
        """
        super().__init__(config, **kwargs)
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.group_propositions = group_propositions
        self.propositions_per_chunk = propositions_per_chunk
        self._llm = None

    @property
    def llm(self):
        """Lazy load LLM provider."""
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
        Extract propositions from text.

        Args:
            text: Input text

        Returns:
            List of Chunk objects
        """
        if not text:
            return []

        try:
            propositions = self._extract_propositions(text)

            if self.group_propositions:
                return self._group_propositions(propositions)
            else:
                return self._create_individual_chunks(propositions)

        except Exception as e:
            # Fallback
            from chunkwise.strategies.sentence import SentenceChunker

            sentence_chunker = SentenceChunker(config=self.config)
            chunks = sentence_chunker._chunk_text(text)
            for chunk in chunks:
                chunk.metadata["strategy"] = "proposition_fallback"
                chunk.metadata["fallback_reason"] = str(e)
            return chunks

    def _extract_propositions(self, text: str) -> List[str]:
        """
        Extract propositions using LLM.

        Args:
            text: Input text

        Returns:
            List of proposition strings
        """
        # Process in chunks if text is too long
        if len(text) > 4000:
            from chunkwise.strategies.paragraph import ParagraphChunker

            para_chunker = ParagraphChunker(
                config=ChunkConfig(chunk_size=3000, chunk_overlap=0)
            )
            paragraphs = para_chunker._chunk_text(text)

            all_propositions = []
            for para in paragraphs:
                props = self._extract_propositions(para.content)
                all_propositions.extend(props)
            return all_propositions

        prompt = self.PROPOSITION_PROMPT.format(text=text)
        response = self.llm.generate(prompt)

        try:
            json_match = re.search(r"\[.*\]", response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return [text]  # Return original if parsing fails
        except json.JSONDecodeError:
            return [text]

    def _create_individual_chunks(self, propositions: List[str]) -> List[Chunk]:
        """
        Create one chunk per proposition.

        Args:
            propositions: List of propositions

        Returns:
            List of Chunk objects
        """
        chunks = []
        char_position = 0

        for index, prop in enumerate(propositions):
            chunks.append(
                Chunk(
                    content=prop,
                    index=index,
                    start_char=char_position,
                    end_char=char_position + len(prop),
                    metadata={"strategy": "proposition"},
                )
            )
            char_position += len(prop) + 1

        return chunks

    def _group_propositions(self, propositions: List[str]) -> List[Chunk]:
        """
        Group propositions into chunks.

        Args:
            propositions: List of propositions

        Returns:
            List of Chunk objects
        """
        chunks = []
        index = 0
        char_position = 0

        for i in range(0, len(propositions), self.propositions_per_chunk):
            group = propositions[i : i + self.propositions_per_chunk]
            chunk_text = "\n".join(f"- {p}" for p in group)

            chunks.append(
                Chunk(
                    content=chunk_text,
                    index=index,
                    start_char=char_position,
                    end_char=char_position + len(chunk_text),
                    metadata={
                        "strategy": "proposition",
                        "proposition_count": len(group),
                    },
                )
            )

            char_position += len(chunk_text) + 1
            index += 1

        return chunks
