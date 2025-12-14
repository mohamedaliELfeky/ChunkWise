"""
Hybrid Chunking Strategy

Combine multiple chunking strategies for optimal results.
"""

from typing import List, Optional, Union, Type
from chunkwise.base import BaseChunker
from chunkwise.chunk import Chunk
from chunkwise.config import ChunkConfig


class HybridChunker(BaseChunker):
    """
    Combine multiple chunking strategies.

    Applies strategies in sequence or uses fallback chains.
    Useful for complex documents that benefit from multiple approaches.

    Example:
        >>> # Sequential: try strategies in order until one succeeds
        >>> chunker = HybridChunker(
        ...     strategies=["semantic", "recursive"],
        ...     mode="fallback"
        ... )

        >>> # Pipeline: apply multiple strategies
        >>> chunker = HybridChunker(
        ...     strategies=["paragraph", "sentence"],
        ...     mode="pipeline"
        ... )
    """

    def __init__(
        self,
        config: Optional[ChunkConfig] = None,
        strategies: Optional[List[str]] = None,
        mode: str = "fallback",
        **kwargs,
    ):
        """
        Initialize hybrid chunker.

        Args:
            config: ChunkConfig instance
            strategies: List of strategy names to combine
            mode: Combination mode:
                - "fallback": Try strategies in order until one succeeds
                - "pipeline": Apply strategies sequentially
                - "merge": Run all strategies and merge results
            **kwargs: Override config parameters
        """
        super().__init__(config, **kwargs)
        self.strategies = strategies or ["recursive", "sentence"]
        self.mode = mode

    def _chunk_text(self, text: str) -> List[Chunk]:
        """
        Apply hybrid chunking.

        Args:
            text: Input text

        Returns:
            List of Chunk objects
        """
        if not text:
            return []

        if self.mode == "fallback":
            return self._fallback_chunk(text)
        elif self.mode == "pipeline":
            return self._pipeline_chunk(text)
        elif self.mode == "merge":
            return self._merge_chunk(text)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _fallback_chunk(self, text: str) -> List[Chunk]:
        """
        Try strategies in order until one succeeds.

        Args:
            text: Input text

        Returns:
            List of Chunk objects
        """
        last_error = None

        for strategy_name in self.strategies:
            try:
                chunker = self._get_chunker(strategy_name)
                chunks = chunker._chunk_text(text)

                if chunks:
                    for chunk in chunks:
                        chunk.metadata["strategy"] = f"hybrid_{strategy_name}"
                    return chunks

            except Exception as e:
                last_error = e
                continue

        # All strategies failed, use basic character chunking
        from chunkwise.strategies.fixed import CharacterChunker

        chunker = CharacterChunker(config=self.config)
        chunks = chunker._chunk_text(text)

        for chunk in chunks:
            chunk.metadata["strategy"] = "hybrid_fallback"
            if last_error:
                chunk.metadata["fallback_reason"] = str(last_error)

        return chunks

    def _pipeline_chunk(self, text: str) -> List[Chunk]:
        """
        Apply strategies sequentially.

        Each strategy processes the output of the previous one.

        Args:
            text: Input text

        Returns:
            List of Chunk objects
        """
        current_chunks = [
            Chunk(
                content=text,
                index=0,
                start_char=0,
                end_char=len(text),
                metadata={},
            )
        ]

        for strategy_name in self.strategies:
            new_chunks = []
            chunker = self._get_chunker(strategy_name)

            for chunk in current_chunks:
                sub_chunks = chunker._chunk_text(chunk.content)

                # Adjust positions relative to original
                for sub in sub_chunks:
                    sub.start_char += chunk.start_char
                    sub.end_char = sub.start_char + len(sub.content)
                    sub.metadata["pipeline_stage"] = strategy_name

                new_chunks.extend(sub_chunks)

            current_chunks = new_chunks

        # Re-index
        for i, chunk in enumerate(current_chunks):
            chunk.index = i
            chunk.metadata["strategy"] = "hybrid_pipeline"

        return current_chunks

    def _merge_chunk(self, text: str) -> List[Chunk]:
        """
        Run all strategies and merge results.

        Takes the best chunks from each strategy based on size fit.

        Args:
            text: Input text

        Returns:
            List of Chunk objects
        """
        all_results = []

        for strategy_name in self.strategies:
            try:
                chunker = self._get_chunker(strategy_name)
                chunks = chunker._chunk_text(text)

                for chunk in chunks:
                    chunk.metadata["source_strategy"] = strategy_name

                all_results.append((strategy_name, chunks))
            except Exception:
                continue

        if not all_results:
            return self._fallback_chunk(text)

        # Score and select best chunks
        best_chunks = self._select_best_chunks(all_results, text)

        for i, chunk in enumerate(best_chunks):
            chunk.index = i
            chunk.metadata["strategy"] = "hybrid_merge"

        return best_chunks

    def _select_best_chunks(
        self,
        results: List[tuple],
        text: str,
    ) -> List[Chunk]:
        """
        Select best chunks from multiple strategy results.

        Args:
            results: List of (strategy_name, chunks) tuples
            text: Original text

        Returns:
            Best chunks
        """
        # Simple approach: choose the strategy with best average chunk size
        best_score = float("inf")
        best_chunks = []
        target_size = self.config.chunk_size

        for strategy_name, chunks in results:
            if not chunks:
                continue

            # Score based on how close chunks are to target size
            sizes = [len(c.content) for c in chunks]
            avg_diff = sum(abs(s - target_size) for s in sizes) / len(sizes)

            if avg_diff < best_score:
                best_score = avg_diff
                best_chunks = chunks

        return best_chunks if best_chunks else results[0][1] if results else []

    def _get_chunker(self, strategy_name: str) -> BaseChunker:
        """
        Get a chunker instance by strategy name.

        Args:
            strategy_name: Strategy name

        Returns:
            Chunker instance
        """
        strategy_map = {
            "character": "chunkwise.strategies.fixed.CharacterChunker",
            "token": "chunkwise.strategies.fixed.TokenChunker",
            "word": "chunkwise.strategies.fixed.WordChunker",
            "sentence": "chunkwise.strategies.sentence.SentenceChunker",
            "multi_sentence": "chunkwise.strategies.sentence.MultiSentenceChunker",
            "paragraph": "chunkwise.strategies.paragraph.ParagraphChunker",
            "recursive": "chunkwise.strategies.recursive.RecursiveChunker",
            "sliding_window": "chunkwise.strategies.sliding_window.SlidingWindowChunker",
            "markdown": "chunkwise.strategies.document_structure.MarkdownChunker",
            "html": "chunkwise.strategies.document_structure.HTMLChunker",
            "code": "chunkwise.strategies.document_structure.CodeChunker",
            "semantic": "chunkwise.strategies.semantic.SemanticChunker",
            "agentic": "chunkwise.strategies.agentic.AgenticChunker",
            "proposition": "chunkwise.strategies.agentic.PropositionChunker",
            "late": "chunkwise.strategies.late.LateChunker",
        }

        if strategy_name not in strategy_map:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        # Import and instantiate
        module_path, class_name = strategy_map[strategy_name].rsplit(".", 1)
        module = __import__(module_path, fromlist=[class_name])
        chunker_class = getattr(module, class_name)

        return chunker_class(config=self.config)


class AdaptiveChunker(BaseChunker):
    """
    Automatically select the best chunking strategy based on content.

    Analyzes the input text and chooses an appropriate strategy.
    """

    def __init__(
        self,
        config: Optional[ChunkConfig] = None,
        **kwargs,
    ):
        """
        Initialize adaptive chunker.

        Args:
            config: ChunkConfig instance
            **kwargs: Override config parameters
        """
        super().__init__(config, **kwargs)

    def _chunk_text(self, text: str) -> List[Chunk]:
        """
        Automatically select and apply the best strategy.

        Args:
            text: Input text

        Returns:
            List of Chunk objects
        """
        if not text:
            return []

        strategy = self._detect_best_strategy(text)
        chunker = self._get_chunker(strategy)

        chunks = chunker._chunk_text(text)

        for chunk in chunks:
            chunk.metadata["strategy"] = f"adaptive_{strategy}"
            chunk.metadata["detected_strategy"] = strategy

        return chunks

    def _detect_best_strategy(self, text: str) -> str:
        """
        Detect the best strategy for the text.

        Args:
            text: Input text

        Returns:
            Strategy name
        """
        # Check for Markdown
        if self._is_markdown(text):
            return "markdown"

        # Check for HTML
        if self._is_html(text):
            return "html"

        # Check for code
        if self._is_code(text):
            return "code"

        # Check for structured document
        if self._has_paragraphs(text):
            return "paragraph"

        # Default to recursive
        return "recursive"

    def _is_markdown(self, text: str) -> bool:
        """Check if text is Markdown."""
        import re

        # Look for Markdown headers
        return bool(re.search(r"^#{1,6}\s+", text, re.MULTILINE))

    def _is_html(self, text: str) -> bool:
        """Check if text is HTML."""
        import re

        return bool(re.search(r"<\w+[^>]*>", text))

    def _is_code(self, text: str) -> bool:
        """Check if text is source code."""
        import re

        code_patterns = [
            r"^def\s+\w+",  # Python function
            r"^class\s+\w+",  # Class definition
            r"^function\s+\w+",  # JavaScript function
            r"^import\s+",  # Import statement
            r"^from\s+\w+\s+import",  # Python import
        ]

        for pattern in code_patterns:
            if re.search(pattern, text, re.MULTILINE):
                return True
        return False

    def _has_paragraphs(self, text: str) -> bool:
        """Check if text has clear paragraph structure."""
        return "\n\n" in text

    def _get_chunker(self, strategy_name: str) -> BaseChunker:
        """Get chunker instance."""
        hybrid = HybridChunker(config=self.config, strategies=[strategy_name])
        return hybrid._get_chunker(strategy_name)
