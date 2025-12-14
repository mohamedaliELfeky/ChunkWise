"""
Sentence-Based Chunking Strategies

Chunkers that split text on sentence boundaries.
"""

from typing import List, Optional

from chunkwise.base import BaseChunker
from chunkwise.chunk import Chunk
from chunkwise.config import ChunkConfig, Language
from chunkwise.language.detector import detect_language
from chunkwise.language.arabic.sentence_splitter import ArabicSentenceSplitter
from chunkwise.language.english.sentence_splitter import EnglishSentenceSplitter


class SentenceChunker(BaseChunker):
    """
    Split text into chunks where each chunk contains one or more complete sentences.

    This strategy respects sentence boundaries and creates chunks that don't
    cut sentences in half. Automatically detects language and uses appropriate
    sentence splitters for Arabic and English.

    Example:
        >>> chunker = SentenceChunker(chunk_size=512)  # max 512 chars per chunk
        >>> chunks = chunker.chunk("Hello world. How are you? I'm fine.")

        >>> # Arabic text
        >>> chunker = SentenceChunker(language="ar")
        >>> chunks = chunker.chunk("كيف حالك؟ أنا بخير.")
    """

    def __init__(self, config: Optional[ChunkConfig] = None, **kwargs):
        """
        Initialize sentence chunker.

        Args:
            config: ChunkConfig instance
            **kwargs: Override config parameters
        """
        super().__init__(config, **kwargs)
        self._arabic_splitter = ArabicSentenceSplitter()
        self._english_splitter = EnglishSentenceSplitter()

    def _chunk_text(self, text: str) -> List[Chunk]:
        """
        Split text into sentence-based chunks.

        Args:
            text: Input text

        Returns:
            List of Chunk objects
        """
        if not text:
            return []

        # Detect language if auto
        language = self.config.language
        if language == Language.AUTO:
            language = detect_language(text)

        # Split into sentences
        sentences = self._split_sentences(text, language)

        if not sentences:
            return [
                Chunk(
                    content=text,
                    index=0,
                    start_char=0,
                    end_char=len(text),
                    metadata={"strategy": "sentence", "language": language},
                )
            ]

        # Group sentences into chunks
        return self._group_sentences(sentences, text, language)

    def _split_sentences(self, text: str, language: str) -> List[str]:
        """
        Split text into sentences based on language.

        Args:
            text: Input text
            language: Language code

        Returns:
            List of sentences
        """
        if language == "ar":
            return self._arabic_splitter.split(text)
        else:
            return self._english_splitter.split(text)

    def _group_sentences(
        self, sentences: List[str], text: str, language: str
    ) -> List[Chunk]:
        """
        Group sentences into chunks respecting size limits.

        Args:
            sentences: List of sentences
            text: Original text
            language: Detected language

        Returns:
            List of Chunk objects
        """
        chunks = []
        current_sentences = []
        current_length = 0
        index = 0
        char_position = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            # Check if adding this sentence exceeds limit
            if current_length + sentence_length > self.config.chunk_size and current_sentences:
                # Save current chunk
                chunk_text = " ".join(current_sentences)
                start_char = self._find_text_position(text, current_sentences[0], char_position)
                end_char = start_char + len(chunk_text)

                chunks.append(
                    Chunk(
                        content=chunk_text,
                        index=index,
                        start_char=start_char,
                        end_char=end_char,
                        metadata={
                            "strategy": "sentence",
                            "language": language,
                            "sentence_count": len(current_sentences),
                        },
                    )
                )

                char_position = end_char
                index += 1

                # Handle overlap (keep last sentence if overlap is enabled)
                if self.config.chunk_overlap > 0 and current_sentences:
                    overlap_sentences = self._get_overlap_sentences(
                        current_sentences, self.config.chunk_overlap
                    )
                    current_sentences = overlap_sentences
                    current_length = sum(len(s) for s in current_sentences)
                else:
                    current_sentences = []
                    current_length = 0

            current_sentences.append(sentence)
            current_length += sentence_length + 1  # +1 for space

        # Don't forget the last chunk
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            start_char = self._find_text_position(text, current_sentences[0], char_position)
            end_char = start_char + len(chunk_text)

            chunks.append(
                Chunk(
                    content=chunk_text,
                    index=index,
                    start_char=start_char,
                    end_char=end_char,
                    metadata={
                        "strategy": "sentence",
                        "language": language,
                        "sentence_count": len(current_sentences),
                    },
                )
            )

        return chunks

    def _get_overlap_sentences(
        self, sentences: List[str], target_overlap: int
    ) -> List[str]:
        """
        Get sentences from the end to fill overlap.

        Args:
            sentences: List of sentences
            target_overlap: Target overlap in characters

        Returns:
            List of sentences for overlap
        """
        overlap_sentences = []
        current_length = 0

        for sentence in reversed(sentences):
            if current_length + len(sentence) <= target_overlap:
                overlap_sentences.insert(0, sentence)
                current_length += len(sentence) + 1
            else:
                break

        return overlap_sentences

    def _find_text_position(self, text: str, search: str, start: int = 0) -> int:
        """Find position of text, with fallback."""
        pos = text.find(search[:50], start)
        return pos if pos >= 0 else start


class MultiSentenceChunker(BaseChunker):
    """
    Split text into chunks with a fixed number of sentences per chunk.

    Unlike SentenceChunker which targets a character size, this chunker
    creates chunks with exactly N sentences each.

    Example:
        >>> chunker = MultiSentenceChunker(sentences_per_chunk=3)
        >>> chunks = chunker.chunk("Sentence 1. Sentence 2. Sentence 3. Sentence 4.")
        >>> len(chunks)
        2  # [sentences 1-3], [sentence 4]
    """

    def __init__(self, config: Optional[ChunkConfig] = None, **kwargs):
        """
        Initialize multi-sentence chunker.

        Args:
            config: ChunkConfig instance
            **kwargs: Override config parameters
        """
        # Default to 3 sentences per chunk
        if "sentences_per_chunk" not in kwargs and (
            config is None or config.sentences_per_chunk == 3
        ):
            kwargs.setdefault("sentences_per_chunk", 3)

        super().__init__(config, **kwargs)
        self._arabic_splitter = ArabicSentenceSplitter()
        self._english_splitter = EnglishSentenceSplitter()

    def _chunk_text(self, text: str) -> List[Chunk]:
        """
        Split text into chunks with fixed sentence count.

        Args:
            text: Input text

        Returns:
            List of Chunk objects
        """
        if not text:
            return []

        # Detect language
        language = self.config.language
        if language == Language.AUTO:
            language = detect_language(text)

        # Split into sentences
        if language == "ar":
            sentences = self._arabic_splitter.split(text)
        else:
            sentences = self._english_splitter.split(text)

        if not sentences:
            return [
                Chunk(
                    content=text,
                    index=0,
                    start_char=0,
                    end_char=len(text),
                    metadata={"strategy": "multi_sentence"},
                )
            ]

        # Group by fixed count
        n = self.config.sentences_per_chunk
        overlap_count = max(0, self.config.chunk_overlap // (len(text) // len(sentences) or 1))

        chunks = []
        index = 0
        char_position = 0
        i = 0

        while i < len(sentences):
            # Get N sentences
            chunk_sentences = sentences[i : i + n]
            chunk_text = " ".join(chunk_sentences)

            start_char = text.find(chunk_sentences[0][:30], char_position)
            if start_char < 0:
                start_char = char_position
            end_char = start_char + len(chunk_text)

            chunks.append(
                Chunk(
                    content=chunk_text,
                    index=index,
                    start_char=start_char,
                    end_char=end_char,
                    metadata={
                        "strategy": "multi_sentence",
                        "language": language,
                        "sentence_count": len(chunk_sentences),
                    },
                )
            )

            char_position = end_char
            index += 1

            # Move forward, accounting for overlap
            step = max(1, n - overlap_count)
            i += step

        return chunks
