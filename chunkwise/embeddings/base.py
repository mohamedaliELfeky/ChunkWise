"""
Base Embedding Interface

Abstract base class for all embedding providers.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np


class BaseEmbedding(ABC):
    """
    Abstract base class for embedding providers.

    All embedding providers must implement the embed method.
    """

    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        pass

    def embed_single(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Text string to embed

        Returns:
            numpy array of shape (embedding_dim,)
        """
        return self.embed([text])[0]

    @property
    def embedding_dim(self) -> int:
        """Return the dimension of embeddings."""
        # Generate a test embedding to determine dimension
        test_embedding = self.embed(["test"])
        return test_embedding.shape[1]

    def similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Cosine similarity score (0 to 1)
        """
        embeddings = self.embed([text1, text2])
        return self._cosine_similarity(embeddings[0], embeddings[1])

    def similarity_matrix(self, texts: List[str]) -> np.ndarray:
        """
        Calculate pairwise similarity matrix.

        Args:
            texts: List of texts

        Returns:
            Similarity matrix of shape (len(texts), len(texts))
        """
        embeddings = self.embed(texts)
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-10)
        # Compute similarity matrix
        return np.dot(normalized, normalized.T)

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)


def get_embedding_provider(
    provider: str = "sentence-transformers",
    model: str = "all-MiniLM-L6-v2",
    api_key: Optional[str] = None,
) -> BaseEmbedding:
    """
    Factory function to get an embedding provider.

    Args:
        provider: Provider name ("sentence-transformers", "openai")
        model: Model name
        api_key: API key for cloud providers

    Returns:
        Embedding provider instance
    """
    if provider == "sentence-transformers":
        from chunkwise.embeddings.sentence_transformers_embed import SentenceTransformersEmbedding

        return SentenceTransformersEmbedding(model_name=model)
    elif provider == "openai":
        from chunkwise.embeddings.openai_embed import OpenAIEmbedding

        return OpenAIEmbedding(model_name=model, api_key=api_key)
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")
