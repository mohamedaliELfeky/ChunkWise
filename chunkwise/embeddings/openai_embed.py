"""
OpenAI Embedding Provider

Embedding generation using OpenAI's API.
"""

import os
from typing import List, Optional
import numpy as np

from chunkwise.embeddings.base import BaseEmbedding
from chunkwise.exceptions import EmbeddingError


class OpenAIEmbedding(BaseEmbedding):
    """
    Embedding provider using OpenAI's API.

    Supports models:
    - text-embedding-3-small (1536 dims, fast)
    - text-embedding-3-large (3072 dims, highest quality)
    - text-embedding-ada-002 (1536 dims, legacy)

    Example:
        >>> embedder = OpenAIEmbedding(api_key="sk-...")
        >>> embeddings = embedder.embed(["Hello world", "Hi there"])

        >>> # Using environment variable
        >>> os.environ["OPENAI_API_KEY"] = "sk-..."
        >>> embedder = OpenAIEmbedding()
    """

    # Model dimensions
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        dimensions: Optional[int] = None,
    ):
        """
        Initialize OpenAI embedding provider.

        Args:
            model_name: OpenAI embedding model name
            api_key: OpenAI API key (or use OPENAI_API_KEY env var)
            dimensions: Output dimensions (for 3-series models)
        """
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.dimensions = dimensions
        self._client = None

    @property
    def client(self):
        """Lazy load the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise EmbeddingError(
                    "openai not installed. Install with: pip install openai"
                )
            except Exception as e:
                raise EmbeddingError(f"Failed to initialize OpenAI client: {e}")

        return self._client

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for texts.

        Args:
            texts: List of texts to embed

        Returns:
            numpy array of embeddings
        """
        if not texts:
            return np.array([])

        if not self.api_key:
            raise EmbeddingError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        try:
            # Build request parameters
            params = {
                "model": self.model_name,
                "input": texts,
            }

            # Add dimensions for 3-series models
            if self.dimensions and "text-embedding-3" in self.model_name:
                params["dimensions"] = self.dimensions

            response = self.client.embeddings.create(**params)

            # Extract embeddings
            embeddings = [item.embedding for item in response.data]
            return np.array(embeddings)

        except Exception as e:
            raise EmbeddingError(f"OpenAI embedding failed: {e}")

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 100,
    ) -> np.ndarray:
        """
        Generate embeddings in batches.

        OpenAI has a limit on tokens per request, so we batch automatically.

        Args:
            texts: List of texts to embed
            batch_size: Texts per batch

        Returns:
            numpy array of embeddings
        """
        if not texts:
            return np.array([])

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = self.embed(batch)
            all_embeddings.append(batch_embeddings)

        return np.vstack(all_embeddings)

    @property
    def embedding_dim(self) -> int:
        """Return the dimension of embeddings."""
        if self.dimensions:
            return self.dimensions
        return self.MODEL_DIMENSIONS.get(self.model_name, 1536)

    def __repr__(self) -> str:
        return f"OpenAIEmbedding(model='{self.model_name}')"
