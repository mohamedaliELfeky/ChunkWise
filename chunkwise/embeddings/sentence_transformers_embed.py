"""
Sentence Transformers Embedding Provider

Local embedding generation using sentence-transformers library.
"""

from typing import List, Optional
import numpy as np

from chunkwise.embeddings.base import BaseEmbedding
from chunkwise.exceptions import EmbeddingError


class SentenceTransformersEmbedding(BaseEmbedding):
    """
    Embedding provider using sentence-transformers library.

    Supports many models including:
    - all-MiniLM-L6-v2 (fast, good quality)
    - all-mpnet-base-v2 (high quality)
    - paraphrase-multilingual-MiniLM-L12-v2 (multilingual)
    - multi-qa-MiniLM-L6-cos-v1 (QA optimized)

    Example:
        >>> embedder = SentenceTransformersEmbedding()
        >>> embeddings = embedder.embed(["Hello world", "Hi there"])
        >>> embeddings.shape
        (2, 384)  # MiniLM produces 384-dim embeddings
    """

    # Popular model recommendations
    RECOMMENDED_MODELS = {
        "fast": "all-MiniLM-L6-v2",
        "quality": "all-mpnet-base-v2",
        "multilingual": "paraphrase-multilingual-MiniLM-L12-v2",
        "qa": "multi-qa-MiniLM-L6-cos-v1",
        "arabic": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    }

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        normalize: bool = True,
    ):
        """
        Initialize sentence transformers embedding provider.

        Args:
            model_name: Model name or path (or key from RECOMMENDED_MODELS)
            device: Device to use ("cpu", "cuda", etc.)
            normalize: Whether to normalize embeddings to unit length
        """
        # Resolve model name alias
        self.model_name = self.RECOMMENDED_MODELS.get(model_name, model_name)
        self.device = device
        self.normalize = normalize
        self._model = None

    @property
    def model(self):
        """Lazy load the model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                )
            except ImportError:
                raise EmbeddingError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
            except Exception as e:
                raise EmbeddingError(f"Failed to load model {self.model_name}: {e}")

        return self._model

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

        try:
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
            )
            return np.array(embeddings)
        except Exception as e:
            raise EmbeddingError(f"Embedding generation failed: {e}")

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Generate embeddings in batches.

        Args:
            texts: List of texts to embed
            batch_size: Batch size
            show_progress: Show progress bar

        Returns:
            numpy array of embeddings
        """
        if not texts:
            return np.array([])

        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=self.normalize,
                show_progress_bar=show_progress,
            )
            return np.array(embeddings)
        except Exception as e:
            raise EmbeddingError(f"Batch embedding failed: {e}")

    @property
    def embedding_dim(self) -> int:
        """Return the dimension of embeddings."""
        return self.model.get_sentence_embedding_dimension()

    def __repr__(self) -> str:
        return f"SentenceTransformersEmbedding(model='{self.model_name}')"
