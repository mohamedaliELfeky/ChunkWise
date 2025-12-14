"""
Embeddings Module

Embedding providers for semantic chunking.
"""

from chunkwise.embeddings.base import BaseEmbedding
from chunkwise.embeddings.sentence_transformers_embed import SentenceTransformersEmbedding
from chunkwise.embeddings.openai_embed import OpenAIEmbedding

__all__ = [
    "BaseEmbedding",
    "SentenceTransformersEmbedding",
    "OpenAIEmbedding",
]
