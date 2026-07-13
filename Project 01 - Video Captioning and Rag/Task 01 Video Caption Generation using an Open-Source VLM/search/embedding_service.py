"""
Embedding generation, abstracted behind an interface so the backend (right
now, a local sentence-transformers model) can be swapped later - e.g. for
an API-based embedding service - without touching the rest of the search
pipeline. Mirrors the same ObjectDetector/Validator abstraction pattern
already used in the validation pipeline.
"""

from abc import ABC, abstractmethod
from typing import List


class EmbeddingService(ABC):
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts at once (used when building the index)."""
        raise NotImplementedError

    @abstractmethod
    def embed_one(self, text: str) -> List[float]:
        """Embed a single text (used for a user's search query)."""
        raise NotImplementedError


class SentenceTransformerEmbeddingService(EmbeddingService):
    """Local, free, open-source embedding model. No API key, no internet
    dependency once the model weights are downloaded the first time."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [vector.tolist() for vector in self._model.encode(texts)]

    def embed_one(self, text: str) -> List[float]:
        return self._model.encode(text).tolist()