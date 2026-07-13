"""Embeds a user's natural language query and retrieves the most
semantically similar videos from the vector database."""

from typing import List

import config
from models import SearchResult
from embedding_service import EmbeddingService, SentenceTransformerEmbeddingService
from vector_store import VectorStore, ChromaVectorStore


class SearchService:
    def __init__(self, embedder: EmbeddingService, store: VectorStore, top_k: int):
        self._embedder = embedder
        self._store = store
        self._top_k = top_k

    def search(self, query: str) -> List[SearchResult]:
        query_embedding = self._embedder.embed_one(query)
        matches = self._store.query(query_embedding, top_k=self._top_k)

        results = []
        for m in matches:
            similarity = 1 - m["distance"]  # cosine distance -> similarity
            results.append(SearchResult(
                video=m["metadata"].get("video", m["id"]),
                summary_caption=m["metadata"].get("summary_caption", m["document"]),
                score=round(similarity, 4),
            ))
        return results


def build_default_search_service() -> SearchService:
    """Convenience factory wiring up the default embedder + vector store."""
    embedder = SentenceTransformerEmbeddingService(config.EMBEDDING_MODEL_NAME)
    store = ChromaVectorStore(config.CHROMA_PERSIST_DIR, config.COLLECTION_NAME)
    return SearchService(embedder, store, config.TOP_K_RESULTS)