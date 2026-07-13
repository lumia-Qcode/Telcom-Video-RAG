"""
Vector database abstraction so the backend (currently ChromaDB, persisted
to disk locally) can be swapped later without changing the search service.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class VectorStore(ABC):
    @abstractmethod
    def upsert(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def query(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        raise NotImplementedError


class ChromaVectorStore(VectorStore):
    def __init__(self, persist_dir: str, collection_name: str):
        import chromadb
        self._client = chromadb.PersistentClient(path=persist_dir)
        # cosine space -> distances are directly convertible to an
        # intuitive 0-1 similarity score (1 - distance)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, ids, embeddings, documents, metadatas) -> None:
        self._collection.upsert(
            ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
        )

    def query(self, query_embedding, top_k) -> List[Dict[str, Any]]:
        result = self._collection.query(
            query_embeddings=[query_embedding], n_results=top_k
        )

        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        documents = result.get("documents", [[]])[0]

        return [
            {
                "id": ids[i],
                "distance": distances[i],
                "metadata": metadatas[i],
                "document": documents[i],
            }
            for i in range(len(ids))
        ]