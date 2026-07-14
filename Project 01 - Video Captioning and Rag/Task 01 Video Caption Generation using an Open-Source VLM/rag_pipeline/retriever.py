"""
Hybrid retriever: combines ChromaDB dense/semantic search with a BM25
sparse/keyword search via Reciprocal Rank Fusion (RRF), then applies
structured metadata filters (date range, time-of-day, labels, verdict)
parsed from the query before/alongside ranking.

Because the whole corpus is small (one row per video), filtering is done
as a post-filter over both result sets - simple, fast, and exact.
"""

import pickle
import re
from dataclasses import dataclass
from typing import List, Optional

import chromadb
from sentence_transformers import SentenceTransformer

import config
from query_parsing import QueryFilters, parse_query


@dataclass
class RetrievedDoc:
    video: str
    document: str
    metadata: dict
    score: float


def _tokenize(text: str):
    return re.findall(r"[a-z0-9]+", text.lower())


def _passes_filters(meta: dict, filters: QueryFilters) -> bool:
    if filters.date_start and filters.date_end:
        ts = meta.get("timestamp", -1)
        if ts < 0:
            return False
        if not (filters.date_start.timestamp() <= ts <= filters.date_end.timestamp()):
            return False

    if filters.hour_start is not None and filters.hour_end is not None:
        hour = meta.get("hour", -1)
        if hour < 0:
            return False
        if not (filters.hour_start <= hour <= filters.hour_end):
            return False

    for label in filters.labels:
        key = f"has_{label.replace(' ', '_')}"
        if not meta.get(key, False):
            return False

    for verdict_kw in filters.verdict_keywords:
        verdict_text = meta.get("verdict", "")
        classification = meta.get("classification", "")
        if verdict_kw not in verdict_text and verdict_kw not in classification:
            return False

    return True


class HybridRetriever:
    def __init__(self):
        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL)
        client = chromadb.PersistentClient(path=config.CHROMA_DIR)
        self.collection = client.get_collection(config.COLLECTION_NAME)

        with open(config.BM25_INDEX_PATH, "rb") as f:
            bm25_data = pickle.load(f)
        self.bm25 = bm25_data["bm25"]
        self.bm25_ids = bm25_data["ids"]
        self.bm25_docs = bm25_data["docs"]
        self.bm25_metadatas = bm25_data["metadatas"]
        self._id_to_doc = dict(zip(self.bm25_ids, self.bm25_docs))
        self._id_to_meta = dict(zip(self.bm25_ids, self.bm25_metadatas))

    def _semantic_search(self, query: str, top_k: int) -> List[str]:
        """Returns ranked list of video ids (best first)."""
        q_emb = self.embedder.encode([query], normalize_embeddings=True).tolist()
        res = self.collection.query(query_embeddings=q_emb, n_results=top_k)
        return res["ids"][0] if res.get("ids") else []

    def _bm25_search(self, query: str, top_k: int) -> List[str]:
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(self.bm25_ids, scores), key=lambda x: x[1], reverse=True)
        return [vid for vid, score in ranked[:top_k] if score > 0]

    def _rrf_fuse(self, ranked_lists: List[List[str]]) -> List[str]:
        scores = {}
        for ranked in ranked_lists:
            for rank, vid in enumerate(ranked):
                scores[vid] = scores.get(vid, 0.0) + 1.0 / (config.RRF_K + rank + 1)
        return [vid for vid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]

    def retrieve(self, query: str, top_k: int = None, filters: Optional[QueryFilters] = None) -> List[RetrievedDoc]:
        top_k = top_k or config.FINAL_TOP_K
        filters = filters if filters is not None else parse_query(query)

        semantic_ranked = self._semantic_search(query, config.SEMANTIC_TOP_K)
        bm25_ranked = self._bm25_search(query, config.BM25_TOP_K)
        fused = self._rrf_fuse([semantic_ranked, bm25_ranked])

        # apply structured filters, preserving fused rank order
        filtered = [vid for vid in fused if _passes_filters(self._id_to_meta.get(vid, {}), filters)]

        # If filters wiped out everything (e.g. query mentions a label that
        # matches no video), fall back to the unfiltered fused ranking so
        # the LLM still gets *something* relevant to reason about/deny.
        final_ids = filtered if filtered else fused
        final_ids = final_ids[:top_k]

        results = []
        rank_scores = {vid: 1.0 / (i + 1) for i, vid in enumerate(fused)}
        for vid in final_ids:
            results.append(
                RetrievedDoc(
                    video=vid,
                    document=self._id_to_doc.get(vid, ""),
                    metadata=self._id_to_meta.get(vid, {}),
                    score=rank_scores.get(vid, 0.0),
                )
            )
        return results


if __name__ == "__main__":
    retriever = HybridRetriever()
    for q in ["technicians working on solar panels", "any hallucinated detections", "substation last week"]:
        print("\nQuery:", q)
        for doc in retriever.retrieve(q, top_k=3):
            print(f"  [{doc.score:.3f}] {doc.video}  ({doc.metadata.get('date')} {doc.metadata.get('time')})")
