"""
Builds the two retrieval indexes from video_report.csv:
  1. ChromaDB collection (semantic / dense vector search), embedded with
     sentence-transformers.
  2. A BM25 index (sparse / keyword search), pickled to disk.

Run this once after (re)generating video_report.csv:
    python ingest.py
"""

import pickle
import re

import chromadb
from sentence_transformers import SentenceTransformer

import config
from data_loader import build_document_text, load_video_records


def _tokenize(text: str):
    return re.findall(r"[a-z0-9]+", text.lower())


def build_metadata(r) -> dict:
    """Chroma metadata values must be str/int/float/bool - flatten everything."""
    meta = {
        "video": r.video,
        "date": r.date or "",
        "time": r.time or "",
        "timestamp": r.datetime_obj.timestamp() if r.datetime_obj else -1.0,
        "hour": r.datetime_obj.hour if r.datetime_obj else -1,
        "classification": r.classification or "",
        "expected_labels": ", ".join(r.expected_labels),
        "verdict": r.verdict or "",
        "summary_caption": r.summary_caption or "",
    }
    for label in config.LABEL_SYNONYMS:
        meta[f"has_{label.replace(' ', '_')}"] = r.has_label(label)
    return meta


def main():
    print("Loading video_report.csv ...")
    records = load_video_records()
    if not records:
        print("No records found - is video_report.csv populated?")
        return
    print(f"Loaded {len(records)} video records.")

    docs = [build_document_text(r) for r in records]
    ids = [r.video for r in records]
    metadatas = [build_metadata(r) for r in records]

    # --- 1. Dense / semantic index (ChromaDB) ---
    print(f"Embedding documents with '{config.EMBEDDING_MODEL}' ...")
    embedder = SentenceTransformer(config.EMBEDDING_MODEL)
    embeddings = embedder.encode(docs, show_progress_bar=True, normalize_embeddings=True).tolist()

    print("Writing to ChromaDB ...")
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(ids=ids, documents=docs, metadatas=metadatas, embeddings=embeddings)
    print(f"ChromaDB collection '{config.COLLECTION_NAME}' has {collection.count()} documents.")

    # --- 2. Sparse / keyword index (BM25) ---
    print("Building BM25 index ...")
    from rank_bm25 import BM25Okapi

    tokenized_corpus = [_tokenize(d) for d in docs]
    bm25 = BM25Okapi(tokenized_corpus)

    with open(config.BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "ids": ids, "docs": docs, "metadatas": metadatas}, f)
    print(f"BM25 index saved to '{config.BM25_INDEX_PATH}'.")

    print("\nIngestion complete. Run `python app.py` to start the RAG API + frontend.")


if __name__ == "__main__":
    main()
