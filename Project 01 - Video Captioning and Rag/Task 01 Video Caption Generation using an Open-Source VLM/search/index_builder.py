"""
Builds the vector database index from your video captions.

Run this once after run_pipeline.py has produced video_report.csv, and
again any time your captions change:

    python index_builder.py
"""

import config
from caption_loader import CaptionLoader
from embedding_service import SentenceTransformerEmbeddingService
from vector_store import ChromaVectorStore


def build_index():
    documents = CaptionLoader.load(config.CAPTIONS_CSV)
    if not documents:
        print(f"No captions found in '{config.CAPTIONS_CSV}'. Run run_pipeline.py first.")
        return

    print(f"Loaded {len(documents)} video captions.")

    embedder = SentenceTransformerEmbeddingService(config.EMBEDDING_MODEL_NAME)
    store = ChromaVectorStore(config.CHROMA_PERSIST_DIR, config.COLLECTION_NAME)

    ids = [doc.video for doc in documents]
    texts = [doc.text for doc in documents]
    metadatas = [
        {"video": doc.video, "summary_caption": doc.summary_caption}
        for doc in documents
    ]

    print("Generating embeddings...")
    embeddings = embedder.embed(texts)

    print("Storing in vector database...")
    store.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

    print(f"Indexed {len(documents)} videos into collection '{config.COLLECTION_NAME}'.")


if __name__ == "__main__":
    build_index()