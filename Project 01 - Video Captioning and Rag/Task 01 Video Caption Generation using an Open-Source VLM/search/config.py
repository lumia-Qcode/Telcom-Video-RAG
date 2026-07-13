"""Configuration for the semantic video search module."""

import os

# This file lives in <project_root>/search/config.py
SEARCH_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SEARCH_DIR)

# Reuses the combined report your captioning/validation pipeline already produces
CAPTIONS_CSV = os.path.join(PROJECT_ROOT, "video_report.csv")
VIDEO_DIR = os.path.join(PROJECT_ROOT, "Videos")

# ChromaDB persists to disk here - the index survives between runs, you only
# need to rebuild it when your captions change (re-run index_builder.py).
CHROMA_PERSIST_DIR = os.path.join(SEARCH_DIR, "chroma_store")
COLLECTION_NAME = "video_captions"

# Local, free, open-source embedding model - runs fine on CPU for this
# amount of data (a handful of short captions).
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K_RESULTS = 3

FLASK_HOST = "127.0.0.1"
FLASK_PORT = 5000