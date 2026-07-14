"""
Central configuration for the video-report RAG pipeline.
Edit paths/model names here; everything else reads from this file.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(ROOT_DIR, "video_report.csv")
CHROMA_DIR = os.path.join(ROOT_DIR, "chroma_db")
BM25_INDEX_PATH = os.path.join(ROOT_DIR, "bm25_index.pkl")
COLLECTION_NAME = "video_reports"

# Folder containing the actual .mp4/.mov/etc files, so the frontend can play them back.
# Defaults to a "Videos" folder next to this file - copy/symlink your Videos folder here,
# or point VIDEOS_DIR at wherever they already live (e.g. the same folder run_pipeline.py used).
VIDEOS_DIR = os.getenv("VIDEOS_DIR", os.path.join(ROOT_DIR, "Videos"))

# --- LLM ---
# "groq" = free (no credit card, generous rate limits, fast). "anthropic" = paid, higher quality.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")  # free tier, strong quality

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

LLM_MAX_TOKENS = 1200

# --- Embeddings ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # fast, solid general-purpose sentence embedder

# --- Retrieval ---
SEMANTIC_TOP_K = 12
BM25_TOP_K = 12
FINAL_TOP_K = 10         # how many detailed chunks get sent to the LLM after fusion
RRF_K = 60               # reciprocal rank fusion constant

# --- Known label vocabulary (mirrors validation/config.py GROUND_TRUTH categories) ---
LABEL_SYNONYMS = {
    "substation": ["substation", "transformer", "insulator", "electrical grid", "power grid"],
    "cell tower": ["cell tower", "telecom tower", "communication tower", "antenna", "mast", "cellular", "telecom antenna"],
    "technician": ["technician", "worker", "electrician", "person", "engineer", "workers", "utility worker"],
    "utility tower": ["utility tower", "utility pole", "transmission tower", "power line", "electrical grid"],
    "solar panel": ["solar panel", "solar", "photovoltaic"],
    "battery": ["battery", "battery bank", "battery pack", "battery rack", "electrical box", "utility cabinet"],
    "power plant": ["power plant", "power station", "industrial plant", "cooling tower", "factory", "plant"],
    "radio telescope": ["radio telescope", "satellite dish", "dish antenna", "parabolic dish", "observatory"],
    "robotic arm": ["robotic arm", "industrial arm", "automation arm", "robot"],
    "conveyor belt": ["conveyor belt", "conveyor", "assembly line"],
    "machine": ["machine", "equipment", "machinery", "industrial equipment", "generator"],
}

# Anchor "today" for relative date parsing ("last week", "yesterday", etc.)
# Matches the anchor used when synthetic Date/Time columns were generated.
from datetime import datetime  # noqa: E402
TODAY = datetime(2026, 7, 14)
