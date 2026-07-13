"""Data classes used across the semantic search module."""

from dataclasses import dataclass


@dataclass
class VideoCaptionDocument:
    """One video's caption, ready to be embedded and stored."""
    video: str
    text: str                # combined text actually used for the embedding
    summary_caption: str
    detailed_caption: str


@dataclass
class SearchResult:
    video: str
    summary_caption: str
    score: float              # cosine similarity, higher = more relevant