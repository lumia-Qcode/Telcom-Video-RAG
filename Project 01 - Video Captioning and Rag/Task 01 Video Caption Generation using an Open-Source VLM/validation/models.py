"""Data classes shared across the captioning and validation pipeline."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class CaptionRecord:
    """One video's caption output. `frames` is populated in-memory during a
    combined pipeline run (so validation can reuse the same extracted frames
    without re-reading the video file) - it is never written to CSV/Excel."""
    video: str
    detailed_caption: str
    summary_caption: str
    frame_captions: List[str] = field(default_factory=list)
    frames: List[Any] = field(default_factory=list, repr=False)


@dataclass
class BoundingBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass
class ValidationResult:
    """Final, combined verdict for one video after all validators have run.
    A video can have MULTIPLE expected labels (e.g. 'technician' AND
    'generator' in the same clip) - each is tracked independently, then
    rolled up into one overall classification for the report row."""
    video: str
    expected_labels: List[str]
    classification: str                          # overall: "All Matched" | "Partial Match (x/y)" | "None Matched" | "Unclassified"
    caption_matches: Dict[str, bool] = field(default_factory=dict)     # label -> matched in caption text
    matched_terms: Dict[str, str] = field(default_factory=dict)        # label -> the actual synonym found
    detection_matches: Dict[str, bool] = field(default_factory=dict)   # label -> found via object detection
    detected_boxes: Dict[str, List[BoundingBox]] = field(default_factory=dict)
    annotated_image_path: Optional[str] = None
    notes: str = ""