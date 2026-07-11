"""Data classes shared across the validation pipeline."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class CaptionRecord:
    """One row of already-generated caption output for a single video."""
    video: str
    detailed_caption: str
    summary_caption: str
    frame_captions: List[str] = field(default_factory=list)


@dataclass
class BoundingBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass
class ValidationResult:
    """Final, combined verdict for one video after all validators have run."""
    video: str
    expected_label: str
    classification: str                 # "True Positive" | "False Positive" | "False Negative" | "Unclassified"
    caption_match: bool
    caption_matched_term: str
    detection_match: bool
    detected_boxes: Dict[str, List[BoundingBox]]
    annotated_image_path: Optional[str] = None
    notes: str = ""
