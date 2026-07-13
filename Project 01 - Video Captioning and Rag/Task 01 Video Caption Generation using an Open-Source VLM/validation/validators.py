"""
Validation strategies for checking whether a generated caption correctly
identifies its expected subject(s).

A video can have MULTIPLE expected labels (e.g. a clip showing a technician
working on a generator expects BOTH "technician" and "generator" to be
identified). Each validator checks every expected label independently, then
rolls the per-label results up into one overall classification:

  - "All Matched (n/n)"      -> every expected label was found
  - "Partial Match (k/n)"    -> some, but not all, expected labels were found
  - "None Matched (0/n)"     -> none of the expected labels were found
  - "Unclassified"           -> no ground truth defined for this video

Two independent signals are combined:
  1. CaptionKeywordValidator - does the caption TEXT mention each expected
     label (or a synonym of it)?
  2. BoundingBoxValidator - does an object detector actually LOCATE each
     expected label in the frame?

CombinedValidator merges both per label, then produces the final rollup.
"""

from abc import ABC, abstractmethod
from typing import Dict, List

from PIL import Image

from models import CaptionRecord, ValidationResult, BoundingBox
from object_detector import ObjectDetector


def _rollup(matches: Dict[str, bool]) -> str:
    total = len(matches)
    matched = sum(1 for v in matches.values() if v)
    if total == 0:
        return "Unclassified"
    if matched == total:
        return f"All Matched ({matched}/{total})"
    if matched == 0:
        return f"None Matched (0/{total})"
    return f"Partial Match ({matched}/{total})"


class Validator(ABC):
    @abstractmethod
    def validate(self, record: CaptionRecord, frame: Image.Image) -> ValidationResult:
        raise NotImplementedError


class CaptionKeywordValidator(Validator):
    """Checks the caption text for each expected label or its synonyms."""

    def __init__(self, ground_truth: Dict[str, List[str]], synonyms: Dict[str, List[str]]):
        self.ground_truth = ground_truth
        self.synonyms = synonyms

    def validate(self, record: CaptionRecord, frame: Image.Image) -> ValidationResult:
        expected_labels = self.ground_truth.get(record.video, [])

        if not expected_labels:
            return ValidationResult(
                video=record.video, expected_labels=[],
                classification="Unclassified - add to GROUND_TRUTH",
                notes="No ground truth defined for this video.",
            )

        combined_text = f"{record.detailed_caption} {record.summary_caption}".lower()

        caption_matches, matched_terms = {}, {}
        for label in expected_labels:
            terms = self.synonyms.get(label, [label])
            matched_term = next((t for t in terms if t.lower() in combined_text), "")
            caption_matches[label] = bool(matched_term)
            matched_terms[label] = matched_term

        return ValidationResult(
            video=record.video, expected_labels=expected_labels,
            classification=_rollup(caption_matches),
            caption_matches=caption_matches, matched_terms=matched_terms,
        )


class BoundingBoxValidator(Validator):
    """Checks whether an ObjectDetector locates each expected label in the frame."""

    def __init__(
        self,
        ground_truth: Dict[str, List[str]],
        detection_query: Dict[str, str],
        detector: ObjectDetector,
    ):
        self.ground_truth = ground_truth
        self.detection_query = detection_query
        self.detector = detector

    def validate(self, record: CaptionRecord, frame: Image.Image) -> ValidationResult:
        expected_labels = self.ground_truth.get(record.video, [])

        if not expected_labels:
            return ValidationResult(
                video=record.video, expected_labels=[],
                classification="Unclassified - add to GROUND_TRUTH",
            )

        detection_matches, detected_boxes = {}, {}
        for label in expected_labels:
            query = self.detection_query.get(label, label)
            boxes = self.detector.detect(frame, query)
            detection_matches[label] = len(boxes) > 0
            if boxes:
                detected_boxes[label] = boxes

        return ValidationResult(
            video=record.video, expected_labels=expected_labels,
            classification=_rollup(detection_matches),
            detection_matches=detection_matches, detected_boxes=detected_boxes,
        )


class CombinedValidator(Validator):
    """Merges the keyword and bounding-box signals into one final verdict
    per expected label, then rolls up into an overall classification.

    Per-label priority logic:
      - Detected (regardless of caption wording) -> counts as matched
        (visual proof is the stronger signal)
      - Not detected but caption text claims it    -> counts as NOT matched,
        flagged in notes as a likely caption hallucination
      - Neither                                    -> not matched
    """

    def __init__(self, keyword_validator: CaptionKeywordValidator, box_validator: BoundingBoxValidator):
        self.keyword_validator = keyword_validator
        self.box_validator = box_validator

    def validate(self, record: CaptionRecord, frame: Image.Image) -> ValidationResult:
        keyword_result = self.keyword_validator.validate(record, frame)
        box_result = self.box_validator.validate(record, frame)

        if not keyword_result.expected_labels:
            return keyword_result

        final_matches: Dict[str, bool] = {}
        notes_parts: List[str] = []

        for label in keyword_result.expected_labels:
            caption_ok = keyword_result.caption_matches.get(label, False)
            detected = box_result.detection_matches.get(label, False)

            final_matches[label] = detected or caption_ok

            if detected and caption_ok:
                notes_parts.append(f"{label}: confirmed by caption and detection")
            elif detected and not caption_ok:
                notes_parts.append(f"{label}: detected in frame, but caption did not name it")
            elif caption_ok and not detected:
                notes_parts.append(f"{label}: caption claims it, but not visually confirmed (possible hallucination)")
            else:
                notes_parts.append(f"{label}: not found by either signal")

        return ValidationResult(
            video=record.video,
            expected_labels=keyword_result.expected_labels,
            classification=_rollup(final_matches),
            caption_matches=keyword_result.caption_matches,
            matched_terms=keyword_result.matched_terms,
            detection_matches=box_result.detection_matches,
            detected_boxes=box_result.detected_boxes,
            notes="; ".join(notes_parts),
        )