"""
Validation strategies for checking whether a generated caption correctly
identifies its expected subject.

Two independent signals are used:
  1. CaptionKeywordValidator - does the caption TEXT mention the expected
     subject (or a synonym of it)? Cheap, no extra API calls, but only as
     good as the synonym list.
  2. BoundingBoxValidator - does an object detector actually LOCATE the
     expected subject in the frame? More rigorous (ties the claim to a real
     region of the image) but costs an extra API call per video.

CombinedValidator runs both and produces one final classification, so the
report reflects agreement/disagreement between the two signals rather than
trusting either alone.
"""

from abc import ABC, abstractmethod
from typing import Dict, List

from PIL import Image

from models import CaptionRecord, ValidationResult, BoundingBox
from object_detector import ObjectDetector


class Validator(ABC):
    @abstractmethod
    def validate(self, record: CaptionRecord, frame: Image.Image) -> ValidationResult:
        raise NotImplementedError


class CaptionKeywordValidator(Validator):
    """Checks the caption text for the expected label or one of its synonyms."""

    def __init__(self, ground_truth: Dict[str, str], synonyms: Dict[str, List[str]]):
        self.ground_truth = ground_truth
        self.synonyms = synonyms

    def validate(self, record: CaptionRecord, frame: Image.Image) -> ValidationResult:
        expected_label = self.ground_truth.get(record.video)

        if not expected_label:
            return ValidationResult(
                video=record.video, expected_label="Not labeled",
                classification="Unclassified - add to GROUND_TRUTH",
                caption_match=False, caption_matched_term="",
                detection_match=False, detected_boxes={},
                notes="No ground truth defined for this video.",
            )

        combined_text = f"{record.detailed_caption} {record.summary_caption}".lower()
        terms = self.synonyms.get(expected_label, [expected_label])

        matched_term = next((t for t in terms if t.lower() in combined_text), "")
        matched = bool(matched_term)

        return ValidationResult(
            video=record.video, expected_label=expected_label,
            classification="True Positive" if matched else "False Positive",
            caption_match=matched, caption_matched_term=matched_term,
            detection_match=False, detected_boxes={},
        )


class BoundingBoxValidator(Validator):
    """Checks whether an ObjectDetector actually locates the expected subject
    in the frame, independent of what the caption text says."""

    def __init__(
        self,
        ground_truth: Dict[str, str],
        detection_query: Dict[str, str],
        detector: ObjectDetector,
    ):
        self.ground_truth = ground_truth
        self.detection_query = detection_query
        self.detector = detector

    def validate(self, record: CaptionRecord, frame: Image.Image) -> ValidationResult:
        expected_label = self.ground_truth.get(record.video)

        if not expected_label:
            return ValidationResult(
                video=record.video, expected_label="Not labeled",
                classification="Unclassified - add to GROUND_TRUTH",
                caption_match=False, caption_matched_term="",
                detection_match=False, detected_boxes={},
            )

        query = self.detection_query.get(expected_label, expected_label)
        boxes = self.detector.detect(frame, query)
        detected = len(boxes) > 0

        return ValidationResult(
            video=record.video, expected_label=expected_label,
            classification="True Positive" if detected else "False Negative (not detected)",
            caption_match=False, caption_matched_term="",
            detection_match=detected,
            detected_boxes={expected_label: boxes} if detected else {},
        )


class CombinedValidator(Validator):
    """Merges the keyword and bounding-box signals into one final verdict.

    Priority logic:
      - Detected AND caption mentions it   -> True Positive (both signals agree)
      - Detected but caption text misses it -> True Positive (visual proof is
        the stronger signal; caption wording is just imperfect)
      - Not detected but caption claims it -> False Positive (caption is
        likely hallucinating - nothing was actually located there)
      - Neither                             -> False Negative
    """

    def __init__(self, keyword_validator: CaptionKeywordValidator, box_validator: BoundingBoxValidator):
        self.keyword_validator = keyword_validator
        self.box_validator = box_validator

    def validate(self, record: CaptionRecord, frame: Image.Image) -> ValidationResult:
        keyword_result = self.keyword_validator.validate(record, frame)
        box_result = self.box_validator.validate(record, frame)

        if keyword_result.expected_label == "Not labeled":
            return keyword_result

        caption_match = keyword_result.caption_match
        detection_match = box_result.detection_match

        if detection_match:
            classification = "True Positive"
            notes = (
                "Confirmed by both caption and detection."
                if caption_match else
                "Detected in frame, but caption text did not explicitly name it."
            )
        elif caption_match:
            classification = "False Positive (caption only, not visually confirmed)"
            notes = "Caption mentions the subject, but no matching object was located in the frame."
        else:
            classification = "False Negative"
            notes = "Neither the caption nor object detection found the expected subject."

        return ValidationResult(
            video=record.video,
            expected_label=keyword_result.expected_label,
            classification=classification,
            caption_match=caption_match,
            caption_matched_term=keyword_result.caption_matched_term,
            detection_match=detection_match,
            detected_boxes=box_result.detected_boxes,
            notes=notes,
        )
