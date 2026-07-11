"""
Caption/Detection Validation Pipeline - entry point.

Runs entirely separately from the captioning step: it reads the CSV that the
captioning script already produced, and does not call the captioning model
again. It only calls the detection endpoint, once per video, to check the
expected subject is actually present.

Usage:
    python main.py
"""

import config
from caption_repository import CaptionRepository
from frame_extractor import FrameExtractor
from object_detector import MoondreamDetector
from image_annotator import ImageAnnotator
from validators import CaptionKeywordValidator, BoundingBoxValidator, CombinedValidator
from report_builder import ReportBuilder


def main():
    if not config.API_KEY:
        print("Please set MOONDREAM_API_KEY in your environment (.env file).")
        return

    records = CaptionRepository.load(config.CAPTIONS_CSV)
    if not records:
        print(f"No caption records found in '{config.CAPTIONS_CSV}'.")
        return

    frame_extractor = FrameExtractor(config.VIDEO_DIR)
    detector = MoondreamDetector(api_key=config.API_KEY)
    annotator = ImageAnnotator(config.ANNOTATED_FRAMES_DIR, config.CLASS_COLORS)

    keyword_validator = CaptionKeywordValidator(config.GROUND_TRUTH, config.SYNONYMS)
    box_validator = BoundingBoxValidator(config.GROUND_TRUTH, config.DETECTION_QUERY, detector)
    combined_validator = CombinedValidator(keyword_validator, box_validator)

    results = []

    for record in records:
        print(f"Validating: {record.video}")

        try:
            frame = frame_extractor.representative_frame(record.video, config.FRAMES_PER_VIDEO)
        except ValueError as e:
            print(f"  {e}")
            continue

        result = combined_validator.validate(record, frame)

        if result.detected_boxes:
            result.annotated_image_path = annotator.annotate(frame, result.detected_boxes, record.video)

        print(f"  Expected: {result.expected_label} | Result: {result.classification}")
        print(f"  {result.notes}\n")

        results.append(result)

    ReportBuilder(results).build_csv(config.REPORT_CSV)
    ReportBuilder(results).build_excel(config.REPORT_EXCEL)

    print(f"CSV report saved to '{config.REPORT_CSV}'.")
    print(f"Excel report saved to '{config.REPORT_EXCEL}'.")

    print_summary(results)


def print_summary(results):
    if not results:
        return
    tp = sum(1 for r in results if r.classification == "True Positive")
    fp = sum(1 for r in results if "False Positive" in r.classification)
    fn = sum(1 for r in results if "False Negative" in r.classification)

    print("=" * 90)
    print(f"{'Video'.ljust(45)} | {'Expected'.ljust(15)} | Result")
    print("-" * 90)
    for r in results:
        print(f"{r.video[:45].ljust(45)} | {r.expected_label[:15].ljust(15)} | {r.classification}")
    print("-" * 90)
    print(f"Total: {len(results)}  |  True Positives: {tp}  |  False Positives: {fp}  |  False Negatives: {fn}")
    print("=" * 90)


if __name__ == "__main__":
    main()
