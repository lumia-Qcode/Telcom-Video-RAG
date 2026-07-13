"""
Unified pipeline: captioning + validation in a single run, producing ONE
combined CSV/Excel report (video, frame-by-frame captions, detailed analysis,
summary caption, and - if enabled - validation classification with the
annotated bounding-box image embedded).

Place this file at the project root, alongside the `captioning/` and
`validation/` folders.

TO SKIP VALIDATION (captioning only): comment out ONLY the second
RUN_VALIDATION line below. Nothing else needs to change.
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT_DIR, "captioning"))
sys.path.insert(0, os.path.join(ROOT_DIR, "validation"))

import config
from CaptionGenerator import generate_all_captions
from object_detector import MoondreamDetector
from image_annotator import ImageAnnotator
from validators import CaptionKeywordValidator, BoundingBoxValidator, CombinedValidator
from combined_report_builder import CombinedReportBuilder


RUN_VALIDATION = False
RUN_VALIDATION = True   # <- comment out ONLY this line to skip validation and run captioning only


def main():
    if not config.API_KEY:
        print("Please set MOONDREAM_API_KEY in your environment (.env file).")
        return

    if not os.path.isdir(config.VIDEO_DIR):
        print(f"Folder '{config.VIDEO_DIR}' not found.")
        return

    video_files = sorted(
        f for f in os.listdir(config.VIDEO_DIR)
        if f.lower().endswith(config.VIDEO_EXTENSIONS)
    )
    if not video_files:
        print(f"No videos found in '{config.VIDEO_DIR}'.")
        return

    print(f"--- Step 1: Captioning ({len(video_files)} videos) ---\n")
    records = generate_all_captions(config.VIDEO_DIR, video_files, config.FRAMES_PER_VIDEO)

    validation_results = {}

    if RUN_VALIDATION:
        print(f"\n--- Step 2: Validation ---\n")
        detector = MoondreamDetector(api_key=config.API_KEY)
        annotator = ImageAnnotator(config.ANNOTATED_FRAMES_DIR, config.CLASS_COLORS)
        keyword_validator = CaptionKeywordValidator(config.GROUND_TRUTH, config.SYNONYMS)
        box_validator = BoundingBoxValidator(config.GROUND_TRUTH, config.DETECTION_QUERY, detector)
        combined_validator = CombinedValidator(keyword_validator, box_validator)

        for record in records:
            print(f"Validating: {record.video}")
            rep_frame = record.frames[len(record.frames) // 2]
            result = combined_validator.validate(record, rep_frame)

            if result.detected_boxes:
                result.annotated_image_path = annotator.annotate(
                    rep_frame, result.detected_boxes, record.video
                )

            print(f"  Expected: {', '.join(result.expected_labels)} | Result: {result.classification}\n")
            validation_results[record.video] = result
    else:
        print("\n--- Validation skipped (RUN_VALIDATION is off) ---\n")

    builder = CombinedReportBuilder(records, validation_results, include_validation=RUN_VALIDATION)
    builder.build_csv(config.COMBINED_REPORT_CSV)
    builder.build_excel(config.COMBINED_REPORT_EXCEL)

    print(f"Combined CSV report saved to '{config.COMBINED_REPORT_CSV}'.")
    print(f"Combined Excel report saved to '{config.COMBINED_REPORT_EXCEL}'.")


if __name__ == "__main__":
    main()