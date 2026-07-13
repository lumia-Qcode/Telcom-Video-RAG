"""
Loads video captions from video_report.csv - the same combined report your
captioning/validation pipeline (run_pipeline.py) already produces. No new
data collection step is needed; this just reads what you already have.
"""

import csv
from typing import List

from models import VideoCaptionDocument


class CaptionLoader:
    @staticmethod
    def load(csv_path: str) -> List[VideoCaptionDocument]:
        documents: List[VideoCaptionDocument] = []

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                video = (row.get("Video") or "").strip()
                if not video:
                    # Blank row marks the start of the appended metrics
                    # summary section at the bottom of the CSV - stop here.
                    break

                summary = (row.get("Summary Caption") or "").strip()
                detailed = (row.get("Detailed Analysis") or "").strip()
                combined_text = f"{summary} {detailed}".strip()

                documents.append(VideoCaptionDocument(
                    video=video,
                    text=combined_text,
                    summary_caption=summary,
                    detailed_caption=detailed,
                ))

        return documents