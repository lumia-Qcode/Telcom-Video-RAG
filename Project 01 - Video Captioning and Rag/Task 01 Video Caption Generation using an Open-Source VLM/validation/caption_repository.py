"""Reads the captions CSV produced by the (separate) captioning step."""

import csv
from typing import List

from models import CaptionRecord


class CaptionRepository:
    """Loads previously generated captions so validation can run as an
    independent step, without re-calling the captioning model."""

    @staticmethod
    def load(csv_path: str) -> List[CaptionRecord]:
        records: List[CaptionRecord] = []

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            frame_columns = [c for c in reader.fieldnames if c.startswith("Frame")]

            for row in reader:
                records.append(CaptionRecord(
                    video=row["Video"],
                    detailed_caption=row.get("Detailed Analysis", row.get("Final Caption", "")),
                    summary_caption=row.get("Summary_Caption", row.get("Output_Caption", "")),
                    frame_captions=[row[c] for c in frame_columns if row.get(c)],
                ))

        return records
