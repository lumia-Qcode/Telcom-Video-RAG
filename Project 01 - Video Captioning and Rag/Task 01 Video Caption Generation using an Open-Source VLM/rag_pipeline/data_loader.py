"""
Parses video_report.csv into a list of VideoRecord objects.

The CSV has one row per video, followed by a blank row and then a
'--- Summary Metrics ---' footer block (overall Precision/Recall/F1/etc).
This loader stops reading video rows at that blank line and separately
parses the footer into a dict of overall metrics.
"""

import csv
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import config


@dataclass
class VideoRecord:
    video: str
    detailed_analysis: str
    summary_caption: str
    frame_captions: List[str]
    expected_labels: List[str]
    classification: str
    caption_matches: str
    detection_matches: str
    verdict: str
    notes: str
    date: Optional[str]   # "YYYY-MM-DD"
    time: Optional[str]   # "HH:MM"
    datetime_obj: Optional[datetime] = None
    label_flags: Dict[str, bool] = field(default_factory=dict)

    def has_label(self, label: str) -> bool:
        return self.label_flags.get(label, False)


def _parse_dt(date_str: str, time_str: str) -> Optional[datetime]:
    if not date_str or not time_str:
        return None
    try:
        return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
    except ValueError:
        return None


def load_video_records(csv_path: str = None) -> List[VideoRecord]:
    csv_path = csv_path or config.CSV_PATH

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    if not rows:
        return []

    header = rows[0]
    idx = {name: i for i, name in enumerate(header)}
    frame_cols = sorted(
        [c for c in header if c.startswith("Frame ") and c.endswith("Caption")],
        key=lambda c: int(c.split(" ")[1]),
    )

    records: List[VideoRecord] = []
    for row in rows[1:]:
        if not row or not row[0].strip():
            break  # hit the blank line before the metrics footer

        def get(col: str) -> str:
            i = idx.get(col)
            return row[i].strip() if i is not None and i < len(row) else ""

        expected_labels = [l.strip() for l in get("Expected Labels").split(",") if l.strip()]
        frame_captions = [get(c) for c in frame_cols if get(c)]
        date_str, time_str = get("Date"), get("Time")

        label_flags = {label: (label in expected_labels) for label in config.LABEL_SYNONYMS}

        records.append(
            VideoRecord(
                video=get("Video"),
                detailed_analysis=get("Detailed Analysis"),
                summary_caption=get("Summary Caption"),
                frame_captions=frame_captions,
                expected_labels=expected_labels,
                classification=get("Classification"),
                caption_matches=get("Caption Matches"),
                detection_matches=get("Detection Matches"),
                verdict=get("Verdict"),
                notes=get("Notes"),
                date=date_str or None,
                time=time_str or None,
                datetime_obj=_parse_dt(date_str, time_str),
                label_flags=label_flags,
            )
        )

    return records


def load_overall_metrics(csv_path: str = None) -> Dict[str, str]:
    """Parses the '--- Summary Metrics ---' footer block into a dict."""
    csv_path = csv_path or config.CSV_PATH
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    metrics = {}
    in_footer = False
    for row in rows:
        if row and row[0].strip() == "--- Summary Metrics ---":
            in_footer = True
            continue
        if in_footer and len(row) >= 2 and row[0].strip():
            metrics[row[0].strip()] = row[1].strip()

    return metrics


def build_document_text(r: VideoRecord) -> str:
    """Single text blob per video used for embedding + BM25 + LLM context."""
    parts = [
        f"Video: {r.video}",
        f"Date/Time: {r.date or 'unknown'} {r.time or ''}".strip(),
        f"Expected labels: {', '.join(r.expected_labels) if r.expected_labels else 'none'}",
        f"Classification: {r.classification or 'n/a'}",
        f"Summary: {r.summary_caption}",
        f"Detailed analysis: {r.detailed_analysis}",
    ]
    if r.frame_captions:
        for i, fc in enumerate(r.frame_captions, start=1):
            parts.append(f"Frame {i} caption: {fc}")
    if r.verdict:
        parts.append(f"Per-label verdict: {r.verdict}")
    if r.notes:
        parts.append(f"Notes: {r.notes}")
    return "\n".join(parts)


if __name__ == "__main__":
    recs = load_video_records()
    print(f"Loaded {len(recs)} video records.")
    for r in recs[:2]:
        print("-" * 60)
        print(build_document_text(r))
    print("=" * 60)
    print("Overall metrics:", load_overall_metrics())
