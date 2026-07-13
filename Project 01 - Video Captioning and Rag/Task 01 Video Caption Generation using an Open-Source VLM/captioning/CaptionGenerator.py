"""
Captioning step, refactored as a reusable function instead of a standalone
script. Used by run_pipeline.py to generate captions AND keep the extracted
frames in memory, so the validation step (if enabled) does not need to
re-extract frames from the video files or re-call the captioning model.
"""

from typing import List

import moondream as md

import config
from models import CaptionRecord
from frame_extractor import FrameExtractor


DETAILED_PROMPT = (
    "Describe this frame in detail. Identify any people present, describing their "
    "clothing, physical appearance, what they are doing, and any specific tools or "
    "equipment they are handling. Explicitly look for and describe any infrastructure, "
    "including generators, power plants, utility towers, cell towers, battery banks, "
    "or electrical components. Describe the surrounding environment, weather conditions, "
    "and geography accurately."
)
SHORT_PROMPT = (
    "Summarize the main activity, the primary subjects, and key infrastructure elements "
    "(such as towers, power equipment, or machinery) seen in this frame in one descriptive sentence."
)
FRAME_DIAGNOSTIC_PROMPT = (
    "List the main objects, infrastructure assets, and human actions visible in this "
    "specific frame as a clear, straightforward sentence."
)


def _merge(captions: List[str]) -> str:
    """Concatenate captions, dropping exact duplicates, preserving order."""
    seen, merged = set(), []
    for c in captions:
        if c not in seen:
            merged.append(c)
            seen.add(c)
    return " ".join(merged)


def _pick_best(captions: List[str]) -> str:
    """Frames from the same clip often paraphrase the same scene - pick the
    single most descriptive one instead of concatenating near-duplicates."""
    if not captions:
        return ""
    return max(captions, key=len).strip()


class CaptionGenerator:
    """Wraps the Moondream Cloud API for per-frame captioning."""

    def __init__(self, api_key: str):
        self._model = md.vl(api_key=api_key)

    def generate(self, frames) -> tuple:
        """Returns (detailed_caption, summary_caption, frame_captions) for one video."""
        detailed_answers, short_answers, frame_captions = [], [], []

        for frame in frames:
            detailed_res = self._model.query(frame, DETAILED_PROMPT)
            detailed_answers.append(detailed_res["answer"].strip())

            short_res = self._model.query(frame, SHORT_PROMPT)
            short_answers.append(short_res["answer"].strip())

            diag_res = self._model.query(frame, FRAME_DIAGNOSTIC_PROMPT)
            frame_captions.append(diag_res["answer"].strip())

        return _merge(detailed_answers), _pick_best(short_answers), frame_captions


def generate_all_captions(video_dir: str, video_files: List[str], frames_per_video: int) -> List[CaptionRecord]:
    """Runs captioning for every video and returns CaptionRecord objects with
    their extracted frames attached (record.frames), ready for validation to
    reuse without touching the video files again."""

    generator = CaptionGenerator(api_key=config.API_KEY)
    extractor = FrameExtractor(video_dir)

    records: List[CaptionRecord] = []

    for vfile in video_files:
        print(f"Captioning: {vfile}")
        frames = extractor.extract(vfile, frames_per_video)
        if not frames:
            print(f"  Could not extract frames from '{vfile}', skipping.\n")
            continue

        detailed, summary, frame_caps = generator.generate(frames)

        record = CaptionRecord(
            video=vfile,
            detailed_caption=detailed,
            summary_caption=summary,
            frame_captions=frame_caps,
        )
        record.frames = frames
        records.append(record)

        print(f"  -> Summary: {summary}\n")

    return records