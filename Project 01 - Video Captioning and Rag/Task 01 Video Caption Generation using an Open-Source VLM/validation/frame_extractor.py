"""Video-to-frame extraction, isolated from captioning/detection concerns."""

import os
from typing import List

import cv2
from PIL import Image


class FrameExtractor:
    """Extracts evenly spaced frames from a video file as PIL Images."""

    def __init__(self, video_dir: str):
        self.video_dir = video_dir

    def path_for(self, video_name: str) -> str:
        return os.path.join(self.video_dir, video_name)

    def extract(self, video_name: str, num_frames: int = 3) -> List[Image.Image]:
        path = self.path_for(video_name)
        abs_path = os.path.abspath(path)

        if not os.path.exists(path):
            print(f"    [diagnostic] File does not exist at resolved path: {abs_path}")
            print(f"    [diagnostic] Current working directory: {os.getcwd()}")
            if os.path.isdir(self.video_dir):
                actual_files = os.listdir(self.video_dir)
                print(f"    [diagnostic] Files actually found in '{self.video_dir}':")
                for f in actual_files:
                    print(f"        - {f!r}")
            else:
                print(f"    [diagnostic] Video directory '{self.video_dir}' does not exist "
                      f"relative to the current working directory.")
            return []

        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            print(f"    [diagnostic] File exists but OpenCV could not read frame count "
                  f"(possible unsupported codec): {abs_path}")
            return []

        indices = [int(total * (i + 1) / (num_frames + 1)) for i in range(num_frames)]
        frames: List[Image.Image] = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))

        cap.release()
        return frames

    def representative_frame(self, video_name: str, num_frames: int = 3) -> Image.Image:
        frames = self.extract(video_name, num_frames)
        if not frames:
            raise ValueError(f"Could not extract any frames from '{video_name}'.")
        return frames[len(frames) // 2]