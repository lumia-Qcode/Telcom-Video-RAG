"""Renders detected bounding boxes onto a frame image for visual review."""

import os
from typing import Dict, List

from PIL import Image, ImageDraw, ImageFont

from models import BoundingBox


class ImageAnnotator:
    def __init__(self, output_dir: str, class_colors: Dict[str, str]):
        self.output_dir = output_dir
        self.class_colors = class_colors
        os.makedirs(self.output_dir, exist_ok=True)

    def _font(self):
        try:
            return ImageFont.truetype("arial.ttf", 16)
        except Exception:
            return ImageFont.load_default()

    def annotate(
        self,
        image: Image.Image,
        boxes_by_label: Dict[str, List[BoundingBox]],
        video_name: str,
    ) -> str:
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        w, h = annotated.size
        font = self._font()

        for label, boxes in boxes_by_label.items():
            color = self.class_colors.get(label, "#FF0000")
            for box in boxes:
                x_min, y_min = box.x_min * w, box.y_min * h
                x_max, y_max = box.x_max * w, box.y_max * h

                draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=3)

                text_bbox = draw.textbbox((0, 0), label, font=font)
                text_w = text_bbox[2] - text_bbox[0]
                text_h = text_bbox[3] - text_bbox[1]
                label_y = max(0, y_min - text_h - 4)
                draw.rectangle([x_min, label_y, x_min + text_w + 6, y_min], fill=color)
                draw.text((x_min + 3, label_y), label, fill="white", font=font)

        base_name = os.path.splitext(video_name)[0]
        out_path = os.path.join(self.output_dir, f"{base_name}_boxes.jpg")
        annotated.save(out_path, "JPEG", quality=90)
        return out_path