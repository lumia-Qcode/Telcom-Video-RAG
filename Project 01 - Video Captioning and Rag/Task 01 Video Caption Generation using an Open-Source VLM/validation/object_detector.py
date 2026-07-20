"""
Object detection abstraction.

An abstract base class is used here deliberately: it lets the validation
logic depend on "some object detector" rather than on Moondream specifically.
If the detection backend is ever swapped (a different vision API, a locally
hosted model, etc.), only a new subclass needs to be written - nothing in
validators.py or main.py has to change.
"""

from abc import ABC, abstractmethod
from typing import List

from PIL import Image

from models import BoundingBox


class ObjectDetector(ABC):
    """Interface for any backend that can locate an object described in
    natural language within an image."""

    @abstractmethod
    def detect(self, image: Image.Image, object_query: str) -> List[BoundingBox]:
        """Return all bounding boxes matching `object_query` in `image`.
        Coordinates are normalized (0-1) relative to image width/height."""
        raise NotImplementedError


class MoondreamDetector(ObjectDetector):
    """Object detector backed by the Moondream Cloud API's detect() endpoint."""

    def __init__(self, api_key: str):
        import moondream as md
        self._model = md.vl(api_key=api_key)

    def detect(self, image: Image.Image, object_query: str) -> List[BoundingBox]:
        result = self._model.detect(image, object_query)
        objects = result.get("objects", [])
        return [
            BoundingBox(
                x_min=obj["x_min"], y_min=obj["y_min"],
                x_max=obj["x_max"], y_max=obj["y_max"],
            )
            for obj in objects
        ]