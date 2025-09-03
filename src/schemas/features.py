from typing import Any

from pydantic import BaseModel

from src.schemas.custom_types import NumpyArray


class DetectedFeatures(BaseModel):
    """
    A Pydantic model to represent detected features with NumPy arrays.

    The fields `boxes` and `landmarks` are typed using `NumpyArray` | None,
    which allows them to be either a NumPy array or `None`.

    Attributes:
        boxes (Any | NumpyArray | None): Detected bounding boxes as a NumPy array or None.
        landmarks (Any | NumpyArray | None): Detected landmarks as a NumPy array or None.
    """

    boxes: Any | NumpyArray | None
    landmarks: Any | NumpyArray | None
