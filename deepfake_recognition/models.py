from typing import Any
from pydantic import BaseModel
from .types import NumpyArray


class DetectedFeatures(BaseModel):
    """
    A Pydantic model to represent detected features with NumPy arrays.

    The fields `boxes` and `landmarks` are typed using `NumpyArray` | None,
    which allows them to be either a NumPy array or `None`.
    """
    boxes: Any | NumpyArray | None
    landmarks: Any | NumpyArray | None
