from typing import Any

import numpy as np
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


class CNNHistogramFeatures(BaseModel):
    """
    A Pydantic model to represent concatenated histogram features for CNN input.

    The concatenated matrix combines RGB channel histograms for each window,
    resulting in a feature matrix of shape (num_windows, 768) where each row
    represents the concatenated [R, G, B] histograms for one window.

    Attributes:
        feature_matrix (NumpyArray): Matrix of shape (num_windows, 768) where
            each row contains concatenated RGB histograms for one window.
        num_windows (int): Number of windows processed.
        feature_dimension (int): Feature dimension per window (768 = 256×3 RGB channels).
        window_size (int): Size of the sliding window used.
    """

    feature_matrix: np.ndarray[Any, Any]
    num_windows: int
    feature_dimension: int
    window_size: int

    @classmethod
    def from_histogram_matrix(
        cls, matrix: np.ndarray[Any, Any], window_size: int
    ) -> "CNNHistogramFeatures":
        """
        Create CNNHistogramFeatures from a histogram matrix.

        Args:
            matrix (np.ndarray[Any, Any]): The concatenated histogram matrix.
            window_size (int): The window size used for histogram computation.

        Returns:
            CNNHistogramFeatures: A new instance with computed metadata.
        """
        if matrix.size == 0:
            return cls(
                feature_matrix=matrix,
                num_windows=0,
                feature_dimension=0,
                window_size=window_size,
            )

        return cls(
            feature_matrix=matrix,
            num_windows=matrix.shape[0],
            feature_dimension=matrix.shape[1],
            window_size=window_size,
        )
