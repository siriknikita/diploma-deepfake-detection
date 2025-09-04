from typing import Any, List

import cv2
import numpy as np
from cv2.typing import MatLike

from src.schemas.custom_types import HistogramData, ProjectSettings
from src.schemas.features import CNNHistogramFeatures


def compute_histograms_for_window(
    image: MatLike, cfg: ProjectSettings
) -> List[HistogramData]:
    """
    Computes histograms for a sliding window over the image.

    Args:
        image (cv2.typing.MatLike): The input image (aligned face).
        cfg (ProjectSettings): Configuration settings containing window size.

    Returns:
        List[HistogramData]: A list of dictionaries containing histogram data for each window.
    """
    h, w, _ = image.shape
    window_size = cfg.window_size

    # Split the entire image into channels once.
    b_channel, g_channel, r_channel = cv2.split(image)

    histograms: List[HistogramData] = []
    for y in range(0, h - window_size + 1, window_size):
        for x in range(0, w - window_size + 1, window_size):
            # Extract windows from the pre-split channels
            window_r = r_channel[y : y + window_size, x : x + window_size]
            window_g = g_channel[y : y + window_size, x : x + window_size]
            window_b = b_channel[y : y + window_size, x : x + window_size]

            # Use NumPy's bincount for a much faster histogram calculation.
            hist_r = np.bincount(window_r.flatten(), minlength=256).tolist()
            hist_g = np.bincount(window_g.flatten(), minlength=256).tolist()
            hist_b = np.bincount(window_b.flatten(), minlength=256).tolist()

            histograms.append(
                {
                    "window_x": x,
                    "window_y": y,
                    "hist_r": hist_r,
                    "hist_g": hist_g,
                    "hist_b": hist_b,
                }
            )

    return histograms


def concatenate_histograms_to_matrix(
    histograms: List[HistogramData],
) -> np.ndarray[Any, Any]:
    """
    Concatenates RGB histograms from multiple windows into a single matrix.

    For each window, concatenates the red, green, and blue channel histograms
    into a single feature vector of size 768 (256×3). The result is a matrix
    of shape (num_windows, 768) suitable for CNN input.

    Args:
        histograms (List[HistogramData]): List of histogram data dictionaries
            returned by compute_histograms_for_window.

    Returns:
        np.ndarray: Matrix of shape (num_windows, 768) where each row represents
            the concatenated RGB histograms for one window.
    """
    if not histograms:
        return np.array([])

    num_windows = len(histograms)
    feature_matrix = np.zeros((num_windows, 768), dtype=np.float64)

    for i, hist_data in enumerate(histograms):
        # Concatenate RGB histograms: [R, G, B] -> single vector of length 768
        hist_r = np.array(hist_data["hist_r"], dtype=np.float64)
        hist_g = np.array(hist_data["hist_g"], dtype=np.float64)
        hist_b = np.array(hist_data["hist_b"], dtype=np.float64)

        # Concatenate along the feature dimension
        concatenated_features = np.concatenate([hist_r, hist_g, hist_b])
        feature_matrix[i, :] = concatenated_features

    return feature_matrix


def get_histogram_matrix_for_cnn(
    image: MatLike, cfg: ProjectSettings
) -> np.ndarray[Any, Any]:
    """
    Computes histograms and returns them as a concatenated matrix for CNN input.

    This is a convenience function that combines compute_histograms_for_window
    and concatenate_histograms_to_matrix into a single call.

    Args:
        image (cv2.typing.MatLike): The input image (aligned face).
        cfg (ProjectSettings): Configuration settings containing window size.

    Returns:
        np.ndarray: Matrix of shape (num_windows, 768) ready for CNN input.
    """
    histograms = compute_histograms_for_window(image, cfg)
    return concatenate_histograms_to_matrix(histograms)


def get_cnn_histogram_features(
    image: MatLike, cfg: ProjectSettings
) -> CNNHistogramFeatures:
    """
    Computes histograms and returns them as a structured CNNHistogramFeatures object.

    This function provides the most complete interface, returning both the raw
    matrix and metadata about the features.

    Args:
        image (cv2.typing.MatLike): The input image (aligned face).
        cfg (ProjectSettings): Configuration settings containing window size.

    Returns:
        CNNHistogramFeatures: A structured object containing the feature matrix
            and metadata suitable for CNN input.
    """
    histogram_matrix = get_histogram_matrix_for_cnn(image, cfg)
    return CNNHistogramFeatures.from_histogram_matrix(histogram_matrix, cfg.window_size)
