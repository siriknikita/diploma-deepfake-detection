from typing import List

import cv2
from cv2.typing import MatLike

from src.schemas.custom_types import HistogramData, ProjectSettings


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
    histograms: List[HistogramData] = []

    for y in range(0, h - window_size + 1, window_size):
        for x in range(0, w - window_size + 1, window_size):
            window = image[y : y + window_size, x : x + window_size]
            b, g, r = cv2.split(window)

            hist_r = cv2.calcHist([r], [0], None, [256], [0, 256]).flatten()
            hist_g = cv2.calcHist([g], [0], None, [256], [0, 256]).flatten()
            hist_b = cv2.calcHist([b], [0], None, [256], [0, 256]).flatten()

            histograms.append(
                {
                    "window_x": x,
                    "window_y": y,
                    "hist_r": hist_r.tolist(),
                    "hist_g": hist_g.tolist(),
                    "hist_b": hist_b.tolist(),
                }
            )

    return histograms
