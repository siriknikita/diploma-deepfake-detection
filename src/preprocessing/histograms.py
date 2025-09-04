from typing import List

import cv2
import numpy as np
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

    # Split the entire image into channels once.
    b_channel, g_channel, r_channel = cv2.split(image)

    histograms: List[HistogramData] = []
    for y in range(0, h - window_size + 1, window_size):
        for x in range(0, w - window_size + 1, window_size):
            # Extract windows from the pre-split channels
            window_r = r_channel[y: y + window_size, x: x + window_size]
            window_g = g_channel[y: y + window_size, x: x + window_size]
            window_b = b_channel[y: y + window_size, x: x + window_size]

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
