import cv2
import numpy as np

from src.schemas.types import ProjectSettings
from src.utils.math import round_up_to_multiple


def align_and_square_face(image, face_box, landmarks, cfg: ProjectSettings):
    """
    Aligns a face from the original image based on eye landmarks, squares the crop,
    pads it to be a multiple of the window size, and then resizes the final crop.

    Args:
        image (PIL.Image.Image): The full, original input image.
        face_box (np.ndarray): The bounding box of the face [x1, y1, x2, y2].
        landmarks (np.ndarray): Array of facial landmarks relative to the box.
        window_size (int): The size of the sliding window.
        padding (float): The padding percentage.

    Returns:
        numpy.ndarray: The aligned, squared, and resized face image (BGR format).
    """
    bgr_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    h_orig, w_orig, _ = bgr_image.shape

    x1_box, y1_box, x2_box, y2_box = face_box
    w_box, h_box = x2_box - x1_box, y2_box - y1_box

    # Center of face in the original image coordinates
    cx = (x1_box + x2_box) // 2
    cy = (y1_box + y2_box) // 2

    window_size = cfg.window_size
    padding = cfg.padding

    # Step 1: calculate required square size with padding
    size = max(w_box, h_box)
    size = int(size * (1 + padding))

    # Step 2: make sure size is a multiple of WINDOW_SIZE
    size = round_up_to_multiple(size, window_size)

    # Step 3: landmarks for alignment
    # Convert landmarks to original image coordinates
    left_eye_orig = (landmarks[0][0] + x1_box, landmarks[0][1] + y1_box)
    right_eye_orig = (landmarks[1][0] + x1_box, landmarks[1][1] + y1_box)

    dx = right_eye_orig[0] - left_eye_orig[0]
    dy = right_eye_orig[1] - left_eye_orig[1]
    angle = np.degrees(np.arctan2(dy, dx))

    # Step 4: rotate the entire image
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(bgr_image, M, (w_orig, h_orig))

    # Step 5: get new bounding box coordinates after rotation
    new_x1 = cx - size // 2
    new_y1 = cy - size // 2
    new_x2 = cx + size // 2
    new_y2 = cy + size // 2

    # Step 6: calculate padding required to keep the full square
    pad_left = max(0, -new_x1)
    pad_top = max(0, -new_y1)
    pad_right = max(0, new_x2 - rotated.shape[1])
    pad_bottom = max(0, new_y2 - rotated.shape[0])

    # Step 7: apply padding to the rotated image
    padded = cv2.copyMakeBorder(
        rotated,
        int(pad_top), int(pad_bottom), int(pad_left), int(pad_right),
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )

    # Step 8: crop the image
    crop_x1 = int(new_x1 + pad_left)
    crop_y1 = int(new_y1 + pad_top)
    crop_x2 = int(new_x2 + pad_left)
    crop_y2 = int(new_y2 + pad_top)

    final_crop = padded[crop_y1:crop_y2, crop_x1:crop_x2]

    # Step 9: Guarantee the final dimensions are a multiple of window size with a resize
    final_resized_crop = cv2.resize(
        final_crop, (size, size), interpolation=cv2.INTER_AREA)

    # Final check
    if final_resized_crop.shape[0] % window_size != 0 or final_resized_crop.shape[1] % window_size != 0:
        print(f"Warning: Final crop size {final_resized_crop.shape} is not a multiple of {window_size}")

    return final_resized_crop
