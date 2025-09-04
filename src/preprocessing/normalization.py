import cv2
import numpy as np
import PIL.Image
from cv2.typing import MatLike
from numpy.typing import NDArray

from src.schemas.custom_types import ProjectSettings
from src.utils.math import round_up_to_multiple


def align_face(
    image: MatLike,
    face_box: NDArray[np.int_],
    landmarks: NDArray[np.int_],
) -> MatLike:
    """
    Aligns a face in the image based on eye landmarks.

    Args:
        image (MatLike): The input image in BGR format.
        face_box (np.ndarray): The bounding box of the face [x1, y1, x2, y2].
        landmarks (np.ndarray): Array of facial landmarks relative to the box.

    Returns:
        MatLike: The rotated image with the face aligned.
    """
    x1_box, y1_box, x2_box, y2_box = face_box
    cx = (x1_box + x2_box) // 2
    cy = (y1_box + y2_box) // 2

    # Convert landmarks to original image coordinates
    original_left_eye = (landmarks[0][0] + x1_box, landmarks[0][1] + y1_box)
    original_right_eye = (landmarks[1][0] + x1_box, landmarks[1][1] + y1_box)

    # Calculate rotation angle
    dx = original_right_eye[0] - original_left_eye[0]
    dy = original_right_eye[1] - original_left_eye[1]
    rotation_angle = np.degrees(np.arctan2(dy, dx))

    # Rotate the image
    rotation_matrix = cv2.getRotationMatrix2D((cx, cy), rotation_angle, 1.0)
    height, width, _ = image.shape
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    return rotated_image


def square_and_resize_face(
    image: MatLike,
    face_box: NDArray[np.int_],
    cfg: ProjectSettings,
) -> MatLike:
    """
    Squares the face crop, applies padding, and resizes it to the desired size.

    Args:
        image (MatLike): The rotated image with the face aligned.
        face_box (np.ndarray): The bounding box of the face [x1, y1, x2, y2].
        cfg (ProjectSettings): Configuration settings including window size and padding.

    Returns:
        MatLike: The squared and resized face image.
    """
    x1_box, y1_box, x2_box, y2_box = face_box
    window_size = cfg.window_size
    padding = cfg.padding

    cx = (x1_box + x2_box) // 2
    cy = (y1_box + y2_box) // 2

    w_box, h_box = x2_box - x1_box, y2_box - y1_box

    # Calculate square size with padding
    size = max(w_box, h_box)
    size = int(size * (1 + padding))
    size = round_up_to_multiple(number=size, base=window_size)

    # Calculate new bounding box
    new_x1 = cx - size // 2
    new_y1 = cy - size // 2
    new_x2 = cx + size // 2
    new_y2 = cy + size // 2

    # Calculate padding
    pad_left = max(0, -new_x1)
    pad_top = max(0, -new_y1)
    pad_right = max(0, new_x2 - image.shape[1])
    pad_bottom = max(0, new_y2 - image.shape[0])

    # Apply padding
    padded = cv2.copyMakeBorder(
        image,
        int(pad_top),
        int(pad_bottom),
        int(pad_left),
        int(pad_right),
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )

    # Crop the image
    crop_x1 = int(new_x1 + pad_left)
    crop_y1 = int(new_y1 + pad_top)
    crop_x2 = int(new_x2 + pad_left)
    crop_y2 = int(new_y2 + pad_top)
    final_crop = padded[crop_y1:crop_y2, crop_x1:crop_x2]

    # Resize to final dimensions
    final_resized_crop = cv2.resize(
        final_crop, (size, size), interpolation=cv2.INTER_AREA
    )

    return final_resized_crop


def normalize_face(
    image: PIL.Image.Image,
    face_box: NDArray[np.int_],
    landmarks: NDArray[np.int_],
    cfg: ProjectSettings,
) -> MatLike:
    """
    Aligns and squares a face from the original image.

    Args:
        image (PIL.Image.Image): The full, original input image.
        face_box (np.ndarray): The bounding box of the face [x1, y1, x2, y2].
        landmarks (np.ndarray): Array of facial landmarks relative to the box.
        cfg (ProjectSettings): Configuration settings including window size and padding.

    Returns:
        MatLike: The aligned, squared, and resized face image (BGR format).
    """
    bgr_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    aligned_image = align_face(bgr_image, face_box, landmarks)
    squared_image = square_and_resize_face(aligned_image, face_box, cfg)

    return squared_image
