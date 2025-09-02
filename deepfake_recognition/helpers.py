from facenet_pytorch import MTCNN
from PIL import Image

import os

from typing import Any, Literal, overload
from deepfake_recognition.models import DetectedFeatures


@overload
def detect_and_save_cropped_faces(
    image_path: str,
    output_dir: str = "./output/",
    with_output: bool = False,
    with_landmarks: bool = False,
    *,
    as_dict: Literal[False],
    as_json: Literal[True],
) -> str: ...


@overload
def detect_and_save_cropped_faces(
    image_path: str,
    output_dir: str = "./output/",
    with_output: bool = False,
    with_landmarks: bool = False,
    *,
    as_dict: Literal[True],
    as_json: Literal[False],
) -> dict[str, Any]: ...


@overload
def detect_and_save_cropped_faces(
    image_path: str,
    output_dir: str = "./output/",
    with_output: bool = False,
    with_landmarks: bool = False,
    *,
    as_dict: Literal[False] = False,
    as_json: Literal[False] = False,
) -> DetectedFeatures: ...


def detect_and_save_cropped_faces(
    image_path,
    output_dir='./output/',
    with_output=False,
    with_landmarks=False,
    as_dict=True,
    as_json=False
) -> dict[str, Any] | str | DetectedFeatures:
    """
    Detects faces in an image, crops each face, and saves them to a directory.

    Args:
        image_path (str): The path to the input image file.
        output_dir (str): The directory to save the cropped faces.
        with_output (bool): If True, saves the cropped faces; if False, only returns bounding boxes.
        with_landmarks (bool): If True, returns facial landmarks along with bounding boxes.
        as_dict (bool): If True, returns results as a dictionary.
        as_json (bool): If True, returns results in JSON format.

    Returns:
        if as_dict is True: dict with bounding boxes and landmarks (if requested).
        if as_json is True: JSON string with bounding boxes and landmarks (if requested).
        else: DetectedFeatures object containing bounding boxes and landmarks (if requested).
    Raises:
        ValueError: If both as_dict and as_json are set to True.
    """
    if as_dict and as_json:
        raise ValueError("Cannot set both as_dict and as_json to True.")

    if with_output and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize MTCNN for face detection
    mtcnn = MTCNN(keep_all=True)

    # Load the image
    img = Image.open(image_path)

    # Detect faces and get bounding box coordinates
    detected_image_features = mtcnn.detect(img, landmarks=with_landmarks)

    boxes = detected_image_features[0]
    if boxes is None:
        print("No faces detected in the image.")

        result = DetectedFeatures(boxes=None, landmarks=None)

        if as_dict:
            return result.model_dump()
        if as_json:
            return result.model_dump_json()

        return result

    landmarks = None
    can_extract_landmarks = all([
        with_landmarks,
        len(detected_image_features) > 1,
        detected_image_features[2] is not None
    ])

    if can_extract_landmarks:
        landmarks = detected_image_features[2]

    if with_output:
        print(f"Detected {len(boxes)} faces.")

        for i, box in enumerate(boxes):
            # The bounding box format is [x_min, y_min, x_max, y_max]
            # Use the box coordinates to crop the original image
            cropped_face = img.crop(box.tolist())

            # Create a unique filename for each cropped face
            base_name = os.path.basename(image_path)
            name, ext = os.path.splitext(base_name)
            output_file = os.path.join(output_dir, f"{name}_face_{i}{ext}")

            # Save the cropped face image
            cropped_face.save(output_file)
            print(f"Cropped face {i} saved to {output_file}")

    result = DetectedFeatures(boxes=boxes, landmarks=landmarks)
    if as_dict:
        return result.model_dump()
    if as_json:
        return result.model_dump_json()

    return result
