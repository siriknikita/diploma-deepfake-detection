from typing import Any, Literal, overload

from facenet_pytorch import MTCNN
from PIL import Image

from src.schemas.features import DetectedFeatures


@overload
def detect_face_features(
    image_path: str,
    with_landmarks: bool = False,
    *,
    as_dict: Literal[False],
    as_json: Literal[True],
) -> str: ...


@overload
def detect_face_features(
    image_path: str,
    with_landmarks: bool = False,
    *,
    as_dict: Literal[True],
    as_json: Literal[False],
) -> dict[str, Any]: ...


@overload
def detect_face_features(
    image_path: str,
    with_landmarks: bool = False,
    *,
    as_dict: Literal[False] = False,
    as_json: Literal[False] = False,
) -> DetectedFeatures: ...


def detect_face_features(
    image_path: str,
    with_landmarks: bool = False,
    as_dict: bool = True,
    as_json: bool = False,
) -> dict[str, Any] | str | DetectedFeatures:
    """
    Detects faces in an image, optionally extracting facial landmarks.

    Args:
        image_path (str): The path to the input image file.
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

    # Initialize MTCNN for face detection
    mtcnn = MTCNN(keep_all=True)

    # Load the image
    img = Image.open(image_path)
    boxes, landmarks = None, None

    # Detect faces and get bounding box coordinates
    detected_image_features = mtcnn.detect(img, landmarks=with_landmarks)

    can_extract_boxes = all(
        [
            len(detected_image_features) >= 1,
            detected_image_features[0] is not None,
        ]
    )

    if can_extract_boxes:
        boxes = detected_image_features[0]

    can_extract_landmarks = all(
        [
            len(detected_image_features) > 1,
            detected_image_features[2] is not None,
        ]
    )

    if with_landmarks and can_extract_landmarks:
        landmarks = detected_image_features[2]

    result = DetectedFeatures(boxes=boxes, landmarks=landmarks)
    if as_dict:
        return result.model_dump()
    if as_json:
        return result.model_dump_json()

    return result
