from PIL import Image

from src.config import load_config

from src.preprocessing.histograms import compute_histograms_for_window
from src.schemas.enums.config_paths import ConfigName
from src.preprocessing.detection import detect_face_features
from src.preprocessing.normalization import normalize_face


def main():
    """Main function to detect, align, and analyze faces from an image."""
    # Configuration object
    cfg = load_config(ConfigName.DEFAULT)

    input_file_path = cfg.input_file_path

    # Step 1: Use the provided function to detect faces and get landmarks
    detected_features = detect_face_features(
        image_path=input_file_path,
        with_landmarks=True,
        as_dict=True,
        as_json=False
    )

    # Check if any faces were detected
    no_faces_detected = any([
        detected_features['boxes'] is None,
        len(detected_features['boxes']) == 0
    ])
    if no_faces_detected:
        print("No faces detected.")
        return

    # Load the original image once
    original_img = Image.open(input_file_path).convert('RGB')

    # Step 2: Iterate through each detected face
    for _, (box, landmarks) in enumerate(zip(
        detected_features['boxes'],
        detected_features['landmarks']
    )):
        # Step 3: Align and square the face
        normalized_face = normalize_face(
            image=original_img,
            face_box=box,
            landmarks=landmarks,
            cfg=cfg
        )

        # Step 4: Compute histograms
        compute_histograms_for_window(
            image=normalized_face,
            cfg=cfg,
        )


if __name__ == "__main__":
    main()
