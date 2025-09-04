import os
import cv2

from PIL import Image

from src.config import load_config

from src.preprocessing.histograms import compute_histograms_for_window
from src.schemas.enums.config_paths import ConfigName
from src.preprocessing.detection import detect_face_features
from src.preprocessing.normalization import align_and_square_face


def main():
    """Main function to detect, align, and analyze faces from an image."""
    input_file = './data/test-dataset/girl-in-sunlight.jpg'
    output_dir = './results/histograms'

    # Configuration object
    cfg = load_config(ConfigName.DEFAULT)

    # Step 1: Use the provided function to detect faces and get landmarks
    # We set with_output=False because we handle our own saving of the final, aligned crops.
    detected_features = detect_face_features(
        image_path=input_file,
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
    original_img = Image.open(input_file).convert('RGB')

    # Step 2: Iterate through each detected face
    for i, (box, landmarks) in enumerate(zip(
        detected_features['boxes'],
        detected_features['landmarks']
    )):
        face_index = i + 1

        # Step 3: Align and square the face using the new function
        aligned_face = align_and_square_face(
            image=original_img,
            face_box=box,
            landmarks=landmarks,
            cfg=cfg
        )

        # Step 4: Save the aligned face image
        os.makedirs(output_dir, exist_ok=True)
        image_filename = os.path.join(
            output_dir, f"aligned_face_{face_index}.jpg")
        cv2.imwrite(image_filename, aligned_face)

        # Step 5: Compute histograms
        compute_histograms_for_window(
            image=aligned_face,
            cfg=cfg,
        )


if __name__ == "__main__":
    main()
