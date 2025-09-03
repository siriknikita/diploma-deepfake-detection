import os
import cv2
from PIL import Image

from src.preprocessing.detection import detect_and_save_cropped_faces
from src.preprocessing.histograms import compute_and_save_histograms
from src.preprocessing.normalization import align_and_square_face

# Parameters
WINDOW_SIZE = 9
PADDING = 0.25


def main():
    """Main function to detect, align, and analyze faces from an image."""
    input_file = './data/test-dataset/girl-in-sunlight.jpg'
    output_dir = './results/histograms'

    # Step 1: Use the provided function to detect faces and get landmarks
    # We set with_output=False because we handle our own saving of the final, aligned crops.
    detected_features = detect_and_save_cropped_faces(
        image_path=input_file,
        with_output=False,
        with_landmarks=True,
        as_dict=True,
        as_json=False
    )

    # Check if any faces were detected
    if detected_features['boxes'] is None or len(detected_features['boxes']) == 0:
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
            window_size=WINDOW_SIZE,
            padding=PADDING
        )

        # Step 4: Save the aligned face image
        os.makedirs(output_dir, exist_ok=True)
        image_filename = os.path.join(
            output_dir, f"aligned_face_{face_index}.jpg")
        cv2.imwrite(image_filename, aligned_face)

        # Step 5: Compute and save histograms
        compute_and_save_histograms(
            image=aligned_face,
            face_index=face_index,
            window_size=WINDOW_SIZE,
            output_dir=output_dir
        )


if __name__ == "__main__":
    main()
