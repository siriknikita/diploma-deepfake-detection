from PIL import Image

from src.config import load_config

from src.preprocessing.histograms import compute_histograms_for_window, get_cnn_histogram_features
from src.schemas.enums.config_paths import ConfigName
from src.preprocessing.detection import detect_face_features
from src.preprocessing.normalization import normalize_face
import numpy as np


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
    for face_idx, (box, landmarks) in enumerate(zip(
        detected_features['boxes'],
        detected_features['landmarks']
    )):
        print(f"\nProcessing face {face_idx + 1}...")
        
        # Step 3: Align and square the face
        normalized_face = normalize_face(
            image=original_img,
            face_box=box,
            landmarks=landmarks,
            cfg=cfg
        )

        # Step 4: Compute histograms and get structured CNN features
        cnn_features = get_cnn_histogram_features(
            image=normalized_face,
            cfg=cfg,
        )
        
        # Display information about the concatenated matrix
        print(f"Face {face_idx + 1} CNN features:")
        print(f"  - Matrix shape: {cnn_features.feature_matrix.shape}")
        print(f"  - Number of windows: {cnn_features.num_windows}")
        print(f"  - Feature dimension per window: {cnn_features.feature_dimension} (256×3 RGB channels)")
        print(f"  - Window size used: {cnn_features.window_size}")
        
        # Also compute individual histograms for comparison
        individual_histograms = compute_histograms_for_window(
            image=normalized_face,
            cfg=cfg,
        )
        print(f"  - Number of individual histogram entries: {len(individual_histograms)}")
        
        # Show a sample of the concatenated features for the first window
        if cnn_features.feature_matrix.size > 0:
            first_window_features = cnn_features.feature_matrix[0, :]
            print(f"  - First window feature vector (first 10 values): {first_window_features[:10]}")
            print(f"  - First window feature vector (last 10 values): {first_window_features[-10:]}")
            
            # Verify the concatenation structure
            hist_r_sample = first_window_features[:256]
            hist_g_sample = first_window_features[256:512]
            hist_b_sample = first_window_features[512:768]
            print(f"  - Red channel histogram sum: {np.sum(hist_r_sample):.2f}")
            print(f"  - Green channel histogram sum: {np.sum(hist_g_sample):.2f}")
            print(f"  - Blue channel histogram sum: {np.sum(hist_b_sample):.2f}")


if __name__ == "__main__":
    main()
