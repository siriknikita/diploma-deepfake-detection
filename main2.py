from facenet_pytorch import MTCNN
from PIL import Image
import os


def detect_and_save_cropped_faces(
    image_path,
    output_dir='./output/',
    with_output=False,
    with_landmarks=False,
):
    """
    Detects faces in an image, crops each face, and saves them to a directory.

    Args:
        image_path (str): The path to the input image file.
        output_dir (str): The directory to save the cropped faces.
        with_output (bool): If True, saves the cropped faces; if False, only returns bounding boxes.
        with_landmarks (bool): If True, returns facial landmarks along with bounding boxes.
    """
    # Initialize MTCNN for face detection
    mtcnn = MTCNN(keep_all=True)

    # Load the image
    img = Image.open(image_path)

    # Detect faces and get bounding box coordinates
    detected_image_features = mtcnn.detect(img, landmarks=with_landmarks)

    boxes = detected_image_features[0]
    if boxes is None:
        print("No faces detected in the image.")
        return [None, None]

    landmarks = detected_image_features[2] if with_landmarks else None

    if with_output:
        print(f"Detected {len(boxes)} faces.")

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

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

    return [boxes, landmarks]


def main():
    input_file = './input/girl-in-sunlight.jpg'
    [detected_faces, landmarks] = detect_and_save_cropped_faces(
        input_file,
        with_output=True,
        with_landmarks=False
    )
    print("Detected face bounding boxes:", detected_faces)
    print("Detected landmarks:", landmarks)


if __name__ == "__main__":
    main()
