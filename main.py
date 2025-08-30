import cv2
import dlib
import numpy as np

# Parameters
WINDOW_SIZE = 9
PADDING = 0.25  # 25% extra around face


def round_up_to_multiple(x, base):
    return int(np.ceil(x / base) * base)


# Load image
image_path = "girl-in-sunlight.jpg"
image = cv2.imread(image_path)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Load models
cnn_face_detector = dlib.cnn_face_detection_model_v1(
    "mmod_human_face_detector.dat")
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Detect faces
faces = cnn_face_detector(rgb_image, 1)
print("Found {} face(s)".format(len(faces)))

for i, face in enumerate(faces):
    # Original rect
    x1 = face.rect.left()
    y1 = face.rect.top()
    x2 = face.rect.right()
    y2 = face.rect.bottom()
    w, h = x2 - x1, y2 - y1

    # Center of face
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    # Step 1: calculate required square size with padding
    size = max(w, h)
    size = int(size * (1 + PADDING))

    # Step 2: make sure size is a multiple of WINDOW_SIZE
    size = round_up_to_multiple(size, WINDOW_SIZE)

    # Step 3: landmarks for alignment
    shape = predictor(rgb_image, face.rect)
    left_eye = (shape.part(36).x, shape.part(36).y)
    right_eye = (shape.part(45).x, shape.part(45).y)

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))

    # Step 4: rotate the entire image
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    # Step 5: get new bounding box coordinates after rotation
    # Note: cx, cy are the center in the original image.
    # The center of the rotated face will still be at the same pixel coordinates (cx, cy)
    # in the rotated image because we rotated around that point.
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
        pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0)  # black padding
    )

    # Step 8: crop the image
    # The new crop coordinates are relative to the padded image.
    # We must add the padding amounts to our old coordinates to get the new ones.
    crop_x1 = new_x1 + pad_left
    crop_y1 = new_y1 + pad_top
    crop_x2 = new_x2 + pad_left
    crop_y2 = new_y2 + pad_top

    final_crop = padded[crop_y1:crop_y2, crop_x1:crop_x2]

    print(f"Face {i+1} final crop shape: {final_crop.shape}")

    # Show result
    cv2.imshow(f"Aligned Square Face {i+1}", final_crop)
    cv2.waitKey(0)

cv2.destroyAllWindows()
