import dlib
import cv2

# Load the image
image_path = "girl-in-sunlight.jpg"
image = cv2.imread(image_path)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Load the dlib deep learning face detector
cnn_face_detector = dlib.cnn_face_detection_model_v1(
    "mmod_human_face_detector.dat")

# Run the detector
faces = cnn_face_detector(rgb_image, 1)

print("Found {} face(s) in the image.".format(len(faces)))

# Iterate over the detected faces
for i, face in enumerate(faces):
    x1 = face.rect.left()
    y1 = face.rect.top()
    x2 = face.rect.right()
    y2 = face.rect.bottom()

    print("Face {}: Left:{} Top:{} Right:{} Bottom:{}".format(i+1, x1, y1, x2, y2))

    # Draw a rectangle around the face
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Display the image with the detected face(s)
cv2.imshow("Detected Faces", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
