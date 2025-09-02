from deepfake_recognition.helpers import detect_and_save_cropped_faces


def main():
    input_file = './input/girl-in-sunlight.jpg'
    output = detect_and_save_cropped_faces(
        input_file,
        with_output=True,
        with_landmarks=True,
        as_dict=True,
        as_json=False
    )
    print(output)


if __name__ == "__main__":
    main()
