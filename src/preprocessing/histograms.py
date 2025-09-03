import cv2
import csv
import os


def compute_and_save_histograms(image, face_index, window_size, output_dir="histograms"):
    """
    Computes histograms for a sliding window over the image and saves the data to a CSV.

    Args:
        image (numpy.ndarray): The input image (aligned face).
        face_index (int): Index of the face for naming the output file.
        window_size (int): Size of the sliding window.
        output_dir (str): Directory to save the output files.

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_filename = os.path.join(
        output_dir, f"face_{face_index}_histograms.csv")
    h, w, _ = image.shape
    header = ['window_x', 'window_y'] + [
        f'bin_{i}_{channel}' for channel in ['r', 'g', 'b'] for i in range(256)
    ]

    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        for y in range(0, h - window_size + 1, window_size):
            for x in range(0, w - window_size + 1, window_size):
                window = image[y:y+window_size, x:x+window_size]
                b, g, r = cv2.split(window)

                hist_r = cv2.calcHist(
                    [r], [0], None, [256], [0, 256]).flatten()
                hist_g = cv2.calcHist(
                    [g], [0], None, [256], [0, 256]).flatten()
                hist_b = cv2.calcHist(
                    [b], [0], None, [256], [0, 256]).flatten()

                row = [x, y] + hist_r.tolist() + hist_g.tolist() + \
                    hist_b.tolist()
                writer.writerow(row)
