import cv2
import numpy as np
import os
from pathlib import Path

def create_artificial_anomalies(input_directory, output_directory):
    """
    Creates artificial anomalies by modifying images in the input directory and saves them to the output directory.

    Args:
        input_directory (str): Path to the directory containing original images.
        output_directory (str): Path to save the images with artificial anomalies.
    """
    os.makedirs(output_directory, exist_ok=True)

    for filename in os.listdir(input_directory):
        if not filename.endswith(".jpg"):
            continue

        img_path = os.path.join(input_directory, filename)
        img = cv2.imread(img_path)

        # Anomaly: Add a white line
        img_with_line = img.copy()
        height, width, _ = img.shape
        start_point = (int(width * 0.2), int(height * 0.5))
        end_point = (int(width * 0.8), int(height * 0.5))
        cv2.line(img_with_line, start_point, end_point, (255, 255, 255), thickness=3)

        # Anomaly: Add random noise
        img_with_noise = img.copy()
        noise = np.random.randint(0, 50, (height, width, 3), dtype='uint8')
        img_with_noise = cv2.add(img_with_noise, noise)

        # Save modified images
        anomaly_line_path = os.path.join(output_directory, f"anomaly_line_{filename}")
        anomaly_noise_path = os.path.join(output_directory, f"anomaly_noise_{filename}")
        cv2.imwrite(anomaly_line_path, img_with_line)
        cv2.imwrite(anomaly_noise_path, img_with_noise)

        print(f"Anomalies created for {filename} and saved in {output_directory}")