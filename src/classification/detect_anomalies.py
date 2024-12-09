import os
import cv2
import numpy as np
import tensorflow as tf
from typing import Tuple

def detect_and_visualize_anomalies(
    model_path: str,
    test_directory: str,
    output_directory: str,
    input_shape: Tuple[int, int, int] = (128, 128, 3),
    threshold: float = 0.05
):
    """
    Detect anomalies in images, separate correct and incorrect, and highlight only anomalous areas.

    Args:
        model_path (str): Path to the trained autoencoder model.
        test_directory (str): Directory containing test images.
        output_directory (str): Directory to save the results with anomalies highlighted.
        input_shape (Tuple[int, int, int]): Shape of the input for the autoencoder.
        threshold (float): Threshold for anomaly detection.

    Returns:
        None
    """
    autoencoder = tf.keras.models.load_model(model_path, compile=False)

    correct_directory = os.path.join(output_directory, "correct")
    anomalies_directory = os.path.join(output_directory, "anomalies")
    os.makedirs(correct_directory, exist_ok=True)
    os.makedirs(anomalies_directory, exist_ok=True)

    for filename in os.listdir(test_directory):
        if not filename.endswith(".jpg") and not filename.endswith(".jpeg"):
            continue

        image_path = os.path.join(test_directory, filename)
        img = cv2.imread(image_path)
        img_resized = cv2.resize(img, (input_shape[0], input_shape[1]))
        img_normalized = img_resized / 255.0

        img_input = np.expand_dims(img_normalized, axis=0)

        reconstructed_img = autoencoder.predict(img_input)[0]
        reconstruction_error = np.mean((img_normalized - reconstructed_img) ** 2)

        if reconstruction_error > threshold:
            diff = np.abs(img_normalized - reconstructed_img)
            diff_amplified = (diff * 255).astype(np.uint8)

            gray_diff = cv2.cvtColor(diff_amplified, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray_diff, 50, 255, cv2.THRESH_BINARY)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(img_resized, (x, y), (x + w, y + h), (0, 0, 255), 2)

            output_path = os.path.join(anomalies_directory, f"anomaly_{filename}")
        else:
            output_path = os.path.join(correct_directory, filename)

        cv2.imwrite(output_path, img_resized)

        print(f"Processed {filename} - Reconstruction Error: {reconstruction_error:.4f}")

    print(f"Anomaly detection completed. Results saved in {output_directory}.")
