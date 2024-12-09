from test_labels.labels import ground_truth
from sklearn.metrics import precision_score, recall_score, f1_score
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
    autoencoder = tf.keras.models.load_model(model_path, compile=False)

    os.makedirs(output_directory, exist_ok=True)

    predictions = {}

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
            predictions[filename] = 1  # Anomaly
        else:
            predictions[filename] = 0  # Correct

    calculate_metrics(ground_truth, predictions)


def calculate_metrics(ground_truth: dict, predictions: dict):
    y_true = list(ground_truth.values())
    y_pred = list(predictions.values())

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")
