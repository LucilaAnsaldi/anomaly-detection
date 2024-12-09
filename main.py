from src.detection.detect_objects import detect_print_area
from src.classification.detect_anomalies import detect_and_visualize_anomalies
from src.classification.crop_detected_area import crop_detected_area
import os

def main(yolo_model_path, autoencoder_model_path, test_directory, output_directory, input_shape, anomaly_threshold=0.05):

    detected_directory = os.path.join(output_directory, "detected")
    detect_print_area(
        model_path=yolo_model_path,
        frames_directory=test_directory,
        output_directory=detected_directory
    )

    cropped_directory = os.path.join(output_directory, "cropped")
    crop_detected_area(
        model_path=yolo_model_path,
        frames_directory=test_directory,
        output_directory=cropped_directory
    )

    results_directory = os.path.join(output_directory, "results")
    detect_and_visualize_anomalies(
        model_path=autoencoder_model_path,
        test_directory=cropped_directory,
        output_directory=results_directory,
        input_shape=input_shape,
        threshold=anomaly_threshold,
    )

if __name__ == "__main__":
    main(
    yolo_model_path="models/print_area_detector.pt",
    autoencoder_model_path="models/autoencoder.h5",
    test_directory="data/test_images",
    output_directory="data/test_results",
    input_shape=(128, 128, 3),
    anomaly_threshold=0.007,
)
