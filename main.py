from pathlib import Path
from src.preprocessing.extract_frames import extract_frames_from_video
from src.detection.detect_objects import detect_print_areas
from src.classification.label_images import label_images
from src.classification.classify_images import train_and_classify_images
from ultralytics import YOLO


def main():
    print("Starting pipeline...")

    # Configuración de rutas
    video_path = Path("video.mp4")  # Ruta del video
    frames_directory = Path("data/frames")  # Carpeta donde se guardan los frames
    detection_output_directory = Path("data/detections")  # Carpeta de detecciones
    correct_images_directory = Path("data/correct")  # Carpeta de imágenes correctas
    incorrect_images_directory = Path("data/incorrect")  # Carpeta de imágenes incorrectas
    model_path = Path("models/yolov8s.pt")  # Ruta del modelo YOLO
    classifier_model_path = Path("models/classifier.h5")  # Ruta del modelo de clasificación

    # Paso 1: Extraer frames
    print("Step 1: Extracting frames...")
    extracted_files = extract_frames_from_video(
        video_path=str(video_path),
        output_directory=str(frames_directory),
        frame_interval_seconds=2
    )
    print(f"Frames extracted: {len(extracted_files)}")

    # Paso 2: Detectar objetos
    print("Step 2: Detecting objects...")
    yolo_model = YOLO(model_path)  # Cargar el modelo YOLO
    detected_files = detect_print_areas(
        model_path=str(model_path),
        input_directory=str(frames_directory),
        output_directory=str(detection_output_directory)
    )
    print(f"Objects detected in {len(detected_files)} frames.")

    # Paso 3: Etiquetar imágenes
    print("Step 3: Labeling images...")
    label_images(
        yolo_model=yolo_model,
        frames_directory=frames_directory,
        output_directory=detection_output_directory
    )
    print("Images labeled and saved.")

    # Paso 4: Entrenar y clasificar imágenes (opcional)
    print("Step 4: Training and classifying images...")
    classifier_path = train_and_classify_images(
        data_directory="data",
        model_save_path=str(classifier_model_path)
    )
    print(f"Classifier trained and saved at {classifier_path}")

    print("Pipeline completed!")


if __name__ == "__main__":
    main()
