from ultralytics import YOLO
import cv2
from pathlib import Path
from typing import List

def crop_detected_areas_without_txt(
    model_path: str,
    frames_directory: str,
    output_directory: str
) -> List[str]:
    """
    Crops detected areas from frames based on YOLO detections without using .txt files.
    
    Args:
        model_path (str): Path to the YOLO model.
        frames_directory (str): Directory containing the original frames.
        output_directory (str): Directory where cropped areas will be saved.

    Returns:
        List[str]: Paths of cropped images.
    """
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    cropped_images = []
    model = YOLO(model_path)

    for frame_path in Path(frames_directory).glob("*.jpg"):
        frame = cv2.imread(str(frame_path))
        results = model.predict(source=str(frame_path), save=False)

        for i, box in enumerate(results[0].boxes.xyxy):
            x1, y1, x2, y2 = map(int, box[:4].tolist())
            cropped = frame[y1:y2, x1:x2]
            
            # Save the cropped image
            cropped_path = Path(output_directory) / f"{frame_path.stem}_crop_{i}.jpg"
            cv2.imwrite(str(cropped_path), cropped)
            cropped_images.append(str(cropped_path))

    return cropped_images
