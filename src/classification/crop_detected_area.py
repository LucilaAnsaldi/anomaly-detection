from ultralytics import YOLO
import cv2
from pathlib import Path
from typing import List

def crop_detected_area(
    model_path: str, frames_directory: str, output_directory: str
) -> List[str]:
    """
    Crops the largest detected area from frames based on YOLO detections.

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

        # Find the largest bounding box
        largest_box = None
        largest_area = 0
        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4].tolist())
            area = (x2 - x1) * (y2 - y1)
            if area > largest_area:
                largest_area = area
                largest_box = (x1, y1, x2, y2)

        # Crop the largest detected area
        if largest_box:
            x1, y1, x2, y2 = largest_box
            cropped = frame[y1:y2, x1:x2]

            cropped_path = Path(output_directory) / f"{frame_path.stem}_largest_crop.jpg"
            cv2.imwrite(str(cropped_path), cropped)
            cropped_images.append(str(cropped_path))

    return cropped_images
