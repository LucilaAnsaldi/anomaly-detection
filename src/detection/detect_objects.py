from typing import List
from pathlib import Path
import cv2
from ultralytics import YOLO

def detect_print_area(
    model_path: str, frames_directory: str, output_directory: str
) -> List[str]:
    """
    Detects the largest area of interest in frames using a trained YOLOv8 model.

    Args:
        model_path (str): Path to YOLO model.
        frames_directory (str): Directory containing frames.
        output_directory (str): Directory to save detected print areas.

    Returns:
        List[str]: List of paths to frames with detected areas highlighted.
    """
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    processed_frames = []
    model = YOLO(model_path)

    for frame_path in Path(frames_directory).glob("*.jpg"):
        frame = cv2.imread(str(frame_path))
        results = model.predict(source=str(frame_path), save=False)

        largest_box = None
        largest_area = 0

        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4].tolist())
            area = (x2 - x1) * (y2 - y1)
            if area > largest_area:
                largest_area = area
                largest_box = (x1, y1, x2, y2)

        if largest_box:
            x1, y1, x2, y2 = largest_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue box for detection

        output_path = Path(output_directory) / frame_path.name
        cv2.imwrite(str(output_path), frame)
        processed_frames.append(str(output_path))

    return processed_frames
