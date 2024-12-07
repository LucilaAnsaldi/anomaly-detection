import cv2
import os
from typing import List

def extract_frames_from_video(video_path: str, output_directory: str, frame_interval_seconds: int = 1) -> List[str]:
    """
    Extracts frames from a video at regular intervals.
    Args:
        video_path (str): Path to the input video file.
        output_directory (str): Folder where frames will be saved.
        interval (int): Interval in seconds between extracted frames.
    Returns:
        List[str]: List of file paths of the extracted frames.
    """
    os.makedirs(output_directory, exist_ok=True)
    video_capture = cv2.VideoCapture(video_path)
    frame_rate = int(video_capture.get(cv2.CAP_PROP_FPS))
    frame_interval = frame_rate * frame_interval_seconds
    frame_count = 0
    extracted_files = []

    while video_capture.isOpened():
        success, frame = video_capture.read()
        if not success:
            break
        if frame_count % frame_interval == 0:
            frame_path = os.path.join(output_directory, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
            extracted_files.append(frame_path)
        frame_count += 1
    
    video_capture.release()
    return extracted_files