from ultralytics import YOLO

def train_yolo(data_yaml: str, model_output_path: str, epochs: int = 50):
    """
    Trains YOLOv8 for detecting the print area.
    
    Args: 
        data_yaml (str): Path to de YAMLfile specifying training and validation data.
        model_output_path (str): Path to save the trained YOLO model.
        epochs (int): Number of training epochs.
    """
    model = YOLO("models/yolov8s.pt")
    model.train(data=data_yaml, epochs=epochs, imgsz=640)
    model.save(model_output_path)
    print(f"Model trained and saved at {model_output_path}")