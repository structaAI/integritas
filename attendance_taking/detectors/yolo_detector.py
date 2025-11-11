from ultralytics import YOLO
import numpy as np
from models.classroom import Detection

class YOLODetector:
  def __init__(self, model_path: str, confidence_threshold: float = 0.5):
    self.model = YOLO(model_path)
    self.confidence = confidence_threshold
  
  def detect(self, image_path: str):
    """Detect persons and tables in image"""
    results = self.model(image_path)
    persons = []
    tables = []
    
    for result in results:
      boxes = result.boxes
      if boxes is not None:
        for box in boxes:
          conf = box.conf[0].cpu().numpy()
          if conf < self.confidence:
            continue
          
          class_id = int(box.cls[0])
          class_name = self.model.names[class_id]
          bbox = box.xyxy[0].cpu().numpy()
          
          detection = Detection(
            bbox=bbox,
            confidence=conf,
            class_name=class_name,
            center=self._get_center(bbox)
          )
          
          if class_name == 'person':
            persons.append(detection)
          elif class_name == 'table':
            tables.append(detection)
  
    return persons, tables
  
  def _get_center(self, bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)