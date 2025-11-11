import cv2
import numpy as np
from config import Config

def draw_detections(image, persons, tables):
  """Draw bounding boxes for persons and tables"""
  for person in persons:
    image = draw_bounding_box(image, person, Config.COLORS['person'], 'Person')
  
  for table in tables:
    image = draw_bounding_box(image, table, Config.COLORS['table'], 'Table')
  
  return image

def draw_bounding_box(image, detection, color, label):
  """Draw a single bounding box with label"""
  x1, y1, x2, y2 = detection.bbox.astype(int)
  
  # Draw rectangle
  cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
  
  # Draw label
  label_text = f"{label} {detection.confidence:.2f}"
  cv2.putText(image, label_text, (x1, y1-10), 
  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
  
  return image

def draw_attendance_status(image, classroom):
  """Draw attendance status on tables"""
  if classroom.table_grid is None:
      return image
  
  for row in range(classroom.rows):
    for col in range(classroom.cols):
      if classroom.table_grid[row, col] is not None:
        center_x, center_y = classroom.table_grid[row, col].center
        status = "Present" if classroom.attendance[row, col] == 1 else "Absent"
        color = Config.COLORS['present'] if classroom.attendance[row, col] == 1 else Config.COLORS['absent']
        
        text = f"({row},{col}): {status}"
        cv2.putText(image, text, (int(center_x)-50, int(center_y)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
  
  return image