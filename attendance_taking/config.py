import os

class Config:
  # Classroom settings
  CLASSROOM_ROWS = 11
  CLASSROOM_COLS = 3
  
  # Model settings
  MODEL_PATH = "yolov8n.pt"  # or your custom trained model
  CONFIDENCE_THRESHOLD = 0.5
  
  # Assignment settings
  MAX_DISTANCE_RATIO = 0.4
  FALLBACK_DISTANCE = 200
  
  # Paths
  INPUT_DIR = "data/input"
  OUTPUT_DIR = "data/output"
  
  # Visualization
  COLORS = {
    'person': (0, 255, 0),      # Green
    'table': (255, 0, 0),       # Blue  
    'present': (0, 255, 0),     # Green
    'absent': (0, 0, 255)       # Red
  }