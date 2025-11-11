import json
import csv
import os
from datetime import datetime
import cv2

def save_attendance_report(classroom, output_dir="data/output"):
  """Save attendance report as JSON and CSV"""
  os.makedirs(output_dir, exist_ok=True)
  
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  
  # Save JSON
  report = classroom.get_report()
  json_path = os.path.join(output_dir, f"attendance_{timestamp}.json")
  with open(json_path, 'w') as f:
    json.dump(report, f, indent=2)
  
  # Save CSV
  csv_path = os.path.join(output_dir, f"attendance_{timestamp}.csv")
  with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Row', 'Column', 'Status'])
    
    for row in range(classroom.rows):
      for col in range(classroom.cols):
        status = "Present" if classroom.attendance[row, col] == 1 else "Absent"
        writer.writerow([row, col, status])
  
  return json_path, csv_path

def save_visualization(image, image_path, output_dir="data/output"):
  """Save visualization image"""
  os.makedirs(output_dir, exist_ok=True)
  
  filename = os.path.basename(image_path)
  output_path = os.path.join(output_dir, f"annotated_{filename}")
  
  cv2.imwrite(output_path, image)
  return output_path