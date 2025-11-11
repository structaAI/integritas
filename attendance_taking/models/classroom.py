import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Dict

@dataclass
class Detection:
  bbox: np.ndarray  # [x1, y1, x2, y2]
  confidence: float
  class_name: str
  center: Tuple[float, float]

class Classroom:
  def __init__(self, rows: int, cols: int):
    self.rows = rows
    self.cols = cols
    self.attendance = np.zeros((rows, cols), dtype=int)
    self.table_grid = None  # type: np.ndarray | None
    self.assignments = {}
  
  def update_attendance(self, assignments: Dict[Tuple[int, int], Detection]):
    """Update attendance matrix based on assignments"""
    self.attendance = np.zeros((self.rows, self.cols), dtype=int)
    for (row, col), person in assignments.items():
      if 0 <= row < self.rows and 0 <= col < self.cols:
        self.attendance[row, col] = 1
    self.assignments = assignments
  
  def get_report(self) -> dict:
    """Generate attendance report"""
    total = self.rows * self.cols
    present = np.sum(self.attendance)
    
    return {
      'matrix': self.attendance.tolist(),
      'total_seats': total,
      'present': present,
      'absent': total - present,
      'percentage': (present / total) * 100
    }