import numpy as np
from models.classroom import Detection

class AssignmentProcessor:
  def __init__(self, max_distance_ratio: float = 0.4, fallback_distance: float = 200):
    self.max_distance_ratio = max_distance_ratio
    self.fallback_distance = fallback_distance
  
  def assign_persons_to_tables(self, persons: list, table_grid: np.ndarray, grid_shape: tuple):
    """Assign persons to tables based on proximity"""
    assignments = {}
    
    for person in persons:
      table_pos = self._find_best_table(person.center, table_grid, grid_shape)
      if table_pos:
        assignments[table_pos] = person
    
    return assignments
  
  def _find_best_table(self, person_center, table_grid, grid_shape):
    """Find the closest table to a person"""
    best_distance = float('inf')
    best_position = None
    
    for row in range(grid_shape[0]):
      for col in range(grid_shape[1]):
        if table_grid[row, col] is not None:
          table_center = table_grid[row, col].center
          distance = self._calculate_distance(person_center, table_center)
          
          max_allowed = self._get_max_allowed_distance(table_grid, grid_shape)
          if distance < best_distance and distance < max_allowed:
            best_distance = distance
            best_position = (row, col)
    
    return best_position
  
  def _calculate_distance(self, point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
  
  def _get_max_allowed_distance(self, table_grid, grid_shape):
    """Calculate maximum allowed distance for assignment"""
    centers = []
    for row in range(grid_shape[0]):
      for col in range(grid_shape[1]):
        if table_grid[row, col] is not None:
          centers.append(table_grid[row, col].center)
    
    if len(centers) > 1:
      distances = []
      for i in range(len(centers)):
          for j in range(i+1, len(centers)):
              distances.append(self._calculate_distance(centers[i], centers[j]))
      return np.mean(distances) * self.max_distance_ratio
    
    return self.fallback_distance