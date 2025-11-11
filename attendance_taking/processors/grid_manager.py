import numpy as np
from models.classroom import Detection

class GridManager:
  def __init__(self, rows: int, cols: int):
    self.rows = rows
    self.cols = cols
  
  def create_table_grid(self, tables: list) -> np.ndarray:
    """Arrange tables in grid based on positions"""
    if not tables:
      return None
      
      # Sort by y position (rows), then x position (columns)
    tables_sorted = sorted(tables, key=lambda t: (t.center[1], t.center[0]))
    
    try:
      grid = np.array(tables_sorted).reshape(self.rows, self.cols)
      return grid
    except:
      # Handle case where table count doesn't match grid size
      return self._create_partial_grid(tables_sorted)
  
  def _create_partial_grid(self, tables):
    """Create grid with available tables"""
    grid = np.full((self.rows, self.cols), None, dtype=object)
    
    for idx, table in enumerate(tables):
      if idx < self.rows * self.cols:
        row = idx // self.cols
        col = idx % self.cols
        grid[row, col] = table
    
    return grid