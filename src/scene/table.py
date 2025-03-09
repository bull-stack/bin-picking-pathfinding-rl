import numpy as np

class Table:
    """
    Represents the table in the simulation.
    The table defines the boundaries for agents and bins.
    """
    def __init__(self, screen_width: int, screen_height: int, table_width: int, table_height: int) -> None:
        self.screen_width: int = screen_width
        self.screen_height: int = screen_height
        self.table_width: int = table_width
        self.table_height: int = table_height
        
        self.top_left: np.ndarray = np.array([(screen_width - table_width) // 2, (screen_height - table_height) // 2])
        self.bottom_right: np.ndarray = self.top_left + np.array([table_width, table_height])

    def contains(self, position: np.ndarray) -> bool:
        """Check if a position is within the table boundaries."""
        return np.all(self.top_left <= position) and np.all(position <= self.bottom_right)
