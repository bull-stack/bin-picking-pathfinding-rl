from typing import Union

import threading
import numpy as np

from table import Table
from agent import Agent

class Bin:
    def __init__(self, position: Union[list, tuple, np.ndarray], width: int, height: int, entry_offset: int, entry_vector: np.ndarray) -> None:
        self.position: np.ndarray = np.array(position, dtype=np.float32)
        self.width: int = width
        self.height: int = height
        self.entry_offset: int = entry_offset
        self.entry_vector: np.ndarray = np.array(entry_vector, dtype=np.float32)
        self.available: bool = True
        self.lock: threading.Lock = threading.Lock()
        self.reactivation_time: threading.Timer = None
    def reset(self):
        """Reset the bin's state."""
        self.available = True
        self.reactivation_time = 0
    def get_boundaries(self) -> np.ndarray:
        """
        Return the boundaries of the bin as a bounding box (AABB).
        """
        half_size = np.array([self.width / 2, self.height / 2])
        min_point = self.position - half_size  # Min point of the bounding box
        max_point = self.position + half_size  # Max point of the bounding box
        return np.array([min_point, max_point])
    
    def get_bin_entry_zone(self, table: Table) -> tuple:
        """
        Get the entry zone of the bin as a tuple of coordinates and dimensions.
        """
        entry_length = 90

        if np.array_equal(self.entry_vector, [0, 1]):  # Top entry
            entry_zone = (
                self.position[0] - self.width // 2,  # x position
                table.top_left[1] - (entry_length // 2) + 10,  # y position (top entry)
                self.width,
                entry_length # Extend the entry into the table
            )
        elif np.array_equal(self.entry_vector, [0, -1]):  # Bottom entry
            entry_zone = (
                self.position[0] - self.width // 2,  # x position
                table.bottom_right[1] - (entry_length // 2) - 10,  # y position (bottom entry)
                self.width,
                entry_length # Extend the entry into the table
            )
        else:
            raise ValueError("Invalid entry vector for the bin.")
        
        return entry_zone

            
    def temporarily_disable(self, delay: float, agent: Agent):
        """Disable the bin for a set duration."""
        self.available = False
        self.reactivation_time = threading.Timer(delay, self.reactivate, args=(agent, ))
        self.reactivation_time.start()

    def reactivate(self, agent: Agent):
        """Reactivate the bin after the delay."""
        with self.lock:
            self.available = True
            self.reactivation_time = None 
            agent.terminated = True
                    
    def __eq__(self, other: object) -> bool:
        """
        Check for equality between two Bin instances based on their position.
        """
        if isinstance(other, Bin):
            return np.array_equal(self.position, other.position)
        return False