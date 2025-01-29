from table import Table
from typing import List
import numpy as np
from bin import Bin
class Agent:
    def __init__(self, radius: int = 25, step_size: int = 4, Kp: float = 1.0, Ki: float = 0.0, Kd: float = 0.0) -> None:
        self.radius: int = radius
        self.velocity: np.ndarray = np.zeros(2, dtype=np.float32)
        self.position: np.ndarray = np.zeros(2, dtype=np.float32)
        self.new_position: np.ndarray = np.zeros(2, dtype=np.float32)
        self.path: List[np.ndarray] = []  # Add a path to store positions
        self.step_size = step_size
        self.in_entry_zone: bool = False
        self.is_done: bool = False
        self.is_gone: bool = False

    
    def randomize_position(self, table: Table) -> np.ndarray:
        """
        Reset the object's position to a random valid location within the table.
        """
        self.position = np.random.rand(2) * [
            table.table_width - 2 * self.radius,
            table.table_height - 2 * self.radius,
        ]
        self.position += [
            table.top_left[0] + self.radius,
            table.top_left[1] + self.radius,
        ]
        return self.position.astype(np.float32)

    def update(self, action: np.ndarray) -> np.ndarray:
        """
        Update the agent's position based on the given movement and step size.
        Apply PID control for correcting deviation from the path.
        """            
        angle_rad = action[0] * np.pi  # Map normalized action to [-π, π]
        move = np.array([np.cos(angle_rad), np.sin(angle_rad)]) * self.step_size
        blending_factor = 0.7
                # Update velocity
        # Compute adaptive blending factor based on movement magnitude

        # Update velocity with adaptive blending
        self.velocity = blending_factor * self.velocity + (1 - blending_factor) * move

        velocity_magnitude = np.linalg.norm(self.velocity)
        if velocity_magnitude > 0:
            self.velocity = (self.velocity / velocity_magnitude) * self.step_size

        # Update position
        self.new_position = self.position + self.velocity
        self.path.append(self.new_position)  # Store the new position in the path

    
    def update_position(self):
        """ 
        Update the agent's position to the new position.
        """
        self.position = self.new_position

    def get_boundaries(self) -> np.ndarray:
        """
        Compute the object's boundaries as a bounding box (AABB).
        The bounding box is defined by two points: min and max corners.
        """
        half_size = self.radius  # Assuming it's a sphere, you could use width/height for a box
        min_point = self.position - half_size  # The minimum point (left-bottom-back)
        max_point = self.position + half_size  # The maximum point (right-top-front)
        return np.array([min_point, max_point])  # Returning the corners of the bounding box
