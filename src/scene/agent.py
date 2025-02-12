from typing import List
import numpy as np

class Agent:
    def __init__(self, id: int = None, radius: int = 25, step_size: int = 4) -> None:
        self.id: int = id
        self.radius: int = radius
        self.step_size: int = step_size
        self.position: np.ndarray = np.zeros(2, dtype=np.float32)
        self.new_position: np.ndarray = np.zeros(2, dtype=np.float32)
        self.velocity: np.ndarray = np.zeros(2, dtype=np.float32)
        self.path: List[np.ndarray] = [] # Stores historical positions
        self.is_done: bool = False
        self.terminated: bool = False  
        
    def randomize_position(self, table_width: int, table_height: int, top_left: np.ndarray) -> None:
        """Randomize the agent's position within the table."""
        self.position = np.random.rand(2) * [table_width - 2 * self.radius, table_height - 2 * self.radius]
        self.position += [top_left[0] + self.radius, top_left[1] + self.radius]
        self.path.append(self.position)  

    def get_boundaries(self) -> np.ndarray:
        """Get the bounding box of the agent."""
        half_size = self.radius
        return np.array([self.position - half_size, self.position + half_size])
    
    def apply_action(self, action: np.ndarray) -> None:
        """Update the agent's position based on the action."""
        angle_rad = action[0] * np.pi  # Map action to angle
        move = np.array([np.cos(angle_rad), np.sin(angle_rad)]) * self.step_size
        blending_factor = 0.88

        # Update velocity with blending
        self.velocity = blending_factor * self.velocity + (1 - blending_factor) * move
        velocity_magnitude = np.linalg.norm(self.velocity)
        if velocity_magnitude > 0:
            self.velocity = (self.velocity / velocity_magnitude) * self.step_size

        # Update new position
        self.new_position = self.position + self.velocity
        self.path.append(self.new_position)

    def update_position(self) -> None:
        """Commit the new position."""
        self.position = self.new_position