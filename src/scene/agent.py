from table import Table
from bins import Bins
from bin import Bin
import numpy as np
import pygame
class Agent:
    def __init__(self, radius:int, table:Table, bins:Bins):
        self.table = table
        self.bins = bins
        self.radius = radius
        self.velocity = np.zeros(2)
        self.position = np.zeros(2)
        self.prev_positions = np.zeros(2)
        self.target_bin: Bin = None
        self.distance = 0
        self.prev_distance = 0 
        
        
    def randomize_position(self):
        """
        Reset the object's position to a random valid location within the table.
        """
        self.position = np.random.rand(2) * [
            self.table.table_width - 2 * self.radius,
            self.table.table_height - 2 * self.radius,
        ]
        self.position += [
            self.table.top_left[0] + self.radius,
            self.table.top_left[1] + self.radius,
        ]
        return self.position.astype(np.float32)
    
    def update(self, move, step_size):

        # Smooth the movement
        blending_factor = 0.7
        self.velocity = blending_factor * self.velocity + (1 - blending_factor) * move
        velocity_magnitude = np.linalg.norm(self.velocity)

        # Normalize velocity for consistent speed
        if velocity_magnitude > 0:
            self.velocity = (self.velocity / velocity_magnitude) * step_size
        return self.position + self.velocity
    
    def update_position(self, new_pos):
        self.position  = new_pos
        
    def set_closest_bin(self):
        available_bins = [bin for bin in self.bins.get_bins() if not bin.in_use]
        if not available_bins:
            self.target_bin = None  
            return
        distances = [np.linalg.norm(bin.position - self.position) for bin in available_bins]
        self.target_bin = available_bins[np.argmin(distances)]
        
    def get_distance_to_bin(self):
        available_bins = [bin for bin in self.bins.get_bins() if not bin.in_use]
        if not available_bins:
            self.target_bin = None  
            return
        distances = [np.linalg.norm(bin.position - self.position) for bin in available_bins]
        closest_bin: Bin = available_bins[np.argmin(distances)]
        return np.linalg.norm(self.position - closest_bin.position)
        
    def get_target_bin(self):
        return self.target_bin
    
    # def get_distance_to_bin(self):
    #     return np.linalg.norm(self.position - self.target_bin.position)
    
    def get_boundaries(self, new_pos):
        # Compute object boundaries
        agent_rect = pygame.Rect(
            new_pos[0] - self.radius,
            new_pos[1] - self.radius,
            self.radius * 2,
            self.radius * 2,
        )
        return agent_rect
    
    def in_entry_zone(self, new_pos):
        bin_entry_zone = self.target_bin.get_bin_entry_zone(self.table)
        return bin_entry_zone.collidepoint(int(new_pos[0]), int(new_pos[1]))
    
    def in_target_bin(self):
        bin_rect = self.target_bin.get_boundaries()
        return bin_rect.contains(self.get_boundaries())
        
    def is_closer_to_bin(self, new_pos):
        self.prev_distance = np.linalg.norm(self.position - self.target_bin.position)
        self.distance = np.linalg.norm(new_pos - self.target_bin.position)
        return self.distance < self.prev_distance
    
    def get_distance_change(self):
        return self.distance - self.prev_distance
    
    def bin_direction(self):
        bin_dir = self.target_bin.position - self.position
        bin_distance = np.linalg.norm(bin_dir)
        return bin_dir / bin_distance if bin_distance > 0 else np.array([0, 0])   
    
    def __eq__(self, other):
        if isinstance(other, Agent):
            return np.array_equal(self.position, other.position) and np.array_equal(self.velocity, other.velocity)
        return False