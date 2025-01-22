import numpy as np
import pygame
class Bin:
    def __init__(self, position, width, height, entry_offset, entry_vector):
        self.position = np.array(position, dtype=np.float32)
        self.width = width
        self.height = height
        self.entry_offset = entry_offset
        self.in_use = False
        self.entry_vector = entry_vector

    # def get_boundaries(self):
    #     left = self.position[0] - self.width / 2
    #     right = self.position[0] + self.width / 2
    #     top = self.position[1] - self.height / 2
    #     bottom = self.position[1] + self.height / 2
    #     return left, right, top, bottom
    
    def get_boundaries(self):
        bin_rect = pygame.Rect(
            self.position[0] - self.width // 2,
            self.position[1] - self.height // 2,
            self.width,
            self.height,
        )
        return bin_rect

    def get_bin_entry_zone(self, table):
        entry_length = 50

        # Define the rectangle extending in the entry direction
        if np.array_equal(self.entry_vector, [0, 1]):  # Top entry
            rect = pygame.Rect(
                self.position[0] - self.width // 2,
                table.top_left[1] - entry_length,
                self.width,
                entry_length,
            )
        elif np.array_equal(self.entry_vector, [0, -1]):  # Bottom entry
            rect = pygame.Rect(
                self.position[0] - self.width // 2,
                table.bottom_right[1],
                self.width,
                entry_length,
            )
        else:
            raise ValueError("Invalid entry vector for goal.")
        return rect

    
        
    def __eq__(self, other):
        if isinstance(other, Bin):
            return np.array_equal(self.position, other.position)
        return False