from table import Table
from bin import Bin
import numpy as np

class Bins:
    def __init__(self, bin_width, bin_height, entry_offset, table:Table):
        self.bin_width = bin_width
        self.bin_height = bin_height
        self.entry_offset = entry_offset
        self.table = table
        self.bins = []

    def initialize_bins(self, num_bins=4):
        # Initialize bins list
        
        # Create bins for the top row
        top_bin_x = self.table.top_left[0] + self.bin_width
        bottom_bin_x = self.table.bottom_right[0] - self.bin_width
        top_bin_y = self.table.top_left[1] - self.bin_height // 2
        bottom_bin_y = self.table.bottom_right[1] + self.bin_height // 2

        positions = np.array([
            [top_bin_x, top_bin_y],
            [bottom_bin_x, top_bin_y],
            [top_bin_x, bottom_bin_y],
            [bottom_bin_x, bottom_bin_y],
        ]).astype(np.float32)
        
        entry_vector_top = np.array([0, 1], dtype=np.float32)    
        entry_vector_bottom = np.array([0, -1], dtype=np.float32)
        # Create bins for the bottom row
        for i, pos in enumerate(positions):
            if i < 2:
                self.bins.append(Bin(pos, self.bin_width, self.bin_height, self.entry_offset, entry_vector_top))
            else:
                self.bins.append(Bin(pos, self.bin_width, self.bin_height, self.entry_offset, entry_vector_bottom))
        
    def get_bins(self):
        """Return all bins."""
        return self.bins
    def reset(self):
        """Reset bins (static in this implementation)."""
        pass
