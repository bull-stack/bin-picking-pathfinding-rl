import numpy as np


class Table:
    def __init__(self, screen_width, screen_height, table_width, table_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.table_width = table_width
        self.table_height = table_height

        self.top_left = (
            (self.screen_width - self.table_width) // 2,
            (self.screen_height - self.table_height) // 2,
        )
        self.bottom_right = (
            self.top_left[0] + self.table_width,
            self.top_left[1] + self.table_height,
        )

    def contains(self, position):
        return (
            self.top_left[0] <= position[0] <= self.bottom_right[0] and
            self.top_left[1] <= position[1] <= self.bottom_right[1]
        )