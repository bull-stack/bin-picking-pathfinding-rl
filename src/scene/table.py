from typing import Tuple

class Table:
    def __init__(
        self, 
        screen_width: int, 
        screen_height: int, 
        table_width: int, 
        table_height: int
    ) -> None:
        """
        Initialize a Table object.

        Args:
            screen_width (int): The width of the screen.
            screen_height (int): The height of the screen.
            table_width (int): The width of the table.
            table_height (int): The height of the table.
        """
        self.screen_width: int = screen_width
        self.screen_height: int = screen_height
        self.table_width: int = table_width
        self.table_height: int = table_height

        self.top_left: Tuple[int, int] = (
            (self.screen_width - self.table_width) // 2,
            (self.screen_height - self.table_height) // 2,
        )
        self.bottom_right: Tuple[int, int] = (
            self.top_left[0] + self.table_width,
            self.top_left[1] + self.table_height,
        )

    def contains(self, position: Tuple[float, float]) -> bool:
        """
        Check if a given position is within the table boundaries.

        Args:
            position (Tuple[float, float]): The (x, y) position to check.

        Returns:
            bool: True if the position is within the table boundaries, False otherwise.
        """
        return (
            self.top_left[0] <= position[0] <= self.bottom_right[0] and
            self.top_left[1] <= position[1] <= self.bottom_right[1]
        )
