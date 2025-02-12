from visualization_base import VisualizationBase
from simulator import Simulator

from typing import List, Dict
import pygame
import numpy as np

class RendererPyGame(VisualizationBase):
    def __init__(self, screen_width: int, screen_height: int, colors: Dict[str, tuple], title: str = "Table Pathfinding") -> None:
        self.screen_width: int = screen_width
        self.screen_height: int = screen_height
        self.title: str = title
        self.colors: Dict[str, tuple] = colors
        self.screen: pygame.Surface = None
        self.clock: pygame.time.Clock = None

    def initialize(self) -> None:
        """
        Initialize the Pygame window and clock.
        """
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption(self.title)
        self.clock = pygame.time.Clock()

    def draw_table(self, simulator: Simulator) -> None:
        """
        Draw the table on the screen.
        """
        pygame.draw.rect(
            self.screen,
            self.colors["table"],
            pygame.Rect(
                simulator.table.top_left[0], 
                simulator.table.top_left[1],
                simulator.table.table_width, 
                simulator.table.table_height
            ),
        )

    def draw_bins(self, simulator: Simulator) -> None:
        """
        Draw the bins and their entry zones on the screen.
        """
        for bin in simulator.bins:
            color = (
                self.colors["bin_available"] if bin.available == True 
                else self.colors["bin"] 
            )
            pygame.draw.rect(
                self.screen,
                color,
                pygame.Rect(
                    bin.position[0] - bin.width // 2,
                    bin.position[1] - bin.height // 2,
                    bin.width,
                    bin.height
                ),
            )
            # Draw goal entry zone
            bin_entry_zone = bin.get_bin_entry_zone(simulator.table)
            x, y, width, height = bin_entry_zone
            entry_zone_rect = pygame.Rect(
                x, y, width, height
            )
            pygame.draw.rect(self.screen, (100, 100, 255), entry_zone_rect, 2)

    def draw_object(self, simulator: Simulator) -> None:
        """
        Draw agents (as circles) on the screen.
        """
        for agent in simulator.agents:
            if agent.terminated:
                continue  # Skip drawing agents that have gone
            color = (
                self.colors["agent"] if agent == simulator.active_agent else
                self.colors["arrow"] 
            )
            pygame.draw.circle(
                self.screen, color, agent.position.astype(int), agent.radius
            )
            # Draw the agent's path as a line
            if len(agent.path) <= 1:
                continue  # Skip drawing agents with no path
            if agent.path:
                pygame.draw.lines(
                    self.screen,
                    self.colors["path"],  # Path color (e.g., red)
                    False,  # Don't connect the last point to the first one
                    [pos.astype(int) for pos in agent.path],  # Convert positions to integer for Pygame
                    2  # Line thickness
                )
    def draw_flow_field(self, simulator: Simulator) -> None:
        """Draw a flow field of vectors based on the agent's path."""
        if not simulator.active_agent or not simulator.active_agent.path:
            return

        # Parameters for the flow field grid
        grid_size = 30  # Increase grid size to reduce computations
        rows = int(simulator.table.table_height / grid_size)
        cols = int(simulator.table.table_width / grid_size)

        # Precompute path segment vectors and lengths
        path = simulator.active_agent.path
        segment_vectors = []
        segment_lengths = []
        for i in range(len(path) - 1):
            start = np.array(path[i])
            end = np.array(path[i + 1])
            segment_vector = end - start
            segment_length = np.linalg.norm(segment_vector)
            if segment_length > 0:
                segment_vectors.append(segment_vector / segment_length)  # Normalize
                segment_lengths.append(segment_length)
            else:
                segment_vectors.append(segment_vector)
                segment_lengths.append(0)

        # Precompute grid points
        grid_points = [
            (simulator.table.top_left[0] + col * grid_size + grid_size // 2,
            simulator.table.top_left[1] + row * grid_size + grid_size // 2)
            for row in range(rows) for col in range(cols)
        ]

        # Iterate through each grid cell and determine the flow direction
        for x, y in grid_points:
            closest_direction = None
            min_distance = float('inf')

            for i, (segment_vector, segment_length) in enumerate(zip(segment_vectors, segment_lengths)):
                if segment_length == 0:
                    continue

                start = np.array(path[i])
                end = np.array(path[i + 1])

                # Project the grid point onto the path segment
                t = np.clip(np.dot(np.array([x, y]) - start, segment_vector) / segment_length, 0, 1)
                projection = start + t * (end - start)

                # Compute distance to the projected point
                distance = np.linalg.norm(np.array([x, y]) - projection)
                if distance < min_distance:
                    min_distance = distance
                    closest_direction = segment_vector

            if closest_direction is not None:
                # Draw the arrow (line representing flow field direction)
                arrow_length = 10  # Length of the arrow
                end_x = x + closest_direction[0] * arrow_length
                end_y = y + closest_direction[1] * arrow_length

                # Draw the line (direction arrow)
                pygame.draw.line(self.screen, (0, 255, 0), (x, y), (end_x, end_y), 2)

    def render(self, simulator: Simulator) -> None:
        """
        Render the simulation scene, including the table, bins, and agents.
        """
        if self.screen is None:
            self.initialize()

        self.screen.fill(self.colors["background"])

        # Draw the table, bins, and object
        self.draw_table(simulator)
        self.draw_bins(simulator)  # Assuming active agent is the first
        self.draw_object(simulator)
        # self.draw_flow_field(simulator)  # Draw the flow field
        pygame.display.flip()
        self.clock.tick(30)

    def close(self) -> None:
        """
        Close the Pygame window and perform cleanup.
        """
        if self.screen is not None:
            pygame.quit()




