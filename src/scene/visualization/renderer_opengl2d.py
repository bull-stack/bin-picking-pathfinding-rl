import glfw
from OpenGL.GL import *
from OpenGL.GLU import *

from typing import List, Dict
from math import cos, sin

from simulator import Simulator
from visualization_base import VisualizationBase

class RendererOpenGL2D(VisualizationBase):
    def __init__(self, screen_width: int, screen_height: int, colors: Dict[str, tuple], title: str = "Table Pathfinding") -> None:
        self.screen_width: int = screen_width
        self.screen_height: int = screen_height
        self.title: str = title
        self.colors: Dict[str, tuple] = colors
        self.window = None

    def initialize(self) -> None:
        """
        Initialize the OpenGL window using GLFW.
        """
        if not glfw.init():
            raise Exception("GLFW cannot be initialized.")

        self.window = glfw.create_window(self.screen_width, self.screen_height, self.title, None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("GLFW window cannot be created.")

        glfw.make_context_current(self.window)
        glClearColor(0.1, 0.1, 0.1, 1)  # Background color
        glOrtho(0, self.screen_width, 0, self.screen_height, -1, 1)  # Set up an orthographic projection

    def draw_table(self, simulator: Simulator) -> None:
        """
        Draw the table on the screen using OpenGL.
        """
        table = simulator.table
        glColor3f(*self.colors["table"])  # Set color
        glBegin(GL_QUADS)
        glVertex2f(table.top_left[0], table.top_left[1])
        glVertex2f(table.top_left[0] + table.table_width, table.top_left[1])
        glVertex2f(table.top_left[0] + table.table_width, table.top_left[1] + table.table_height)
        glVertex2f(table.top_left[0], table.top_left[1] + table.table_height)
        glEnd()

    def draw_bins(self, simulator: Simulator) -> None:
        """
        Draw bins on the screen using OpenGL.
        """
        for bin in simulator.bins:
            color = (
                self.colors["bin_reached"] if bin == simulator.active_agent.target_bin
                else self.colors["bin_in_use"] if bin.in_use
                else self.colors["bin"]
            )
            glColor3f(*color)  # Set color
            glBegin(GL_QUADS)
            glVertex2f(bin.position[0] - bin.width / 2, bin.position[1] - bin.height / 2)
            glVertex2f(bin.position[0] + bin.width / 2, bin.position[1] - bin.height / 2)
            glVertex2f(bin.position[0] + bin.width / 2, bin.position[1] + bin.height / 2)
            glVertex2f(bin.position[0] - bin.width / 2, bin.position[1] + bin.height / 2)
            glEnd()

            # Draw the entry zone and entry arrow (using lines)
            bin_entry_zone = bin.get_bin_entry_zone(simulator.table)
            x, y, width, height = bin_entry_zone
    
            # Example drawing function: draw a rectangle (OpenGL 2D drawing, or use appropriate library calls)
            # This depends on your rendering library (Pygame, OpenGL, etc.)
            glColor3f(0.4, 0.4, 1.0)  # Entry zone color (blue)
            glBegin(GL_QUADS)
            glVertex2f(x, y)
            glVertex2f(x + width, y)
            glVertex2f(x + width, y + height)
            glVertex2f(x, y + height)
            glEnd()

    def draw_object(self, simulator: Simulator) -> None:
        """
        Draw agents (as circles) using OpenGL.
        """
        for agent in simulator.agents:
            if agent.is_gone:
                continue  # Skip drawing agents that have gone
            color = (
                self.colors["agent"] if agent == simulator.active_agent else self.colors["arrow"]
            )
            glColor3f(*color)  # Set color
            glBegin(GL_POLYGON)
            num_segments = 20
            angle_increment = 2 * 3.14159 / num_segments
            for i in range(num_segments):
                angle = i * angle_increment
                x = agent.position[0] + agent.radius * cos(angle)
                y = agent.position[1] + agent.radius * sin(angle)
                glVertex2f(x, y)
            glEnd()

    def render(self, simulator: Simulator) -> None:
        """
        Render the simulation scene, including the table, bins, and agents.
        """
        if self.window is None:
            self.initialize()
        glClear(GL_COLOR_BUFFER_BIT)  # Clear the screen
        self.draw_table(simulator)
        self.draw_bins(simulator)
        self.draw_object(simulator)
        glfw.swap_buffers(self.window)

    def close(self) -> None:
        """
        Close the OpenGL window and perform cleanup.
        """
        glfw.terminate()
