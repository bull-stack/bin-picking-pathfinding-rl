import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
from typing import List, Dict
from math import cos, sin, pi
from visualization_base import VisualizationBase
from simulator import Simulator

class RendererOpenGL3D(VisualizationBase):
    def __init__(self, screen_width: int, screen_height: int, colors: Dict[str, tuple], title: str = "Table Pathfinding") -> None:
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.title = title
        self.colors = colors
        self.window = None

    def initialize(self) -> None:
        """
        Initialize the OpenGL window and set up 3D projection.
        """
        if not glfw.init():
            raise Exception("GLFW cannot be initialized.")

        self.window = glfw.create_window(self.screen_width, self.screen_height, self.title, None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("GLFW window cannot be created.")

        glfw.make_context_current(self.window)

        # OpenGL settings for 3D rendering
        glClearColor(0.1, 0.1, 0.1, 1)  # Background color
        glEnable(GL_DEPTH_TEST)  # Enable depth testing for proper object layering
        
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, (self.screen_width / self.screen_height), 0.1, 50.0)  # Perspective projection
        glMatrixMode(GL_MODELVIEW)

    def set_camera(self, camera_position: List[float], look_at: List[float], up_vector: List[float]) -> None:
        """
        Set the camera position and orientation.
        """
        glLoadIdentity()
        gluLookAt(camera_position[0], camera_position[1], camera_position[2], 
                  look_at[0], look_at[1], look_at[2], 
                  up_vector[0], up_vector[1], up_vector[2])

    def draw_table(self, simulator: Simulator) -> None:
        """
        Draw the table on the screen in 3D.
        """
        table = simulator.table
        glColor3f(*self.colors["table"])  # Set color
        glBegin(GL_QUADS)
        # Front face of the table (we can use Z-axis to give the table depth)
        glVertex3f(table.top_left[0], table.top_left[1], 0)  # Top-left corner
        glVertex3f(table.top_left[0] + table.table_width, table.top_left[1], 0)
        glVertex3f(table.top_left[0] + table.table_width, table.top_left[1] + table.table_height, 0)
        glVertex3f(table.top_left[0], table.top_left[1] + table.table_height, 0)
        glEnd()

    def draw_bins(self, simulator: Simulator) -> None:
        """
        Draw bins on the screen in 3D.
        """
        for bin in simulator.bins:
            color = (
                self.colors["bin_reached"] if bin == simulator.active_agent.target_bin
                else self.colors["bin_in_use"] if bin.in_use
                else self.colors["bin"]
            )
            glColor3f(*color)  # Set color
            glBegin(GL_QUADS)
            # Draw the bins as 3D cubes with depth (Z-axis)
            glVertex3f(bin.position[0] - bin.width / 2, bin.position[1] - bin.height / 2, 0)
            glVertex3f(bin.position[0] + bin.width / 2, bin.position[1] - bin.height / 2, 0)
            glVertex3f(bin.position[0] + bin.width / 2, bin.position[1] + bin.height / 2, 0)
            glVertex3f(bin.position[0] - bin.width / 2, bin.position[1] + bin.height / 2, 0)
            glEnd()

            # Draw the entry zone (in 3D)
            bin_entry_zone = bin.get_bin_entry_zone(simulator.table)
            glColor3f(0.4, 0.4, 1.0)  # Blue for entry zone
            glBegin(GL_LINE_LOOP)
            for point in bin_entry_zone:
                glVertex3f(point[0], point[1], 0)
            glEnd()


    def draw_object(self, simulator: Simulator) -> None:
        """
        Draw agents (as spheres) on the screen in 3D.
        """
        for agent in simulator.agents:
            if agent.is_gone:
                continue  # Skip drawing agents that have gone
            color = (
                self.colors["agent"] if agent == simulator.active_agent else self.colors["arrow"]
            )
            glColor3f(*color)  # Set color

            # Draw a 3D sphere for each agent
            num_segments = 20
            num_rings = 20
            radius = agent.radius
            for i in range(num_rings):
                lat0 = pi * (-0.5 + float(i) / num_rings)
                z0 = sin(lat0) * radius
                zr0 = cos(lat0) * radius

                lat1 = pi * (-0.5 + float(i + 1) / num_rings)
                z1 = sin(lat1) * radius
                zr1 = cos(lat1) * radius

                glBegin(GL_QUAD_STRIP)
                for j in range(num_segments):
                    lng = 2 * pi * float(j) / num_segments
                    x = cos(lng)
                    y = sin(lng)

                    glVertex3f(agent.position[0] + zr0 * x, agent.position[1] + zr0 * y, agent.position[2] + z0)
                    glVertex3f(agent.position[0] + zr1 * x, agent.position[1] + zr1 * y, agent.position[2] + z1)
                glEnd()

    def render(self, simulator: Simulator) -> None:
        """
        Render the simulation scene, including the table, bins, and agents in 3D.
        """
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)  # Clear both color and depth buffers

        # Set camera view
        self.set_camera(camera_position=[10, 10, 10], look_at=[0, 0, 0], up_vector=[0, 1, 0])

        # Draw the table, bins, and agents
        self.draw_table(simulator)
        self.draw_bins(simulator)
        self.draw_object(simulator)

        glfw.swap_buffers(self.window)

    def close(self) -> None:
        """
        Close the OpenGL window and perform cleanup.
        """
        glfw.terminate()
