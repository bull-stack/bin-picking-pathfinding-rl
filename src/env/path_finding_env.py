import gymnasium
from gymnasium import spaces
import numpy as np
import pygame

class PathFindingEnv(gymnasium.Env):
    def __init__(self, rows=8, columns=12, cell_size=50, num_objects=4):
        super(PathFindingEnv, self).__init__()

        self.rows = rows
        self.columns = columns
        self.cell_size = cell_size
        self.num_objects = num_objects
        self.grid_size = (rows, columns)
        self.screen_size = (columns * cell_size, rows * cell_size)

        # Define action space: (object index, direction)
        self.action_space = spaces.MultiDiscrete([num_objects, 4])
        self.observation_space = spaces.Box(
            low=0, high=max(rows, columns) - 1, shape=(self.num_objects * 2,), dtype=np.int32
        )

        # Define goal positions (outside the grid on edges)
        self.goals = [
            np.array([-1, 1]),  # Top-left goal
            np.array([-1, columns - 2]),  # Top-right goal
            np.array([rows, 1]),  # Bottom-left goal
            np.array([rows, columns - 2]),  # Bottom-right goal
        ]

        # Rendering settings
        self.colors = {
            "background": (30, 30, 30),
            "grid_lines": (60, 60, 60),
            "object": (0, 255, 0),
            "goal": (255, 0, 0),
        }

        self.screen = None
        self.clock = None

        self.reset()

    def reset(self):
        self.objects = []
        for _ in range(self.num_objects):
            obj = np.random.randint(1, self.rows - 1), np.random.randint(1, self.columns - 1)
            while any((obj == existing).all() for existing in self.objects):
                obj = np.random.randint(1, self.rows - 1), np.random.randint(1, self.columns - 1)
            self.objects.append(np.array(obj))

        self.objects = np.array(self.objects)

        return self._get_state()

    def step(self, action):
        object_index, direction = action
        moves = {
            0: [-1, 0],  # Up
            1: [1, 0],   # Down
            2: [0, -1],  # Left
            3: [0, 1],   # Right
        }

        move = moves[direction]
        new_pos = self.objects[object_index] + move
        new_pos = np.clip(new_pos, 0, [self.rows - 1, self.columns - 1])

        if any((new_pos == obj).all() for i, obj in enumerate(self.objects) if i != object_index):
            reward = -1  # Penalize collisions
        else:
            self.objects[object_index] = new_pos

        if any((new_pos == goal).all() for goal in self.goals):
            reward = 10
            self.objects = np.delete(self.objects, object_index, axis=0)
        else:
            reward = -0.1

        done = len(self.objects) == 0
        return self._get_state(), reward, done, {}

    def render(self, mode='human'):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode(self.screen_size)
            pygame.display.set_caption("Custom Rectangular Table Environment")
            self.clock = pygame.time.Clock()

        self.screen.fill(self.colors["background"])

        # Draw grid starting from the second row/column and stopping before last row/column
        for x in range(self.cell_size, self.screen_size[0] - self.cell_size, self.cell_size):
            pygame.draw.line(self.screen, self.colors["grid_lines"], (x, self.cell_size), (x, self.screen_size[1] - self.cell_size))
        for y in range(self.cell_size, self.screen_size[1] - self.cell_size, self.cell_size):
            pygame.draw.line(self.screen, self.colors["grid_lines"], (self.cell_size, y), (self.screen_size[0] - self.cell_size, y))

        # Draw goals
        for goal in self.goals:
            rect = pygame.Rect(
                goal[1] * self.cell_size if goal[1] >= 0 else 0,
                goal[0] * self.cell_size if goal[0] >= 0 else self.rows * self.cell_size,
                self.cell_size,
                self.cell_size,
            )
            pygame.draw.rect(self.screen, self.colors["goal"], rect)

        # Draw objects
        for obj in self.objects:
            rect = pygame.Rect(
                obj[1] * self.cell_size, obj[0] * self.cell_size, self.cell_size, self.cell_size
            )
            pygame.draw.rect(self.screen, self.colors["object"], rect)

        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        if self.screen is not None:
            pygame.quit()

    def _get_state(self):
        return self.objects.flatten()

