import numpy as np
import pygame
from gymnasium import Env, spaces
from gymnasium.utils import seeding
from collections import deque

class BinpickingPathFindingEnv(Env):
    def __init__(
        self,
        screen_width=800,
        screen_height=600,
        table_width=600,
        table_height=350,
        step_size=10,
        time_limit=500,
    ):
        super(BinpickingPathFindingEnv, self).__init__()

        self.screen_width = screen_width
        self.screen_height = screen_height
        self.table_width = table_width
        self.table_height = table_height
        self.step_size = step_size
        self.time_limit = time_limit

        self.table_top_left = (
            (self.screen_width - self.table_width) // 2,
            (self.screen_height - self.table_height) // 2,
        )
        self.table_bottom_right = (
            self.table_top_left[0] + self.table_width,
            self.table_top_left[1] + self.table_height,
        )

        # Observation space: object position and all goals
        self.observation_space = spaces.Box(
            low=0, high=max(screen_width, screen_height), shape=(2 + 4 * 2,), dtype=np.float32
        )

        # Use a symmetric and normalized Box action space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.colors = {
            "background": (30, 30, 30),
            "table": (50, 50, 150),
            "object": (0, 255, 0),
            "goal": (255, 0, 0),
            "goal_reached": (0, 255, 255),  # Cyan for reached goals
            "arrow": (255, 255, 255),  # Arrow color
        }
        self.prev_positions = deque(maxlen=10)
        self.goal_width = 90
        self.goal_height = 120
        self.object_radius = 25
        self.arrow_length = 50
        self.entry_offset = 30  # Offset to move entry position into the table
        self.screen = None
        self.clock = None
        # Define goal entry sides
        self.entry_sides = np.array([
            [0, 1],  # Top-left goal: enter from below
            [0, 1],  # Top-right goal: enter from below
            [0, -1],  # Bottom-left goal: enter from above
            [0, -1],  # Bottom-right goal: enter from above
        ])
        self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)

        self.object = np.random.rand(2) * [
            self.table_width - 2 * self.object_radius,
            self.table_height - 2 * self.object_radius,
        ]
        self.object += [
            self.table_top_left[0] + self.object_radius,
            self.table_top_left[1] + self.object_radius,
        ]
        self.object = self.object.astype(np.float32)

        top_goal_x = self.table_top_left[0] + self.goal_width
        bottom_goal_x = self.table_bottom_right[0] - self.goal_width
        top_goal_y = self.table_top_left[1] - self.goal_height // 2
        bottom_goal_y = self.table_bottom_right[1] + self.goal_height // 2

        self.goals = np.array([
            [top_goal_x, top_goal_y],
            [bottom_goal_x, top_goal_y],
            [top_goal_x, bottom_goal_y],
            [bottom_goal_x, bottom_goal_y],
        ]).astype(np.float32)

        distances = np.linalg.norm(self.goals - self.object, axis=1)
        self.closest_goal_index = np.argmin(distances)

        self.time_steps = 0
        # Clear previous positions
        self.prev_positions.clear()
        self.prev_positions.append(self.object.copy())
        self.velocity = np.zeros(2)  # Initialize velocity if not present
        return self.get_observation(), {}

    def step(self, action):
        # Ensure action is within the valid range
        angle = action[0] * np.pi  # Map normalized action to [-π, π]
        move = np.array([np.cos(angle), np.sin(angle)]) * self.step_size   
                      
        self.velocity = 0.9 * self.velocity + 0.1 * move
        new_pos = self.object + self.velocity

        # Check if the object is close enough to the goal
        target = self.goals[self.closest_goal_index]
        goal_direction = self.goal_direction(target)

        # Project the velocity onto the goal direction for alignment
        move_direction = new_pos - self.object
        move_direction_norm = np.linalg.norm(move_direction)
        if move_direction_norm > 0:
            move_direction /= move_direction_norm
            alignment = np.dot(move_direction, goal_direction)
            new_pos = self.object + alignment * move_direction * self.step_size

        reward = 0
        done = False
        truncated = False

        # Check bounds
        if (
            new_pos[0] < self.table_top_left[0]
            or new_pos[0] > self.table_bottom_right[0]
            or new_pos[1] < self.table_top_left[1]
            or new_pos[1] > self.table_bottom_right[1]
        ):
            reward -= 100
            done = True
        else:
            self.object = new_pos
            
            reward += 50 * alignment 
            
            goal_direction = self.goal_direction(target)
            
            # Reward for alignment with entry side
            entry_vector = self.get_goal_entry_vector(self.closest_goal_index)
            alignment_with_entry = np.dot(goal_direction, entry_vector)
            reward += 30 * max(alignment_with_entry, 0)  # Reward only for correct alignment

            # Reward for moving closer to the goal
            prev_distance_to_goal = np.linalg.norm(self.object - target)
            distance_to_goal = np.linalg.norm(self.object - target)
            distance_change = prev_distance_to_goal - distance_to_goal

            if distance_change > 0:
                reward += 10 * distance_change
            else:
                reward -= 5 * abs(distance_change)

            # Oscillation penalty
            if len(self.prev_positions) == self.prev_positions.maxlen:
                std_dev = np.std([np.linalg.norm(self.object - pos) for pos in self.prev_positions])
                if std_dev < 1e-2:
                    reward -= 20  # Stronger penalty for oscillatory behavior

            self.prev_positions.append(self.object.copy())

            # Goal completion check
            goal_rect = pygame.Rect(
                target[0] - self.goal_width // 2,
                target[1] - self.goal_height // 2,
                self.goal_width,
                self.goal_height,
            )

            if self.is_correct_goal_entry(goal_rect, self.object, self.object_radius, entry_vector):
                reward += 100  # Large reward for reaching the goal correctly
                done = True

        # Step penalty
        reward -= 0.5
        self.time_steps += 1
        if self.time_steps >= self.time_limit:
            done = True
            reward -= 10

        return self.get_observation(), reward, done, truncated, {}

    def render(self, mode="human"):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Rectangular Table Pathfinding")
            self.clock = pygame.time.Clock()

        self.screen.fill(self.colors["background"])

        # Draw the table
        pygame.draw.rect(
            self.screen,
            self.colors["table"],
            pygame.Rect(
                self.table_top_left[0], self.table_top_left[1], self.table_width, self.table_height
            ),
        )

        # Draw the goals
        for i, goal in enumerate(self.goals):
            color = (
                self.colors["goal_reached"] if i == self.closest_goal_index else self.colors["goal"]
            )
            pygame.draw.rect(
                self.screen,
                color,
                pygame.Rect(
                    goal[0] - self.goal_width // 2,
                    goal[1] - self.goal_height // 2,
                    self.goal_width,
                    self.goal_height,
                ),
            )

            # Draw an arrow indicating the entry direction
            entry_side = self.entry_sides[i]
            arrow_start = goal + (self.entry_offset * entry_side)
            arrow_end = arrow_start + (self.arrow_length * entry_side)
            pygame.draw.line(self.screen, self.colors["arrow"], arrow_start.astype(int), arrow_end.astype(int), 2)

        # Draw the object
        pygame.draw.circle(self.screen, self.colors["object"], self.object.astype(int), self.object_radius)

        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        if self.screen is not None:
            pygame.quit()

    def is_completely_inside_goal(self, goal_rect, object_pos, radius):
        top_left = object_pos - radius
        bottom_right = object_pos + radius
        return (
            goal_rect.left <= top_left[0]
            and goal_rect.right >= bottom_right[0]
            and goal_rect.top <= top_left[1]
            and goal_rect.bottom >= bottom_right[1]
        )
    def get_goal_entry_vector(self, goal_index):
        """
        Returns the entry vector for the given goal index.
        Top goals (index 0 and 1): Entry from below (0, 1)
        Bottom goals (index 2 and 3): Entry from above (0, -1)
        """
        if goal_index in [0, 1]:  # Top goals
            return np.array([0, 1], dtype=np.float32)
        elif goal_index in [2, 3]:  # Bottom goals
            return np.array([0, -1], dtype=np.float32)

    def is_correct_goal_entry(self, goal_rect, object_pos, radius, entry_vector):
        """
        Checks if the object enters the goal correctly.
        It must be fully inside the goal and its movement should align with the entry vector.
        """
        top_left = object_pos - radius
        bottom_right = object_pos + radius
        if not (
            goal_rect.left <= top_left[0]
            and goal_rect.right >= bottom_right[0]
            and goal_rect.top <= top_left[1]
            and goal_rect.bottom >= bottom_right[1]
        ):
            return False

        # Check entry direction alignment
        movement_vector = self.velocity / np.linalg.norm(self.velocity) if np.linalg.norm(self.velocity) > 0 else np.zeros(2)
        alignment = np.dot(movement_vector, entry_vector)
        return alignment > 0.9  # Require high alignment with the entry vector
    def goal_direction(self, target):
        goal_vector = target - self.object
        goal_distance = np.linalg.norm(goal_vector)
        return goal_vector / goal_distance if goal_distance > 0 else np.array([0, 0])
    def get_observation(self):
        return np.concatenate([self.object.flatten(), self.goals.flatten()]).astype(np.float32)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return seed


