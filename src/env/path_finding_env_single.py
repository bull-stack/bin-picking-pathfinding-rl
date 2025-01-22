import numpy as np
import pygame
from gymnasium import Env, spaces
from gymnasium.utils import seeding
from collections import deque

class BinpickingPathFindingEnvS(Env):
    def __init__(
        self,
        screen_width=800,
        screen_height=600,
        table_width=600,
        table_height=350,
        step_size=5,
        time_limit=200,
    ):
        super(BinpickingPathFindingEnvS, self).__init__()

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
        
        self.goal_width = 90
        self.goal_height = 120
        self.object_radius = 25
        self.arrow_length = 50
        self.entry_offset = 30  # Offset to move entry position into the table
        self.screen = None
        self.clock = None

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
        self.prev_positions = deque(maxlen=10)
        self.prev_positions.clear()  # Clear previous positions
        self.prev_positions.append(self.object.copy())  # Add the starting position
        
        self.time_steps = 0
        self.velocity = np.zeros(2)  # Initialize velocity if not present
        return self.get_observation(), {}

    def step(self, action):
        
        reward = 0
        done = False
        truncated = False

        # Ensure action is within the valid range
        angle = action[0] * np.pi  # Map normalized action to [-π, π]
        move = np.array([np.cos(angle), np.sin(angle)]) * self.step_size

        # Smooth the movement
        blending_factor = 0.7
        self.velocity = blending_factor * self.velocity + (1 - blending_factor) * move
        velocity_magnitude = np.linalg.norm(self.velocity)

        # Normalize velocity for consistent speed
        if velocity_magnitude > 0:
            self.velocity = (self.velocity / velocity_magnitude) * self.step_size
        new_pos = self.object + self.velocity

        # Calculate goal direction and alignment
        target = self.goals[self.closest_goal_index]
        goal_direction = self.goal_direction(target)
        
        prev_distance_to_goal = np.linalg.norm(self.object - target)
        distance_to_goal = np.linalg.norm(new_pos - target)
        distance_change = prev_distance_to_goal - distance_to_goal
        # Strong reward for getting closer to the target goal
        if distance_to_goal < prev_distance_to_goal:
            reward += 10 * distance_change
        else:
            reward -= 5 * abs(distance_change)   
        

        # Check if the object is entering the goal area
        goal_entry_zone = self.get_goal_entry_zone(self.closest_goal_index)
        in_entry_zone = goal_entry_zone.collidepoint(int(new_pos[0]), int(new_pos[1]))

        # Penalize changes in direction
        # if np.linalg.norm(self.velocity) > 0 and np.linalg.norm(move) > 0:
        #     current_direction = self.velocity / np.linalg.norm(self.velocity)
        #     new_direction = move / np.linalg.norm(move)
        #     angle_change = np.dot(current_direction, new_direction)
        #     reward -= 10 * (1 - angle_change)  # Mild penalty for direction change

        if in_entry_zone:
            reward += 75  # Bonus for entering the goal area
            self.object = new_pos

            # Check if fully inside the goal
            goal_rect = pygame.Rect(
                target[0] - self.goal_width // 2,
                target[1] - self.goal_height // 2,
                self.goal_width,
                self.goal_height,
            )
            if goal_rect.contains(
                pygame.Rect(
                    new_pos[0] - self.object_radius,
                    new_pos[1] - self.object_radius,
                    self.object_radius * 2,
                    self.object_radius * 2,
                )
            ):
                reward += 200  # Large reward for goal completion
                done = True
        else:
            # Penalize going out of bounds
            if (
                new_pos[0] < self.table_top_left[0]
                or new_pos[0] > self.table_bottom_right[0]
                or new_pos[1] < self.table_top_left[1]
                or new_pos[1] > self.table_bottom_right[1]
            ):
                
                done = True
            else:
                self.object = new_pos
                # reward += 10
                # Reward for alignment with goal direction
                # goal_direction = self.goal_direction(target)
                # entry_vector = self.get_goal_entry_vector(self.closest_goal_index)
                # alignment_with_entry = np.dot(goal_direction, entry_vector)
                # reward += 50 * max(alignment_with_entry, 0)
                # Reward for moving closer to the goal
                

        # Step penalty
        reward -= 0.5 
        self.time_steps += 1
        if self.time_steps >= self.time_limit:
            done = True
            reward -= 20  # Mild penalty for timeouts

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
            goal_entry_zone = self.get_goal_entry_zone(i)
            pygame.draw.rect(self.screen, (100, 100, 255), goal_entry_zone, 2)
            # Draw an arrow indicating the entry direction
            entry_side = self.get_goal_entry_vector(i)
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

    def get_goal_entry_zone(self, goal_index):
        goal = self.goals[goal_index]
        entry_vector = self.get_goal_entry_vector(goal_index)
        entry_length = self.arrow_length

        # Define the rectangle extending in the entry direction
        if np.array_equal(entry_vector, [0, 1]):  # Top entry
            rect = pygame.Rect(
                goal[0] - self.goal_width // 2,
                self.table_top_left[1] - entry_length,
                self.goal_width,
                entry_length,
            )
        elif np.array_equal(entry_vector, [0, -1]):  # Bottom entry
            rect = pygame.Rect(
                goal[0] - self.goal_width // 2,
                self.table_bottom_right[1],
                self.goal_width,
                entry_length,
            )
        else:
            raise ValueError("Invalid entry vector for goal.")

        return rect
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

    
    def goal_direction(self, target):
        goal_vector = target - self.object
        goal_distance = np.linalg.norm(goal_vector)
        return goal_vector / goal_distance if goal_distance > 0 else np.array([0, 0])
    
    def get_observation(self):
        return np.concatenate([self.object.flatten(), self.goals.flatten()]).astype(np.float32)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return seed