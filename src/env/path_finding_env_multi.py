import numpy as np
import pygame
from gymnasium import Env, spaces
from gymnasium.utils import seeding
from collections import deque

class BinpickingPathFindingEnvM(Env):
    def __init__(
        self,
        screen_width=800,
        screen_height=600,
        table_width=600,
        table_height=350,
        step_size=10,
        time_limit=250,
        num_objects=4,
    ):
        super(BinpickingPathFindingEnvM, self).__init__()

        self.screen_width = screen_width
        self.screen_height = screen_height
        self.table_width = table_width
        self.table_height = table_height
        self.step_size = step_size
        self.time_limit = time_limit
        self.num_objects = num_objects

        self.table_top_left = (
            (self.screen_width - self.table_width) // 2,
            (self.screen_height - self.table_height) // 2,
        )
        self.table_bottom_right = (
            self.table_top_left[0] + self.table_width,
            self.table_top_left[1] + self.table_height,
        )

        # Observation space: object positions and goals
        self.observation_space = spaces.Box(
            low=0,
            high=max(screen_width, screen_height),
            shape=(2 * self.num_objects + 2 * self.num_objects,),
            dtype=np.float32,
        )

        # Action space: normalized angle [-1, 1]
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
        self.goal_radius = 30
        self.arrow_length = 50
        self.entry_offset = 30  # Offset to move entry position into the table
        self.screen = None
        self.clock = None
        self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)

        self.objects = []
        self.goals = []
        self.object_goal_pairs = []
        
        min_x = self.table_top_left[0] + self.object_radius
        max_x = self.table_bottom_right[0] - self.object_radius
        min_y = self.table_top_left[1] + self.object_radius
        max_y = self.table_bottom_right[1] - self.object_radius

        # Place objects randomly within the table
        while len(self.objects) < self.num_objects:
            new_object = np.array([np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y),], dtype=np.float32)
            if all(np.linalg.norm(new_object - obj) > 2 * self.object_radius for obj in self.objects):  # Ensure no overlap between objects
                self.objects.append(new_object)
        self.objects = np.array(self.objects, dtype=np.float32)

        # Define goal positions
        top_goal_x = self.table_top_left[0] + self.goal_width
        bottom_goal_x = self.table_bottom_right[0] - self.goal_width
        top_goal_y = self.table_top_left[1] - self.goal_height // 2
        bottom_goal_y = self.table_bottom_right[1] + self.goal_height // 2

        self.goals = np.array([[top_goal_x, top_goal_y],[bottom_goal_x, top_goal_y],[top_goal_x, bottom_goal_y],
                               [bottom_goal_x, bottom_goal_y],]).astype(np.float32)

        # Assign object-goal pairs by shortest distance
        self.pair_idx = []
        for obj_idx, obj in enumerate(self.objects):
            distances = [np.linalg.norm(obj - goal) for goal in self.goals]
            closest_goal_index = np.argmin(distances)
            self.pair_idx.append([obj_idx, closest_goal_index])
        self.pair_idx = np.array(self.pair_idx, dtype=np.int32)
        
        self.current_agent = 0
        self.time_steps = 0
        self.prev_positions = [deque(maxlen=10) for _ in range(self.num_objects)]
        self.velocity = np.zeros((self.num_objects, 2), dtype=np.float32)


        return self.get_observation(), {}


    def step(self, action):
        angle = action[0] * np.pi  # Map action to [-pi, pi]
        move = np.array([np.cos(angle), np.sin(angle)]) * self.step_size

        obj_idx, goal_idx = self.pair_idx[self.current_agent]

        # Compute new velocity and position
        self.velocity[obj_idx] = 0.9 * self.velocity[obj_idx] + 0.1 * move
        new_pos = self.objects[obj_idx] + self.velocity[obj_idx]

        current_object = self.objects[obj_idx]
        current_goal = self.goals[goal_idx]

        # Project the velocity onto the goal direction for alignment
        goal_direction = self.goal_direction(current_goal, current_object)
        move_direction = new_pos - self.objects[obj_idx]
        move_direction_norm = np.linalg.norm(move_direction)
        if move_direction_norm > 0:
            move_direction /= move_direction_norm
            alignment = np.dot(move_direction, goal_direction)
            new_pos = self.objects[obj_idx] + alignment * move_direction * self.step_size

        # Check if the object is entering the goal area
        goal_entry_zone = self.get_goal_entry_zone(goal_idx)
        in_entry_zone = goal_entry_zone.collidepoint(int(new_pos[0]), int(new_pos[1]))

        reward = 0
        done = False
        truncated = False

        # If the object is entering the goal, allow movement outside the table bounds
        if in_entry_zone:
            # Reward for progress towards the goal
            reward += 50
            self.objects[obj_idx] = new_pos

            # Check if the object is fully inside the goal area
            goal_rect = pygame.Rect(
                current_goal[0] - self.goal_width // 2,
                current_goal[1] - self.goal_height // 2,
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
                reward += 100  # Large reward for completing the goal
                self.current_agent += 1
                done = self.current_agent >= self.num_objects - 1
        else:
            # If not entering the goal, check if the object is out of bounds
            object_out_of_table = (
                new_pos[0] < self.table_top_left[0]
                or new_pos[0] > self.table_bottom_right[0]
                or new_pos[1] < self.table_top_left[1]
                or new_pos[1] > self.table_bottom_right[1]
            )

            if object_out_of_table:
                # Object moved out of table bounds without entering the goal
                reward -= 100  # Penalty for going out of bounds
                done = True  # Terminate the episode
            else:
                # Normal in-table movement logic
                self.objects[obj_idx] = new_pos
                reward += 30 * max(0, alignment)

                # Reward for alignment with goal direction
                goal_direction = self.goal_direction(current_goal, current_object)
                entry_vector = self.get_goal_entry_vector(goal_idx)
                alignment_with_entry = np.dot(goal_direction, entry_vector)
                reward += 50 * max(alignment_with_entry, 0)

                # Reward for reducing distance to the goal
                prev_distance_to_goal = np.linalg.norm(current_object - current_goal)
                distance_to_goal = np.linalg.norm(self.objects[obj_idx] - current_goal)
                distance_change = prev_distance_to_goal - distance_to_goal
                if distance_to_goal < prev_distance_to_goal:
                    reward += 10 * (distance_change)
                if distance_change > 0:
                    reward += 10 * distance_change
                else:
                    reward -= 5 * abs(distance_change)

                # Track the previous positions
                self.prev_positions[obj_idx].append(new_pos)

                # Check for oscillation and penalize based on severity
                if len(self.prev_positions[obj_idx]) > 3:
                    # Calculate the differences between consecutive positions
                    deltas = [
                        np.linalg.norm(self.prev_positions[obj_idx][i] - self.prev_positions[obj_idx][i - 1])
                        for i in range(1, len(self.prev_positions[obj_idx]))
                    ]

                    # Calculate a moving average of the deltas
                    smoothed_deltas = np.convolve(deltas, np.ones(3) / 3, mode="valid")

                    # Measure oscillation severity
                    oscillation_amplitude = np.std(smoothed_deltas)

                    # Apply a penalty scaled to oscillation amplitude
                    oscillation_threshold = self.object_radius * 0.2
                    if oscillation_amplitude > oscillation_threshold:
                        penalty = min(50, 10 * (oscillation_amplitude - oscillation_threshold))
                        reward -= penalty

        # Apply step penalty and time truncation
        reward -= 0.5
        self.time_steps += 1
        if self.time_steps >= self.time_limit:
            truncated = True
            reward -= 100

        return self.get_observation(), reward, done, truncated, {}

    def render(self, mode="human"):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Bin Picking and Pathfinding Environment")
            self.clock = pygame.time.Clock()

        self.screen.fill(self.colors["background"])

        pygame.draw.rect(
            self.screen,
            self.colors["table"],
            pygame.Rect(
                self.table_top_left[0],
                self.table_top_left[1],
                self.table_width,
                self.table_height,
            ),
        )

        for i, goal in enumerate(self.goals):
            
            color = (
                self.colors["goal_reached"] if i == self.pair_idx[self.current_agent][1] else self.colors["goal"]
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
            entry_zone = self.get_goal_entry_zone(i)
            pygame.draw.rect(self.screen, (100, 100, 255), entry_zone, 2)
            # Draw an arrow indicating the entry direction
            entry_side = self.get_goal_entry_vector(i)
            arrow_start = goal + (self.entry_offset * entry_side)
            arrow_end = arrow_start + (self.arrow_length * entry_side)
            pygame.draw.line(self.screen, self.colors["arrow"], arrow_start.astype(int), arrow_end.astype(int), 2)

        for i, obj in enumerate(self.objects):
            color = self.colors["object"] if i == self.current_agent else (100, 100, 100)
            pygame.draw.circle(self.screen, color, obj.astype(int), self.object_radius)

        
        pygame.display.flip()
        self.clock.tick(30)
        
    def get_goal_entry_zone(self, goal_index):
        """Get the entry zone rectangle for a given goal."""
        goal = self.goals[goal_index]
        entry_vector = self.get_goal_entry_vector(goal_index)
        entry_length = self.arrow_length

        # Define the rectangle extending in the entry direction
        if np.array_equal(entry_vector, [0, 1]):  # Top entry
            return pygame.Rect(
                goal[0] - self.goal_width // 2,
                self.table_top_left[1] - entry_length,
                self.goal_width,
                entry_length,
            )
        elif np.array_equal(entry_vector, [0, -1]):  # Bottom entry
            return pygame.Rect(
                goal[0] - self.goal_width // 2,
                self.table_bottom_right[1],
                self.goal_width,
                entry_length,
            )
        else:
            raise ValueError("Invalid entry vector for goal.")
        
    def get_goal_entry_vector(self, goal_index):
        if goal_index in [0, 1]:
            return np.array([0, 1], dtype=np.float32)
        elif goal_index in [2, 3]:
            return np.array([0, -1], dtype=np.float32)

    def is_correct_goal_entry(self, goal_rect, object_pos, radius, entry_vector):
        top_left = object_pos - radius
        bottom_right = object_pos + radius
        if not (
            goal_rect.left <= top_left[0]
            and goal_rect.right >= bottom_right[0]
            and goal_rect.top <= top_left[1]
            and goal_rect.bottom >= bottom_right[1]
        ):
            return False

        movement_vector = self.velocity[self.pair_idx[self.current_agent][0]] / np.linalg.norm(self.velocity[self.pair_idx[self.current_agent][0]]) if np.linalg.norm(self.velocity[self.pair_idx[self.current_agent][0]]) > 0 else np.zeros(2)
        alignment = np.dot(movement_vector, entry_vector)
        return alignment > 0.5

    def goal_direction(self, target, object_pos):
        goal_vector = target - object_pos
        goal_distance = np.linalg.norm(goal_vector)
        return goal_vector / goal_distance if goal_distance > 0 else np.array([0, 0])

    def close(self):
        if self.screen is not None:
            pygame.quit()

    def get_observation(self):
        return np.concatenate([self.objects.flatten(), self.goals.flatten()]).astype(np.float32)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return seed
