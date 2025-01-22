from table import Table
from agents import Agents
from bins import Bins
from agent import Agent
from bin import Bin
from visualization.renderer_pygame import Renderer
import numpy as np
import time
import threading
import pygame
class Simulator:
    def __init__(self, screen_width, screen_height, table_width, table_height, time_limit, step_size, num_agents):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.table_width = table_width
        self.table_height = table_height
        self.time_limit = time_limit
        self.step_size = step_size
        self.num_agents = num_agents
        self.renderer = Renderer(
            screen_width, 
            screen_height, 
            colors={
                "background": (30, 30, 30),
                "table": (50, 50, 150),
                "agent": (0, 255, 0),
                "bin": (255, 0, 0),
                "bin_reached": (0, 255, 255),
                "bin_in_use": (0, 125, 180),
                "arrow": (255, 255, 255),
            },
        )
        
        self.table = Table(screen_width, screen_height, table_width, table_height)
        self.bins = Bins(bin_width=90, bin_height=120, entry_offset=30, table=self.table)
        self.agents = Agents(radius=25, table=self.table, bins=self.bins, num_agents=num_agents)
        
        self.bins.initialize_bins()
        self.time_steps = 0

    def reset(self, seed=None):
        np.random.seed(seed)
        self.agents.reset()
        self.bins.reset()
        self.time_steps = 0
        return (self.get_state(), {})
        
    def get_state(self):
        agent_state = np.concatenate([agent.position for agent in self.agents.get_agents()])
        bin_state = np.concatenate([bin.position for bin in self.bins.get_bins()])
        return np.concatenate([agent_state.flatten(), bin_state.flatten()]).astype(np.float32)

    # def action(self, action):
    #     agent:Agent = self.agents.get_active_agent()
    #     reward = 0
    #     done = False
    #     truncated = False
        
    #     agent.update(action)
        
    #     if agent.is_closer():
    #         reward += 10 * (agent.prev_distance - agent.distance)
    #     else:
    #         reward -= 5 * abs(agent.prev_distance - agent.distance)
    #     # Reward for alignment with goal direction
    #     # bin_direction = agent.bin_direction()
    #     # entry_vector = self.bins.get_entry_vector(agent.get_target_bin().index)
    #     # alignment_with_entry = np.dot(bin_direction, entry_vector)
    #     # reward += 50 * max(alignment_with_entry, 0)
        
    #     if agent.in_entry_zone():
    #         reward += 75  # Bonus for entering the goal area
    #         agent.update_position()
    #         if agent.in_target_bin():
    #             reward += 200 
    #             # agent.target_bin.in_use = True
    #             # reset_thread = threading.Thread(target=self.reset_in_use(), args=(agent, np.random.randint(1, 5)))
    #             # reset_thread.start()
    #             done = True
    #             # del agent
    #             # self.agents.pick_new_agent()
    #             # if (self.agents.get_active_agent() is None):
    #             #     reward += 200
    #             #     done = True
    #     else:
    #         if not self.table.contains(agent.position):
    #             done = True
    #             reward -= 50  # Penalty for out of bounds
    #         else:
    #             agent.update_position()
            
    #     reward -= 0.5  # Slight penalty to discourage excessive steps
    #     self.time_steps += 1
    #     if self.is_timeout():
    #         done = True
    #         reward -= 20  # Mild penalty for timeouts
            
    #     return self.get_state(), reward, done, truncated, {} 

    def action(self, action):
        
        reward = 0
        done = False
        truncated = False

        # Ensure action is within the valid range
        angle = action[0] * np.pi  # Map normalized action to [-π, π]
        move = np.array([np.cos(angle), np.sin(angle)]) * self.step_size
        
        agent: Agent = self.agents.get_active_agent()
        new_pos = agent.update(move, self.step_size)

        # Calculate goal direction and alignment
        target: Bin = agent.target_bin
        # goal_direction = self.goal_direction(target)
        
        # Strong reward for getting closer to the target goal
        if agent.is_closer_to_bin(new_pos):
            reward += 10 * agent.get_distance_change()
        else:
            reward -= 5 * abs(agent.get_distance_change())   
        

        # Check if the object is entering the goal area
        bin_entry_zone = agent.target_bin.get_bin_entry_zone(self.table)
        in_entry_zone = bin_entry_zone.collidepoint(int(new_pos[0]), int(new_pos[1]))

        # Penalize changes in direction
        # if np.linalg.norm(self.velocity) > 0 and np.linalg.norm(move) > 0:
        #     current_direction = self.velocity / np.linalg.norm(self.velocity)
        #     new_direction = move / np.linalg.norm(move)
        #     angle_change = np.dot(current_direction, new_direction)
        #     reward -= 10 * (1 - angle_change)  # Mild penalty for direction change

        if in_entry_zone:
            reward += 75  # Bonus for entering the goal area
            agent.position = new_pos

            # Check if fully inside the goal
            goal_rect = pygame.Rect(
                target.position[0] - target.width // 2,
                target.position[1] - target.height // 2,
                target.width,
                target.height,
            )
            if goal_rect.contains(
                pygame.Rect(
                    new_pos[0] - agent.radius,
                    new_pos[1] - agent.radius,
                    agent.radius * 2,
                    agent.radius * 2,
                )
            ):
                reward += 200  # Large reward for goal completion
                done = True
        else:
            # Penalize going out of bounds
            if (
                new_pos[0] < self.table.top_left[0]
                or new_pos[0] > self.table.bottom_right[0]
                or new_pos[1] < self.table.top_left[1]
                or new_pos[1] > self.table.bottom_right[1]
            ):
                
                done = True
            else:
                agent.position = new_pos
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

        return self.get_state(), reward, done, truncated, {}
    
    def is_timeout(self):
        return self.time_steps >= self.time_limit
    
    def reset_in_use(agent, delay):
        time.sleep(delay)
        agent.target_bin.in_use = False
    
    def render(self):
        self.renderer.render(
            self.table,
            self.bins,
            self.agents,
        )
