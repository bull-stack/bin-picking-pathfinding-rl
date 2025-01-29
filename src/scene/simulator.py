from table import Table
from agent import Agent
from bin import Bin
import numpy as np
import threading
import time
from typing import Optional, Tuple, List, Dict

class Simulator:
    
    def __init__(
        self,
        screen_width: int = 800,
        screen_height: int = 600,
        table_width: int = 600,
        table_height: int = 350,
        bin_width: int = 90,
        bin_height: int = 150,
        time_limit: int = 300,
        entry_offset: int = 30,
        num_agents: int = 1,
        agent_size: int = 25,
        visualization = None,
    ) -> None:
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.table_width = table_width
        self.table_height = table_height
        self.bin_width = bin_width
        self.bin_height = bin_height
        self.time_limit = time_limit
        self.entry_offset = entry_offset
        self.num_agents = num_agents
        self.agent_size = agent_size

        self.table = Table(screen_width, screen_height, table_width, table_height)
        self.bins: List[Bin] = []
        self.agents: List[Agent] = []
        self.active_agent: Optional[Agent] = None
        self.target_bin: Optional[Bin] = None
        self.time_steps: int = 0
        self.visualization = visualization
        self.collision_threshold = 0.1  # Minimum safe distance between agents
        self.initialize_bins()

    def initialize_bins(self) -> None:
        top_x = self.table.top_left[0] + self.bin_width
        bottom_x = self.table.bottom_right[0] - self.bin_width
        top_y = self.table.top_left[1] - self.bin_height // 2
        bottom_y = self.table.bottom_right[1] + self.bin_height // 2

        bin_positions = [
            (top_x, top_y, [0, 1]),
            (bottom_x, top_y, [0, 1]),
            (top_x, bottom_y, [0, -1]),
            (bottom_x, bottom_y, [0, -1])
        ]

        self.bins.extend(
            Bin(
                position=np.array([x, y], dtype=np.float32),
                width=self.bin_width,
                height=self.bin_height,
                entry_offset=self.entry_offset,
                entry_vector=np.array(vector, dtype=np.float32)
            )
            for x, y, vector in bin_positions
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            np.random.seed(seed)
        self.reset_agents()
        self.reset_bins()
        self.select_active_agent()
        self.assign_closest_bin_as_target()
        self.time_steps = 0
        return self.get_state(), {}

    def reset_agents(self) -> None:
        self.active_agent = None
        self.agents.clear()
        for _ in range(self.num_agents):
            self.add_agent()
        
        # Clear paths for all agents
        for agent in self.agents:
            agent.path.clear()  # Clear previous paths

    def reset_bins(self) -> None:
        self.target_bin = None
        for bin in self.bins:
            bin.in_use = False

    def add_agent(self) -> None:
        agent = Agent(radius=self.agent_size)
        agent.randomize_position(self.table)
        self.agents.append(agent)

    # def get_state(self) -> np.ndarray:
    #     agent_state = np.concatenate([agent.position for agent in self.agents])
    #     bin_state = np.concatenate([bin.position for bin in self.bins])
    #     return np.concatenate([agent_state, bin_state]).astype(np.float32)

    def get_state(self) -> np.ndarray:
        state = []
        
        # Relative position to the target bin
        if self.target_bin and self.active_agent:
            if self.active_agent.in_entry_zone:
                relative_position = self.target_bin.position - self.active_agent.position
                state.extend(relative_position)

            else: 
                bin_entry_zone = self.target_bin.get_bin_entry_zone(self.table)
                x, y, _, _ = bin_entry_zone
                relative_position = [x, y] - self.active_agent.position
                state.extend(relative_position)

            # Distance to the target bin entry zone
            distance_to_bin = np.linalg.norm(relative_position)
            state.append(distance_to_bin)
        else:
            # Default values if no target bin is assigned
            state.extend([0, 0])
            state.append(float('inf'))

        # Agent's position (normalized)
        if self.active_agent:
            normalized_agent_position = self.active_agent.position / np.array([self.table_width, self.table_height], dtype=np.float32)
            state.extend(normalized_agent_position)
        else:
            # Default values if no active agent is assigned
            state.extend([0, 0])
        # Time remaining (normalized)
        time_remaining = (self.time_limit - self.time_steps) / self.time_limit
        state.append(time_remaining)

        # Positions of other agents (normalized)
        for agent in self.agents:
            if agent != self.active_agent:
                normalized_position = agent.position / np.array([self.table_width, self.table_height], dtype=np.float32)
                state.extend(normalized_position)

        return np.array(state, dtype=np.float32)
    
    def action(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        reward = 0.0
        done = False
        truncated = False
        

        if self.active_agent is None and self.target_bin is None:
            return self.get_state(), reward, done, truncated, {}
            # Calculate the distance to the target bin
            
        self.active_agent.update(action)
        # for agent in self.agents:
        #     colliding, distance_to_agent = self.is_colliding(agent, new_position)
        #     if colliding:
        #         # Dynamic penalty scaling based on proximity
        #         reward -= min(1.0, 1.0 / (distance_to_agent + 1e-6))  # Avoid division by zero
        # Calculate rewards and penalties

        is_closer, distance_change = self.is_closer_to_target()
        if is_closer:
            reward += min(10.0, distance_change * 10.0)  # Reward proportional to distance improvement (capped at 1)
        else:
            reward -= min(2.0, abs(distance_change) * 5.0)  # Penalty for moving away (capped at -1)
        if self.is_in_entry_zone():
            reward += 1.0  # Small reward for reaching the entry zone
            self.active_agent.in_entry_zone = True
            self.active_agent.update_position()
            if self.is_in_target_bin():
                reward += 2.0  # Reward for successfully reaching the target bin
                self.active_agent.is_done = True
                # Schedule bin release
                delay = np.random.uniform(1, 10)
                self.release_bin_after_delay(delay, self.active_agent, self.target_bin)
                self.active_agent = None
                self.target_bin = None
                # Check if all agents are done
                if all(agent.is_done for agent in self.agents):
                    reward += 2.0  # Bonus reward for completing the task
                    return self.get_state(), reward, True, truncated, {}

                # Assign a new active agent and target bin
                self.select_active_agent()
                self.assign_closest_bin_as_target()
        else:
            if not self.table.contains(self.active_agent.new_position):
                reward -= 10.0  # Penalize leaving the table
                return self.get_state(), reward, True, truncated, {}
            else:
                self.active_agent.update_position()

        reward -= 0.4  # Small step penalty to encourage efficiency
        self.time_steps += 1
        return self.get_state(), reward, done, truncated, {}
    
    
    def select_active_agent(self) -> None:
        if not self.agents:
            self.active_agent = None
            return

        available_agents = [agent for agent in self.agents if not agent.is_done]

        # Find the closest agent if there's at least one available
        if available_agents:
            closest_agent = min(
                available_agents,
                key=lambda agent: self.get_distance_to_closest_bin(agent)
            )
        else:
            closest_agent = None  # Handle case where no agents are available
        self.active_agent = closest_agent

    def assign_closest_bin_as_target(self) -> None:
        if not self.active_agent:
            return

        available_bins = [bin for bin in self.bins if not bin.in_use]
        if not available_bins:
            return

        closest_bin = min(available_bins, key=lambda bin: np.linalg.norm(bin.position - self.active_agent.position))
        closest_bin.in_use = True
        self.target_bin = closest_bin

    def get_distance_to_closest_bin(self, agent: Agent) -> float:
        available_bins = [bin for bin in self.bins if not bin.in_use]
        if not available_bins:
            return float('inf')

        distances = [np.linalg.norm(bin.position - agent.position) for bin in available_bins]
        return min(distances)
    
    def release_bin_after_delay(self, delay: float, agent: Agent, bin: Bin) -> None:
        """
        Releases the bin after a given delay (in seconds).
        Runs in a separate thread to avoid blocking.
        """
        def delayed_release():
            time.sleep(delay)
            bin.in_use = False  # Mark the bin as available
            agent.is_gone = True
        threading.Thread(target=delayed_release, daemon=True).start()
            
    def is_closer_to_target(self) -> Tuple[bool, float]:
        """
        Determines if the agent is closer to the target after moving.
        Returns a tuple of (is_closer, distance_change).
        """
        if not self.target_bin:
            return False, 0.0  # Return a neutral state if there's no target bin
        
        # Determine the reference point (target or entry zone)
        if self.active_agent.in_entry_zone:
            target_position = self.target_bin.position
        else:
            bin_entry_zone = self.target_bin.get_bin_entry_zone(self.table)
            target_position = np.array([bin_entry_zone[0], bin_entry_zone[1]])
        
        # Calculate the distance before and after the move
        prev_distance = np.linalg.norm(self.active_agent.position - target_position)
        distance = np.linalg.norm(self.active_agent.new_position - target_position)
        distance_change = prev_distance - distance
        
        # Return whether the agent is closer and the distance change
        is_closer = distance < prev_distance
        return is_closer, distance_change

    def is_in_entry_zone(self) -> bool:
        """
        Check if the agent is in the entry zone of the target bin.
        """
        if not self.target_bin:
            return False

        bin_entry_zone = self.target_bin.get_bin_entry_zone(self.table)
        x, y, width, height = bin_entry_zone
        # Check if the new position (new_pos) is within the boundaries of the entry zone
        if (x <= self.active_agent.new_position[0] <= x + width 
            and y <= self.active_agent.new_position[1] <= y + height):
            return True
        return False
    
    def is_in_target_bin(self) -> bool:
        """
        Check if the agent is fully inside the target bin.
        This uses a bounding box approach rather than pygame.Rect.
        """
        if not self.target_bin:
            return False
        # Get the boundaries of both the agent and the target bin
        agent_min, agent_max = self.active_agent.get_boundaries()
        bin_min, bin_max = self.target_bin.get_boundaries()  # Assuming target bin also uses a similar method
        
        # Check if the agent's bounding box is fully inside the target bin's bounding box
        return np.all(agent_min >= bin_min) and np.all(agent_max <= bin_max)


    def is_colliding(self, agent: Agent) -> bool:
        """
        Check for collisions between agents.
        """
        # Obstacle avoidance: check for collisions with other agents
        if agent is not self.active_agent and not agent.is_done:
            distance_to_agent = np.linalg.norm(self.active_agent.new_position - agent.position)
            return distance_to_agent < self.collision_threshold, distance_to_agent
        return False, -1
              
    def is_timeout(self) -> bool:
        return self.time_steps >= self.time_limit

    def render(self) -> None:
        if self.visualization:
            self.visualization.render(self)

    def close_visualization(self) -> None:
        if self.visualization:
            self.visualization.close()
