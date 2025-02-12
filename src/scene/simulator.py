import numpy as np

from typing import Optional, Tuple, List, Dict

from table import Table
from agent import Agent
from bin import Bin



class Simulator:
    def __init__(self,
        screen_width: int = 800,
        screen_height: int = 600,
        table_width: int = 600,
        table_height: int = 350,
        bin_width: int = 90,
        bin_height: int = 150,
        entry_offset: int = 30,
        num_agents: int = 1,
        agent_size: int = 25,
        visualization = None,
    ) -> None:
        self.screen_width: int = screen_width
        self.screen_height: int = screen_height
        self.table_width: int = table_width
        self.table_height: int = table_height
        self.bin_width: int = bin_width
        self.bin_height: int = bin_height
        self.entry_offset: int = entry_offset
        self.num_agents: int = num_agents
        self.agent_size: int = agent_size

        self.table: Table = Table(screen_width, screen_height, table_width, table_height)
        self.bins: List[Bin] = self._initialize_bins(bin_width, bin_height, entry_offset)
        self.agents: List[Agent] = []
        self.active_agent: Optional[Agent] = None
        self.target_bin: Optional[Bin] = None
        self.visualization = visualization
        self.collision_threshold: float = 0.1  # Minimum safe distance between agents
        self.entry_zone: bool = False
    def _initialize_bins(self, bin_width: int, bin_height: int, entry_offset: int) -> List[Bin]:
        """Initialize bins at predefined positions."""
        positions = [
            (self.table.top_left[0] + bin_width, self.table.top_left[1] - bin_height // 2, [0, 1]),
            (self.table.bottom_right[0] - bin_width, self.table.top_left[1] - bin_height // 2, [0, 1]),
            (self.table.top_left[0] + bin_width, self.table.bottom_right[1] + bin_height // 2, [0, -1]),
            (self.table.bottom_right[0] - bin_width, self.table.bottom_right[1] + bin_height // 2, [0, -1])
        ]
        return [Bin(np.array([x, y], dtype=np.float32), bin_width, bin_height, entry_offset, np.array(vector, dtype=np.float32)) for x, y, vector in positions]

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        self.reset_agents()
        self.reset_bins()
        self.select_active_agent()
        self.select_target_bin()
        self.entry_zone = False

    def reset_agents(self) -> None:
        self.active_agent = None
        self.agents.clear()
        for i in range(self.num_agents):
            agent = Agent(id=i, radius=self.agent_size)
            agent.randomize_position(self.table.table_width, self.table.table_height, self.table.top_left)
            self.agents.append(agent)


    def reset_bins(self) -> None:
        self.target_bin = None
        for bin in self.bins:
            bin.reset()

    def action(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.active_agent.apply_action(action)

    def evaluate_move(self) -> float:
        """Evaluate the active agent's performance. """
        reward = 0.0
        target_pos = self.get_target_pos()

        # Reward for moving in the correct direction (scaled appropriately)
        correct_direction, direction_similarity = self.correct_direction(target_pos)
        if correct_direction:
            reward += max(0.0, direction_similarity) * 5.0  # Scaled up for better guidance

        # Reward for moving closer to the target
        is_closer, distance_change = self.closer_to_target(target_pos)
        if is_closer:
            reward += min(5.0, distance_change * 8.0)  # Adjusted scaling
        else:
            reward -= min(3.0, abs(distance_change) * 6.0)  # Balanced penalty
        return reward
    
    def select_active_agent(self) -> None:
        """Select the agent closest to an available bin."""
        available_agents = [agent for agent in self.agents if not agent.is_done]
        if available_agents:
            self.active_agent = min(available_agents, key=self.get_distance_to_closest_bin)

    def select_target_bin(self) -> None:
        """Assign the closest available bin as the target bin."""
        if self.active_agent:
            self.target_bin = self.get_closest_available_bin(self.active_agent)
            if self.target_bin:
                self.target_bin.available = False

    def get_target_pos(self) -> np.ndarray:
        """ Returns the target position based on the active agent's position. """
        return (
            self.target_bin.position
            if self.entry_zone
            else self.target_bin.get_bin_entry_zone(self.table)[:2]
        )
    def get_distance_to_closest_bin(self, agent: Agent) -> float:
        """Get the shortest distance from the agent to an available bin."""
        closest_bin: Bin = self.get_closest_available_bin(agent)
        return np.linalg.norm(closest_bin.position - agent.position) if closest_bin else float("inf")

    def get_closest_available_bin(self, agent: Agent) -> Bin:
        """Find the closest available bin to a given agent."""
        available_bins = (bin for bin in self.bins if bin.available)
        return min(available_bins, key=lambda bin: np.sum((bin.position - agent.position) ** 2), default=None)
            
    def closer_to_target(self, target_pos: np.ndarray) -> Tuple[bool, float]:
        """Check if the agent is closer to the target after moving.
        Returns a tuple (bool, float) indicating whether the agent is closer and by how much.
        """
        prev_distance = np.linalg.norm(self.active_agent.position - target_pos)
        distance = np.linalg.norm(self.active_agent.new_position - target_pos)
        distance_change = prev_distance - distance
        return distance < prev_distance, distance_change
    
    def correct_direction(self, target_pos: np.ndarray) -> Tuple[bool, float]:
        """Check if the agent is moving in the correct direction.
        Returns a tuple (bool, float) indicating whether the agent is moving in the correct direction and by how much."""
        target_direction = target_pos - self.active_agent.position
        movement_direction = self.active_agent.new_position - self.active_agent.position
        direction_similarity = np.dot(movement_direction, target_direction) / (
            np.linalg.norm(movement_direction) * np.linalg.norm(target_direction)
        )
        return np.linalg.norm(movement_direction) > 0 and np.linalg.norm(target_direction) > 0, direction_similarity


    def in_entry_zone(self) -> bool:
        """Check if the agent is in the entry zone of the target bin."""
        if not self.target_bin:
            return False
        bin_entry_zone = self.target_bin.get_bin_entry_zone(self.table)
        x, y, width, height = bin_entry_zone
        return (
            x <= self.active_agent.new_position[0] <= x + width
            and y <= self.active_agent.new_position[1] <= y + height
        )
    
    def in_target_bin(self) -> bool:
        """Check if the agent is fully inside the target bin."""
        if not self.target_bin:
            return False
        agent_min, agent_max = self.active_agent.get_boundaries()
        bin_min, bin_max = self.target_bin.get_boundaries()
        return np.all(agent_min >= bin_min) and np.all(agent_max <= bin_max)

    def collision(self, agent: Agent) -> bool:
        """
        Check for collisions between agents.
        """
        if agent is not self.active_agent and not agent.is_done:
            distance_to_agent = np.linalg.norm(self.active_agent.new_position - agent.position)
            return distance_to_agent < self.collision_threshold, distance_to_agent
        return False, -1
    
    def task_completed(self) -> bool:
        """Check if all agents have completed their tasks."""
        return all(agent.is_done for agent in self.agents)
    
    def agent_out_of_table(self) -> bool:
        """Check if the agent is out of the table."""
        return not self.table.contains(self.active_agent.new_position)
    def process_agent_in_bin(self) -> None:
        """Process the agent when it reaches the target bin."""
        self.active_agent.is_done = True
        delay = np.random.random_integers(1, 5)
        self.target_bin.temporarily_disable(delay, self.active_agent)
        self.active_agent = None
        self.target_bin = None
        self.entry_zone = False
        
    def render(self) -> None:
        if self.visualization:
            self.visualization.render(self)

    def close_visualization(self) -> None:
        if self.visualization:
            self.visualization.close()


    