from scene.simulator import Simulator
from gymnasium import Env, spaces

import numpy as np

class BinPickingPathFindingEnv(Env):
    def __init__(self, simulator: Simulator, time_limit: int):
        super().__init__()
        self.simulator: Simulator = simulator

        # Define the observation space and action space
        # self.observation_space = spaces.Box(
        #     low=0, high=max(self.simulator.screen_width, self.simulator.screen_height),
        #     shape=(2 * self.simulator.num_agents + 2 * 4,), dtype=np.float32
        # )
        self.observation_space: spaces.Box = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(7 + 2 * (self.simulator.num_agents - 1),),  # 6 for the single-agent state + 2D positions for other agents
            dtype=np.float32
        )
        self.action_space: spaces.Box = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        self.time_limit: int = time_limit
        self.reset()
        
    # def get_state(self) -> np.ndarray:
    #     agent_state = np.concatenate([agent.position for agent in self.agents])
    #     bin_state = np.concatenate([bin.position for bin in self.bins])
    #     return np.concatenate([agent_state, bin_state]).astype(np.float32)

    def get_state(self) -> np.ndarray:
        state = []
        if self.simulator.target_bin and self.simulator.active_agent:
            relative_position = self.simulator.get_target_pos() - self.simulator.active_agent.position
            relative_position = relative_position / np.array([self.simulator.table_width, self.simulator.table_height], dtype=np.float32)
            state.extend(relative_position)
            
            normalized_agent_position = self.simulator.active_agent.position / np.array([self.simulator.table_width, self.simulator.table_height], dtype=np.float32)
            state.extend(normalized_agent_position)
            # Distance to the target bin entry zone (normalized)
            distance_to_bin = np.linalg.norm(relative_position) / np.linalg.norm([self.simulator.table_width, self.simulator.table_height])
            state.append(distance_to_bin)
        else:
            # Default values if no target bin is assigned
            state.extend([0, 0])  # Relative position
            state.append(1.0)  # Distance to the bin (max 1.0)

        # 3. **Time Remaining and Elapsed (normalized)**:
        time_remaining = (self.time_limit - self.time_steps) / self.time_limit
        state.append(time_remaining)
        elapsed_time = self.time_steps / self.time_limit
        state.append(elapsed_time)

        # 4. **Standardized Positions of Other Agents (fixed order)**:
        # Sort agents by ID or use a fixed order
        
        if self.simulator.active_agent:
            agents_to_process = [agent for agent in self.simulator.agents if agent != self.simulator.active_agent]
        else:
            agents_to_process = self.simulator.agents

        for agent in agents_to_process:
            normalized_position = agent.position / np.array([self.simulator.table_width, self.simulator.table_height], dtype=np.float32)
            state.extend(normalized_position)
            
        return np.array(state, dtype=np.float32)
    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.time_steps: int = 0
        self.simulator.reset()
        return self.get_state(), {}

    def step(self, action):
        reward = 0.0
        done = False
        truncated = False
        
        reward -= 4.0
        self.time_steps += 1
        
        if self.simulator.task_completed():
            reward += 50.0  # Bonus for completing all tasks
            reward += (self.time_limit - self.time_steps) * 5
            return self.get_state(), reward, True, truncated, {}
        if self.simulator.active_agent is None:
            return self.get_state(), reward, True, truncated, {}
        if self.is_timeout():
            reward -= 50  # Stronger penalty for leaving the table
            return self.get_state(), reward, True, True, {}
        
        self.simulator.action(action)
        reward += self.simulator.evaluate_move()  # Increase reward for correct movement
        if not self.simulator.agent_out_of_table():
            self.simulator.active_agent.update_position()
        elif self.simulator.in_entry_zone():
            reward += 15.0  # Increased reward for reaching entry zone
            self.simulator.entry_zone = True     
            self.simulator.active_agent.update_position()       
            # Check if the agent is in the target bin
            if self.simulator.in_target_bin():
                reward += 20.0  # Reward for reaching the target bin
                self.simulator.process_agent_in_bin()
                self.simulator.select_active_agent()
                self.simulator.select_target_bin()
        else:
            reward -= 50.0  # Stronger penalty for leaving the table
            return self.get_state(), reward, True, truncated, {}
        return self.get_state(), reward, done, truncated, {}

    def is_timeout(self) -> bool:
        return self.time_steps >= self.time_limit
    
    def render(self, mode="human"):
        self.simulator.render()

    def close(self):
        self.simulator.close_visualization()

