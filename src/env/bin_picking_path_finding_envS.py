from gymnasium import Env, spaces
from scene.simulator import Simulator
import numpy as np


class BinPickingPathFindingEnv(Env):
    def __init__(self, simulator: Simulator):
        super().__init__()
        self.simulator = simulator

        # Define the observation space and action space
        self.observation_space = spaces.Box(
            low=0, high=max(self.simulator.screen_width, self.simulator.screen_height),
            shape=(2 * self.simulator.num_agents + 2 * 4,), dtype=np.float32
        )
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None):
        return self.simulator.reset(seed)

    def step(self, action):
        return self.simulator.action(action)


    def render(self, mode="human"):
        self.simulator.render()

    def close(self):
        self.simulator.renderer.close()

