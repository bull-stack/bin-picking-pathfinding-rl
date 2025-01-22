
import sys
for path in sys.path:
    print(path)
from stable_baselines3 import PPO, TD3, A2C, DQN, DDPG, SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from env.path_finding_env_multi import BinpickingPathFindingEnvM
from env.bin_picking_path_finding_envS import BinPickingPathFindingEnv
from env.path_finding_env_single import BinpickingPathFindingEnvS
from scene.simulator import Simulator
import pandas as pd
import os
import numpy as np


class MetricsLoggerCallback(BaseCallback):
    def __init__(self, filename, check_freq, save_path, n_steps, verbose=1):
        super(MetricsLoggerCallback, self).__init__(verbose)
        self.filename = filename
        self.n_steps = n_steps
        self.rewards = []
        self.average_rewards = []
        self.save_path = save_path
        self.check_freq = check_freq
        
    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
            
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)
        self.rewards.append(self.locals["rewards"])
        # Check if it's time to compute and save the average reward
        if len(self.rewards) >= self.n_steps:
            avg_reward = np.mean(self.rewards[-self.n_steps:])
            self.average_rewards.append(avg_reward)
        return True  
    
    def _on_training_end(self) -> None:
        # Flatten rewards and losses lists
        rewards = [reward for reward in self.average_rewards]
        # Save to CSV
        df = pd.DataFrame({
            'Reward': rewards,
        })
        df.to_csv(self.filename, index=False)  
        
CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'       

def main():
    # Initialize the environment
    simulator = Simulator(
        screen_width=800, 
        screen_height=600, 
        table_width=600, 
        table_height=350, 
        time_limit=400,
        step_size=5,  # Size of each step in the environment
        num_agents=1  # Number of agents in the environment
    )
    env = BinPickingPathFindingEnv(simulator)
    # env = BinpickingPathFindingEnvS()
    # Check the environment
    check_env(env, warn=True)
    # Initialize callback
    callback = MetricsLoggerCallback(
        'rewards_test12.csv', 
        check_freq=10000, 
        save_path=CHECKPOINT_DIR, 
        n_steps=10000
    )
    # Initialize PPO model
    model = PPO("MlpPolicy",
                env, 
                tensorboard_log=LOG_DIR, 
                verbose=1, 
                learning_rate=0.0001, 
                device='cpu'
    )
    # Train the model
    model.learn(total_timesteps=1500000, callback=callback)
    
if __name__ == '__main__':
    main()