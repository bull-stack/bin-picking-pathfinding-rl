

from stable_baselines3 import PPO, TD3, A2C, DQN, DDPG, SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from env.path_finding_env_multi import BinpickingPathFindingEnvM
from env.path_finding_env_single import BinpickingPathFindingEnvS
import pandas as pd
import os
    
class MetricsLoggerCallback(BaseCallback):
    def __init__(self, filename, check_freq, save_path, verbose=1):
        super(MetricsLoggerCallback, self).__init__(verbose)
        self.filename = filename
        self.rewards = []
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
        return True  
    
    def _on_training_end(self) -> None:
        # Flatten rewards and losses lists
        rewards_flat = [reward.item() for reward_list in self.rewards for reward in reward_list]

        # Save to CSV
        df = pd.DataFrame({
            'Reward': rewards_flat,
        })
        df.to_csv(self.filename, index=False)  
        
CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'       

def main():
    # Initialize the environment
    env = BinpickingPathFindingEnvS()
    # Check the environment
    check_env(env, warn=True)
    # Initialize callback
    callback = MetricsLoggerCallback('rewards.csv', check_freq=10000, save_path=CHECKPOINT_DIR)
    # Initialize PPO model
    model = PPO("MlpPolicy", env, tensorboard_log=LOG_DIR, verbose=1, learning_rate=0.0001, device='cpu')
    # Train the model
    model.learn(total_timesteps=2000000, callback=callback)
    
if __name__ == '__main__':
    main()