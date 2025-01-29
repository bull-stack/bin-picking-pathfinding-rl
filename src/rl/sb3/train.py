import os
import numpy as np
import pandas as pd
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from utils import get_model, init_environment

# MetricsLoggerCallback for saving training metrics
class MetricsLoggerCallback(BaseCallback):
    """
    Custom callback for logging training metrics, saving the model, 
    and exporting metrics to a CSV file for further analysis.
    """
    def __init__(self, 
                 filename: str, 
                 check_freq: int, 
                 save_path: str, 
                 n_steps: int, 
                 verbose: int = 1, 
                 model_name: str = "model"):
        super(MetricsLoggerCallback, self).__init__(verbose)
        self.filename = filename
        self.check_freq = check_freq
        self.save_path = save_path
        self.n_steps = n_steps
        self.model_name = model_name
        self.rewards = []
        self.average_rewards = []
        self.episode_lengths = []
        self.metrics_data = []  # Store metrics for logging

    def _init_callback(self) -> None:
        """
        Create necessary directories for saving models and logs.
        """
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        """
        Called at every environment step.
        """
        # Save the model at the specified frequency
        if self.n_calls % self.check_freq == 0:
            model_save_dir = os.path.join(self.save_path, self.model_name)
            os.makedirs(model_save_dir, exist_ok=True)
            model_path = os.path.join(model_save_dir, f"best_model_{self.n_calls}.zip")
            self.model.save(model_path)
            if self.verbose:
                print(f"Model saved at step {self.n_calls} to {model_path}")

        # Collect rewards and episode lengths
        self.rewards.append(self.locals["rewards"][0])  # Collect rewards from the environment
        self.episode_lengths.append(self.locals["dones"][0])  # Track end of episodes

        # Log metrics over the last `n_steps`
        if len(self.rewards) >= self.n_steps:
            avg_reward = np.mean(self.rewards[-self.n_steps:])
            avg_episode_length = np.mean(self.episode_lengths[-self.n_steps:])
            self.average_rewards.append(avg_reward)

            # Store metrics for detailed logging
            self.metrics_data.append({
                "step": self.n_calls,
                "average_reward": avg_reward,
                "average_episode_length": avg_episode_length
            })

            

        return True

    def _on_training_end(self) -> None:
        """
        Called at the end of training to save the final metrics and model.
        """

        # Save metrics to CSV
        metrics_df = pd.DataFrame(self.metrics_data)
        metrics_df.to_csv(self.filename, index=False)
        if self.verbose:
            print(f"Training metrics saved to {self.filename}")


def setup_hyperparameters():
    """
    Returns the hyperparameters based on the model name.
    """
    return {
        "PPO": {
            "n_steps": 1664,
            "batch_size": 64,
            "gamma": 0.9238596445937732,
            "learning_rate": 0.0006491684215919559,
            "clip_range": 0.1618259529960692,
            "gae_lambda": 0.9,
            "ent_coef": 0.000009,
            "vf_coef": 0.0017377429654298819,
            "max_grad_norm": 0.7088234597383188,
            "n_epochs": 10,
        },
        "TD3": {
            "batch_size": 64,
            "gamma": 0.99,
            "learning_rate": 0.0003,
            "buffer_size": 1000000,
            "tau": 0.005,
            "train_freq": 1,
            "gradient_steps": 1,
            "policy_delay": 2,
        },
        "SAC": {
            "batch_size": 64,
            "gamma": 0.99,
            "learning_rate": 0.0003,
            "buffer_size": 1000000,
            "tau": 0.005,
            "train_freq": 1,
            "gradient_steps": 1,
            "ent_coef": "auto",
        },
        "DDPG": {
            "batch_size": 64,
            "gamma": 0.99,
            "learning_rate": 0.0003,
            "buffer_size": 1000000,
            "tau": 0.005,
            "train_freq": 1,
            "gradient_steps": 1,
        }
    }


def setup_environment(num_agents=4):
    """
    Initializes and returns the environment.
    """
    env = init_environment(num_agents=num_agents)
    check_env(env, warn=True)  # Check the environment
    return env


def setup_callback(check_freq, save_path, n_steps, verbose, model_name, filename):
    """
    Initializes and returns the callback for logging metrics.
    """
    
    if model_name is None:
        raise ValueError("model_name is required")
    if filename is None:
        filename = f"model_{model_name}_rewards.csv"
    # Create the MetricsLoggerCallback
    return MetricsLoggerCallback(
        filename=filename, 
        check_freq=check_freq, 
        save_path=save_path, 
        n_steps=n_steps,
        verbose=verbose,
        model_name=model_name,  # Name of the model for saving the best model
    )


def setup_model(model_name, env, hyperparams, log_dir, verbose, device):
    """
    Initializes and returns the model based on the model name and hyperparameters.
    """
    model_class = get_model(model_name)
    model = model_class(
        "MlpPolicy",
        env, 
        tensorboard_log=log_dir, 
        verbose=verbose, 
        **hyperparams[model_name],
        device=device,
    )
    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)
    return model


def train_model(model, total_timesteps=400000, callback=None):
    """
    Trains the model for the specified number of timesteps.
    """
    model.learn(total_timesteps=total_timesteps, callback=callback)


def main():
    """
    Main method that ties everything together and runs the training process.
    """
    # Specify model details
    model_name = "PPO"  # Choose from: "PPO", "TD3", "A2C", "DQN", "DDPG", "SAC"
    num_agents = 4  # Number of agents in the environment
    filename = f"{model_name}_{num_agents}_rewards.csv"
    log_dir = "./logs/"  # Update the path based on your desired log directory
    save_path = "./train/"
    
    # Setup hyperparameters, environment, and callback for logging metrics
    hyperparams = setup_hyperparameters()
    env = setup_environment(num_agents=num_agents)
    callback = setup_callback(check_freq=10000, 
                              save_path=save_path, 
                              n_steps=10000,  
                              verbose=1,  # Set to 0 for no logging
                              model_name=model_name,
                              filename=filename)
    # Setup the model and train it
    model = setup_model(model_name, env, hyperparams, log_dir=log_dir, verbose=1, device='cpu')
    train_model(model, total_timesteps=400000, callback=callback)


if __name__ == '__main__':
    main()
