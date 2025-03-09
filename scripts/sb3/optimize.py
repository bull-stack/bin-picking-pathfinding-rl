import optuna
import yaml
import numpy as np
import os

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from torch import nn 

from utils import init_environment, get_model, get_parameters
from config import FILE_DIR, HYPERPARAMS_FILE
# Define the hyperparameter tuning objective function

def create_model_params(trial, model_name, n_steps=256, n_envs=1):
    """Generate model-specific hyperparameters."""    
    total_rollout_size = n_steps * n_envs  # This must be divisible by batch_size

    # Find valid batch sizes that are divisors of total_rollout_size
    valid_batch_sizes = [bs for bs in range(16, total_rollout_size + 1) if total_rollout_size % bs == 0]
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "gamma": trial.suggest_float("gamma", 0.9, 0.99),
        
        # "policy_kwargs": {
        #     "activation_fn": nn.ELU,
        #     "net_arch": [
        #         32, 32,  # Shared layers
        #         {
        #             "pi": [256, 128, 64],  # Policy network
        #             "vf": [256, 128, 64],  # Value function network
        #         }
        #     ],
        #     },
    }
    # action_noise_std = trial.suggest_float("action_noise_std", 0.1, 0.5)
    # Specific hyperparameters for each algorithm
    # PPO (Clip Range, Entropy Coefficient, Value Function Coefficient, etc.)
    if model_name == "PPO":
        params.update({
            "n_steps": trial.suggest_int("n_steps", 512, 4096, step=512),
            "ent_coef": trial.suggest_float("ent_coef", 1e-5, 1e-1, log=True),
            "vf_coef": trial.suggest_float("vf_coef", 0.1, 1.0),
            "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 1.0),
            "clip_range": trial.suggest_float("clip_range", 0.1, 0.5),
            "gae_lambda": trial.suggest_float("gae_lambda", 0.8, 1.0),
            "n_epochs": trial.suggest_int("n_epochs", 1, 10),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            })
    # TD3 specific hyperparameters
    elif model_name == "TD3":    
        params.update({
            "policy_delay": trial.suggest_int("policy_delay", 2, 5),
            "train_freq": trial.suggest_int("train_freq", 1, 10),
            "buffer_size": trial.suggest_int("buffer_size", 50000, 1000000),
            "tau": trial.suggest_float("tau", 0.001, 0.1),
            "gradient_steps": trial.suggest_int("gradient_steps", 1, 10),
            "target_policy_noise": trial.suggest_float("target_policy_noise", 0.0, 0.2),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        })
    # SAC specific hyperparameters
    elif model_name == "SAC":
        params.update({
            "target_entropy": trial.suggest_float("target_entropy", -5.0, -0.1),
            "learning_starts": trial.suggest_int("learning_starts", 1000, 10000),
            "train_freq": trial.suggest_int("train_freq", 1, 10),
            "gradient_steps": trial.suggest_int("gradient_steps", 1, 10),
            "tau": trial.suggest_float("tau", 0.001, 0.1),
            "ent_coef": trial.suggest_float("ent_coef", 1e-5, 1e-1, log=True),
            "buffer_size": trial.suggest_int("buffer_size", 50000, 1000000),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        })
    # DDPG specific hyperparameters
    elif model_name == "DDPG":
        params.update({
            # "action_noise": NormalActionNoise(mean=0.0, sigma=np.full(1, action_noise_std)),
            "buffer_size": trial.suggest_int("buffer_size", 50000, 1000000),
            "train_freq": trial.suggest_int("train_freq", 1, 10),
            "gradient_steps": trial.suggest_int("gradient_steps", 1, 10),
            "tau": trial.suggest_float("tau", 0.001, 0.1),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        })
    # A2C specific hyperparameters
    elif model_name == "A2C":
        params.update({
            "n_steps": trial.suggest_int("n_steps", 128, 2048, step=128),
            "gae_lambda": trial.suggest_float("gae_lambda", 0.8, 1.0),
            "ent_coef": trial.suggest_float("ent_coef", 1e-5, 1e-1, log=True),
            "vf_coef": trial.suggest_float("vf_coef", 0.1, 1.0),
            "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 1.0),
        })

    # Return the specific model parameters
    return params

def save_best_hyperparameters(save_path, model_name, best_params):
    """Save best hyperparameters to a YAML file."""
    try:
        # Load existing hyperparameters if the file exists
        with open(save_path, "r") as file:
            all_best_params = yaml.safe_load(file) or {}
    except FileNotFoundError:
        all_best_params = {}
    
    # Update the dictionary with the best parameters for the current model
    all_best_params[model_name] = best_params

    # Save back to YAML
    with open(save_path, "w") as file:
        yaml.dump(all_best_params, file, default_flow_style=False)

    print(f"Best hyperparameters for {model_name} saved to best_params.yaml")
    
def make_env(num_agents, renderer_type, time_limit):
    """
    Creates and initializes an environment with dynamic parameters.
    """
    return init_environment(num_agents=num_agents, renderer_type=renderer_type, time_limit=time_limit)
def optimize_model(trial, global_params, n_envs, n_eval_episodes):
    """
    Objective function for optimizing reinforcement learning models with Optuna.
    """
    renderer_type = global_params["renderer_type"]
    model_name = global_params["model_name"]
    num_agents = global_params["num_agents"]
    time_limit = global_params["time_limit"]
    time_steps = global_params["time_steps"]
    
    hyperparams = create_model_params(trial, model_name)
    print(f"Trial {trial.number}: {hyperparams}")
    
    def make_monitored_env(num_agents, renderer_type, time_limit):
        env = make_env(num_agents, renderer_type, time_limit)
        return Monitor(env)  # Wrap with Monitor
    # Set up SubprocVecEnv with start_method="fork"
    env = SubprocVecEnv([lambda: make_env(num_agents=num_agents, renderer_type=renderer_type, time_limit=time_limit) 
                         for _ in range(n_envs)], 
                         start_method='fork')
    
    # env = make_vec_env([lambda: env for _ in range(n_envs)], n_envs=n_envs, vec_env_cls=SubprocVecEnv)  # Vectorized environment for Stable-Baselines3
    
    model_class = get_model(model_name)
    model = model_class(
        "MlpPolicy",
        env, 
        verbose=0, 
        **hyperparams,
        device="cpu",
    )

    # Train the model
    model.learn(total_timesteps=time_steps)
    # Evaluate the trained model
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, deterministic=True)
    if trial.should_prune():
        raise optuna.TrialPruned()
    env.close() #
    return mean_reward

def main():    

    save_path = os.path.join(FILE_DIR, HYPERPARAMS_FILE)
    
    global_params, _ = get_parameters()
    model_name = global_params["model_name"]
    n_envs = 1
    n_eval_episodes=5
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: optimize_model(trial, global_params, n_envs, n_eval_episodes), n_trials=25)
    
    # Print the best hyperparameters
    print(f"Best hyperparameters for {model_name}:", study.best_params)
    save_best_hyperparameters(save_path, model_name, study.best_params)
    
if __name__ == "__main__":
    main()