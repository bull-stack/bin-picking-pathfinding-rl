import optuna
import yaml
import numpy as np

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.noise import NormalActionNoise

from utils import init_environment, get_model, get_parameters
# Define the hyperparameter tuning objective function

def create_model_params(trial, model_name, n_steps=256, n_envs=1):
    """Generate model-specific hyperparameters."""    
    # Add model-specific params
    # Common hyperparameters for all algorithms
    total_rollout_size = n_steps * n_envs  # This must be divisible by batch_size

    # Find valid batch sizes that are divisors of total_rollout_size
    valid_batch_sizes = [bs for bs in range(16, total_rollout_size + 1) if total_rollout_size % bs == 0]
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "gamma": trial.suggest_float("gamma", 0.8, 0.99),
        "batch_size": trial.suggest_categorical("batch_size", valid_batch_sizes),
        "policy_kwargs": {
            "net_arch": [trial.suggest_int("policy_layer1", 64, 512, step=64),
                         trial.suggest_int("policy_layer2", 64, 512, step=64)],
        },
    }

    # Specific hyperparameters for each algorithm
    # PPO (Clip Range, Entropy Coefficient, Value Function Coefficient, etc.)
    if model_name == "PPO":
        params.update({
            "n_steps": trial.suggest_int("n_steps", 128, 1024, step=128),
            "ent_coef": trial.suggest_float("ent_coef", 1e-5, 1e-1, log=True),
            "vf_coef": trial.suggest_float("vf_coef", 0.1, 1.0),
            "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 1.0),
            "clip_range": trial.suggest_float("clip_range", 0.1, 0.5),
            "gae_lambda": trial.suggest_float("gae_lambda", 0.8, 1.0),
        })
    # TD3 specific hyperparameters
    elif model_name == "TD3":
        action_noise_std = trial.suggest_float("action_noise_std", 0.1, 0.5)
        params.update({
            "policy_delay": trial.suggest_int("policy_delay", 2, 5),
            "action_noise": trial.suggest_float("action_noise", 0.0, 0.3),
            "learning_starts": trial.suggest_int("learning_starts", 1000, 10000),
            "train_freq": trial.suggest_int("train_freq", 1, 10),
            "buffer_size": trial.suggest_int("buffer_size", 50000, 1000000),
            "tau": trial.suggest_float("tau", 0.001, 0.1),
            "gradient_steps": trial.suggest_int("gradient_steps", 1, 10),
            "target_policy_noise": trial.suggest_float("target_policy_noise", 0.0, 0.2),
            "action_noise": NormalActionNoise(mean=0.0, sigma=np.full(1, action_noise_std)),
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
        })
    # DDPG specific hyperparameters
    elif model_name == "DDPG":
        action_noise_std = trial.suggest_float("action_noise_std", 0.1, 0.5)
        params.update({
            "action_noise": NormalActionNoise(mean=0.0, sigma=np.full(1, action_noise_std)),
            "buffer_size": trial.suggest_int("buffer_size", 50000, 1000000),
            "learning_starts": trial.suggest_int("learning_starts", 1000, 10000),
            "train_freq": trial.suggest_int("train_freq", 1, 10),
            "gradient_steps": trial.suggest_int("gradient_steps", 1, 10),
            "tau": trial.suggest_float("tau", 0.001, 0.1),
        })
    # A2C specific hyperparameters
    elif model_name == "A2C":
        params.update({
            "n_steps": trial.suggest_int("n_steps", 5, 20, step=5),
            "gae_lambda": trial.suggest_float("gae_lambda", 0.8, 1.0),
            "ent_coef": trial.suggest_float("ent_coef", 1e-5, 1e-1, log=True),
            "vf_coef": trial.suggest_float("vf_coef", 0.1, 1.0),
            "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 1.0),
        })

    # Return the specific model parameters
    return params

def save_best_hyperparameters(model_name, best_params):
    """Save best hyperparameters to a YAML file."""
    try:
        # Load existing hyperparameters if the file exists
        with open("best_hparams.yaml", "r") as file:
            all_best_params = yaml.safe_load(file) or {}
    except FileNotFoundError:
        all_best_params = {}

    # Update the dictionary with the best parameters for the current model
    all_best_params[model_name] = best_params

    # Save back to YAML
    with open("best_hparams.yaml", "w") as file:
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
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, deterministic=True)
    env.close() #
    return mean_reward

def main():    
    # Run the optimization using Optuna
    global_params, _ = get_parameters()
    model_name = global_params["model_name"]
    n_envs = 4
    n_eval_episodes=20
    
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: optimize_model(trial, global_params, n_envs, n_eval_episodes), n_trials=25)
    
    # Print the best hyperparameters
    print(f"Best hyperparameters for {model_name}:", study.best_params)
    save_best_hyperparameters(model_name, study.best_params)
    
if __name__ == "__main__":
    main()