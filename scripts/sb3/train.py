from stable_baselines3.common.logger import configure, CSVOutputFormat
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.utils import constant_fn

import torch.nn as nn # noqa: F401
from logger import MetricsLoggerCallback
from utils import get_model, init_environment, get_parameters

from typing import Any
import os
import re

def setup_callback(num_agents, check_freq, save_path, n_steps, verbose, model_name):
    """
    Initializes and returns the callback for logging metrics.
    """
    if model_name is None:
        raise ValueError("model_name is required")

    # Create the MetricsLoggerCallback
    return MetricsLoggerCallback(
        num_agents=num_agents,
        check_freq=check_freq, 
        save_path=save_path, 
        n_steps=n_steps,
        verbose=verbose,
        model_name=model_name,  # Name of the model for saving the best model
    )


def setup_model(num_agents, model_name, env, hyperparams, log_dir, verbose, device):
    """
    Initializes and returns the model based on the model name and hyperparameters.
    """
    # hyperparams = update_dict(hyperparams.get(model_name, {}))
    # print(hyperparams)
    model_class = get_model(model_name)
    print(hyperparams.get(model_name, {}))
    model = model_class(
        # policy="MlpPolicy",
        env=env, 
        tensorboard_log=log_dir, 
        verbose=verbose, 
        **hyperparams.get(model_name, {}),
        device=device,
    )
    
    os.makedirs(log_dir, exist_ok=True)
    # Find the highest existing counter
    pattern = rf"{model_name}_{num_agents}_metrics_(\d+)\.csv"
    existing_files = [f for f in os.listdir(log_dir) if re.match(pattern, f)]
    
    if existing_files:
        # Extract only the LAST number before ".csv"
        highest_counter = max(int(re.search(r"_(\d+)\.csv$", f).group(1)) for f in existing_files)
        counter = highest_counter + 1
    else:
        counter = 1

    # Define the new log file name
    logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    logger.output_formats = [CSVOutputFormat(os.path.join(log_dir, f"{model_name}_{num_agents}_metrics_{counter}.csv"))]
    model.set_logger(logger)
    return model

def main():
    """
    Main method that ties everything together and runs the training process.
    """
    global_params, hyperparams = get_parameters()
    renderer_type = global_params["renderer_type"]
    model_name = global_params["model_name"]
    num_agents = global_params["num_agents"]
    time_limit = global_params["time_limit"]
    time_steps = global_params["time_steps"]
    log_dir = global_params["log_dir"]
    save_path = global_params["save_path"]
    # Setup hyperparameters, environment, and callback for logging metrics
    env = init_environment(
        num_agents=num_agents, 
        renderer_type=renderer_type, 
        time_limit=time_limit
    )
    check_env(env, warn=True)
    callback = setup_callback(
        num_agents=num_agents,  # Number of agents in the environment
        check_freq=10000, 
        save_path=save_path, 
        n_steps=10000,  
        verbose=1,  # Set to 0 for no logging
        model_name=model_name
    )
    # Setup the model and train it
    model = setup_model(
        num_agents=num_agents,
        model_name=model_name, 
        env=env, 
        hyperparams=hyperparams, 
        log_dir=log_dir, 
        verbose=1, 
        device='cuda'
    )
    
    model.learn(total_timesteps=time_steps, callback=callback)
    env.close()

if __name__ == '__main__':
    main()
