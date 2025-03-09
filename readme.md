# Bin Picking RL

## Description

`bin_picking_pathfinding-rl` is a reinforcement learning environment designed for simulating bin-picking tasks. The environment is built using the OpenAI Gym framework and supports training reinforcement learning agents to solve tasks related to object dragged towards a target bin. Trying to find an optimal path for the agent to take towards its designated bin. The scene is setup with a table where 4 bins are placed 2 on top and 2 on the bottom. There are 1 or more objects on the table that needs to be moved towards its closest bin. These agents do not move simultaneously but in sequence, when 1 agent has arrived at its target, the next agent is finding a new path to its designated target bin so on and so forth. An episode is completed when all agent on the table has successfully made it to their target bins.

## Prerequisites

Make sure you have **conda** installed. You can get it from [Anaconda](https://www.anaconda.com/products/individual).

## Setting Up the Environment

To set up the environment, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/bull-stack/bin-picking-pathfinding-rl.git
   cd bin-picking-pathfinding-rl
    ```


2. Create the conda environment from the environment.yml file:
    ```bash
    conda env create -f environment.yml
    conda activate bin_picking_env
    ```

The environment includes the following dependencies:

    Python 3.10.15
    numpy
    matplotlib
    gymnasium
    stable-baselines3
    optuna
    pyyaml
    torch, torchvision, torchaudio (installed via pip)

### 1. Configuration

The model and environment configurations are defined by a set of parameters in the `global_params` and `hyperparams` dictionaries. Some important parameters include:

- **model_name**: The name of the model (e.g., `PPO`, `DDPG`).
- **num_agents**: The number of agents in the environment.
- **time_limit**: The maximum time each agent is allowed to interact with the environment.
- **time_steps**: The total number of training steps (timesteps) the model should run for.
- **log_dir**: The directory to store the training logs and tensorboard data.
- **save_path**: The path where the trained model will be saved.

You can modify these parameters to suit your needs.

### 2. Running the Training

Once the environment is set up and the necessary configurations are in place, you can train the model by running the `train.py` script. Use the following command to execute the script:

```bash
python scripts/sb3/train.py
```
### 3. Training the Model

The script will initialize the environment and the model with the specified configurations.
The MetricsLoggerCallback is used to log training metrics (e.g., rewards, loss) to the console and to a CSV file.
The model is trained using the specified algorithm (e.g., PPO, DQN) with the training process running for the specified number of time steps (time_steps).
TensorBoard logs will be generated in the log_dir to track the progress of the training process visually.

### 5. Model Saving

During training, the best model will be saved at regular intervals (based on check_freq). After training completes, the model will be saved to the save_path.

# Testing the Trained Model

This section explains how to test a trained reinforcement learning model on your environment and visualize its performance.

## Prerequisites

Ensure that you have set up the environment and trained a model as described in the [Training the Model](#training-the-model) section.

### 1. Configuration

The testing script uses the following parameters:

- **model_name**: The name of the trained model (e.g., `PPO`, `DQN`).
- **num_agents**: The number of agents in the environment.
- **time_limit**: The maximum time each agent is allowed to interact with the environment during an episode.
- **num_episodes**: The number of episodes to test the model on.
- **renderer_type**: The type of rendering (for visualizing the environment).

You can modify these parameters in the `global_params` dictionary.

### 2. Running the Test

Once you have a trained model and the appropriate configuration, you can run the `test.py` script to evaluate the performance of the model.

Use the following command to run the test:

```bash
python scripts/sb3/test.py
```

# Hyperparameter Optimization with Optuna

This section explains how to perform hyperparameter optimization for reinforcement learning models using Optuna. It automates the process of finding the best hyperparameters for your RL model, based on the performance of the model during training and evaluation.

## Prerequisites

Before running this optimization script, ensure that you have the following prerequisites:

1. **Stable-Baselines3**: This script utilizes the Stable-Baselines3 library to define and train RL models. Ensure you have this installed.

2. **Optuna**: Optuna is used to optimize the hyperparameters. You can install it with:
    ```bash
    pip install optuna
    ```

3. **Model**: You should have a working model defined in the `utils.py` script and the corresponding model class available for training.

4. **Config and Environment**: You need to have the correct environment configuration and parameters (such as `num_agents`, `renderer_type`, etc.) defined in `config.py` and `utils.py`.

### 1. Configuration

In the script, we use Optuna to automatically search for the best hyperparameters. The optimization function tunes different hyperparameters depending on the model selected (`PPO`, `TD3`, `SAC`, etc.).

The script uses the following parameters:

- **model_name**: The name of the model to optimize (e.g., `PPO`, `TD3`, `SAC`).
- **n_envs**: The number of environments used for training.
- **n_eval_episodes**: The number of evaluation episodes to test the model during optimization.
- **time_steps**: Total number of time steps to train the model.

You can modify these values in the `global_params` dictionary.

### 2. Running the Hyperparameter Optimization

Once your environment and model are properly set up, you can start the optimization process by running the following command:

```bash
python scripts/sb3/optimize.py