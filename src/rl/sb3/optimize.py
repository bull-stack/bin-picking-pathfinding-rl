import optuna
from stable_baselines3 import PPO, TD3, A2C, DQN, DDPG, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from utils import init_environment
# Define the hyperparameter tuning objective function
def optimize_model(trial):
    """
    Objective function for optimizing PPO with Optuna.
    """
    hyperparams = {
        "PPO": {
            "n_steps": trial.suggest_int("n_steps", 128, 2048, step=128),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            "gamma": trial.suggest_float("gamma", 0.9, 0.9999, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3),
            "clip_range": trial.suggest_float("clip_range", 0.1, 0.4),
            "gae_lambda": trial.suggest_float("gae_lambda", 0.8, 1.0, step=0.05),
            "vf_coef": trial.suggest_float('vf_coef', 0.1, 1.0),
            "ent_coef": trial.suggest_float("ent_coef", 1e-8, 1e-2),
            "max_grad_norm": trial.suggest_float('max_grad_norm', 0.3, 1.0),
        },
        "TD3": {
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            "gamma": trial.suggest_float("gamma", 0.9, 0.9999, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3),
            "buffer_size": trial.suggest_int("buffer_size", 10000, 1000000, step=10000),
            "tau": trial.suggest_float("tau", 0.005, 0.02),
            "train_freq": trial.suggest_int("train_freq", 1, 10),
            "gradient_steps": trial.suggest_int("gradient_steps", 1, 10),
        },
        "SAC": {
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            "gamma": trial.suggest_float("gamma", 0.9, 0.9999, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3),
            "buffer_size": trial.suggest_int("buffer_size", 10000, 1000000, step=10000),
            "tau": trial.suggest_float("tau", 0.005, 0.02),
            "train_freq": trial.suggest_int("train_freq", 1, 10),
            "gradient_steps": trial.suggest_int("gradient_steps", 1, 10),
            "ent_coef": trial.suggest_float("ent_coef", 1e-8, 1e-2),
        },
        "DDPG": {
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            "gamma": trial.suggest_float("gamma", 0.9, 0.9999, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3),
            "buffer_size": trial.suggest_int("buffer_size", 10000, 1000000, step=10000),
            "tau": trial.suggest_float("tau", 0.005, 0.02),
            "train_freq": trial.suggest_int("train_freq", 1, 10),
            "gradient_steps": trial.suggest_int("gradient_steps", 1, 10),
        }
    }
    # Define the simulator
    env = init_environment(num_agents=4)
    env = make_vec_env(lambda: env, n_envs=1)  # Vectorized environment for Stable-Baselines3

    
    # Create the PPO model with sampled hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        **hyperparams,
        device="cpu",  # Use CPU for training
    )

    # Train the model
    model.learn(total_timesteps=400000)
    # Evaluate the trained model
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5, deterministic=True)
    return mean_reward

def main():
    selected_model = "PPO"
    # Define hyperparameters to optimize
    
    # Run the optimization using Optuna
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: optimize_model(trial, model_name=selected_model), n_trials=50)

    # Print the best hyperparameters
    print(f"Best hyperparameters for {selected_model}:", study.best_params)

if __name__ == "__main__":
    main()