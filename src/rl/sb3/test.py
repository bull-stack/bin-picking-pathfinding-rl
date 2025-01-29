import matplotlib.pyplot as plt
from utils import init_environment, get_model

def load_trained_model(model_name, model_path):
    """
    Load the trained model from the given path.
    """
    model_class = get_model(model_name)
    model = model_class.load(model_path, print_system_info=True)  # Load the specified model
    return model

def evaluate_episode(model, env, render=True):
    """
    Evaluate the model for a single episode and return the total reward.
    """
    obs, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        if render:
            env.render()  # Render the environment
        total_reward += reward
    return total_reward

def plot_rewards(total_rewards, model_name):
    """
    Plot the total rewards for each episode.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(total_rewards, label='Total Rewards')
    plt.axhline(y=sum(total_rewards) / len(total_rewards), color='r', linestyle='--', label='Average Reward')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(f'Total Rewards per Episode ({model_name})')
    plt.legend()
    plt.grid()
    plt.show()

def evaluate_model(model_name, model_path, num_episodes=50, num_agents=4, render=True):
    """
    Evaluate a trained RL model on the environment.
    """
    env = init_environment(num_agents=num_agents)  # Create environment
    model = load_trained_model(model_name, model_path)  # Load the trained model

    total_rewards = []  # Store rewards for each episode

    # Evaluate the model over multiple episodes
    for episode in range(num_episodes):
        total_reward = evaluate_episode(model, env, render)
        total_rewards.append(total_reward)  # Record total reward for the episode
        print(f"Total Reward for episode {episode} is {total_reward}")

    env.close()

    # Plot the rewards after evaluation
    plot_rewards(total_rewards, model_name)
    
def main():
    # Specify model details
    model_name = "PPO"  # Choose from: "PPO", "TD3", "A2C", "DQN", "DDPG", "SAC"
    model_path = "train/PPO/best_model_360000"  # Update the path based on your model location
    num_episodes = 50  # Number of episodes to evaluate
    num_agents = 4  # Number of agents in the environment

    # Evaluate the chosen model
    evaluate_model(model_name=model_name, 
                   model_path=model_path, 
                   num_episodes=num_episodes, 
                   num_agents=num_agents, 
                   render=True)
    

if __name__ == '__main__':
    main()
