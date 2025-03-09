import matplotlib.pyplot as plt
import numpy as np

from utils import init_environment, get_model, get_parameters

def load_trained_model(model_name, model_path):
    model_class = get_model(model_name)
    model = model_class.load(model_path, print_system_info=True)
    return model

def evaluate_episode(model, env, render=True):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        if render:
            env.render()
        total_reward += reward
    return total_reward

def plot_rewards(total_rewards, model_name):
    avg_reward = np.mean(total_rewards)
    max_reward = np.max(total_rewards)
    min_reward = np.min(total_rewards)
    std_dev = np.std(total_rewards)
    
    plt.figure(figsize=(12, 6))
    plt.plot(total_rewards, label='Total Rewards', marker='o', linestyle='-')
    plt.axhline(y=avg_reward, color='r', linestyle='--', label=f'Average Reward: {avg_reward:.2f}')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(f'Total Rewards per Episode ({model_name})')
    plt.legend()
    plt.grid()
    plt.show()
    
    print("\nEvaluation Metrics:")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Max Reward: {max_reward:.2f}")
    print(f"Min Reward: {min_reward:.2f}")
    print(f"Standard Deviation: {std_dev:.2f}")
    
    plt.figure(figsize=(8, 5))
    plt.hist(total_rewards, bins=10, edgecolor='black', alpha=0.7)
    plt.axvline(avg_reward, color='r', linestyle='--', label='Mean')
    plt.xlabel('Total Reward')
    plt.ylabel('Frequency')
    plt.title('Reward Distribution')
    plt.legend()
    plt.show()

def evaluate_model(model_name, model_path, renderer_type, time_limit, num_episodes=50, num_agents=4, render=True):
    env = init_environment(num_agents=num_agents, renderer_type=renderer_type, time_limit=time_limit)
    model = load_trained_model(model_name, model_path)
    total_rewards = []

    for episode in range(num_episodes):
        total_reward = evaluate_episode(model, env, render)
        total_rewards.append(total_reward)
        print(f"Episode {episode + 1}: Reward = {total_reward}")

    env.close()
    plot_rewards(total_rewards, model_name)

def main():
    global_params, _ = get_parameters()
    renderer_type = global_params["renderer_type"]
    model_name = global_params["model_name"]
    num_agents = global_params["num_agents"]
    time_limit = global_params["time_limit"]
    time_steps = global_params["time_steps"]

    num_episodes = 50  
    model_path = f"train/{model_name}/best_{num_agents}_{time_steps}"

    evaluate_model(model_name=model_name, 
                   model_path=model_path, 
                   renderer_type=renderer_type,
                   time_limit=time_limit,
                   num_episodes=num_episodes, 
                   num_agents=num_agents, 
                   render=True)

if __name__ == '__main__':
    main()
