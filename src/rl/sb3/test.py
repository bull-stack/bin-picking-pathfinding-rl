from stable_baselines3 import PPO, TD3, DDPG, A2C
from env.path_finding_env_multi import BinpickingPathFindingEnvM
from env.path_finding_env_single import BinpickingPathFindingEnvS
import matplotlib.pyplot as plt

def main():
    env = BinpickingPathFindingEnvS() 
    # Load PPO model
    model = PPO.load("train/best_model_2000000", print_system_info=True)
    
    total_rewards = []  # To store rewards for each episode

    # Test the model
    for episode in range(1000): 
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done: 
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            env.render()  # Render the environment
            total_reward += reward
        total_rewards.append(total_reward)  # Record total reward for the episode
        print('Total Reward for episode {} is {}'.format(episode, total_reward))

    env.close()

    # Plot the rewards
    plt.figure(figsize=(10, 6))
    plt.plot(total_rewards, label='Total Rewards')
    plt.axhline(y=sum(total_rewards) / len(total_rewards), color='r', linestyle='--', label='Average Reward')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Rewards per Episode')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == '__main__':
    main()