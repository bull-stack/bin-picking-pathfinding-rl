from env.path_finding_env import PathFindingEnv
def main():
    env = PathFindingEnv()
    num_episodes = 10

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            env.render()
            action = env.action_space.sample()  # Replace with RL algorithm
            state, reward, done, _ = env.step(action)
            total_reward += reward

        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    env.close()

if __name__ == "__main__":
    main()
