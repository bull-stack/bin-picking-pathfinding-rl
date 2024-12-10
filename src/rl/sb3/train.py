  
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from env.path_finding_env import PathFindingEnv

vec_env = make_vec_env(PathFindingEnv, n_envs=1, env_kwargs=dict(grid_size=10, cell_size=50))

# Train with Stable-Baselines3
model = DQN("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=10000)

# Test the trained model
obs = vec_env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()

vec_env.close()
