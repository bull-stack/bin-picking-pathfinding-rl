from gymnasium.envs.registration import register

register(
    id="PathFindingEnv-v0",
    entry_point="path_finding_env:PathFindingEnv",  # Update with your module and class path
)