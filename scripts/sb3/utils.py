from visualization import RendererPyGame
from visualization import RendererOpenGL2D
from visualization import RendererOpenGL3D

from env import BinPickingPathFindingEnv
from scene import Simulator
from config import FILE_DIR, GLOBALPARAMS_FILE, HYPERPARAMS_FILE
from stable_baselines3 import PPO, TD3, A2C, DDPG, SAC

import yaml
def visualization_type(type: str):
    # Check which visualization type is chosen and initialize the corresponding renderer
    colors={
        "background": (30, 30, 30),
        "table": (50, 50, 150),
        "agent": (0, 255, 0),
        "bin": (255, 0, 0),
        "agent_done": (0, 255, 255),
        "bin_available": (44, 255, 5),
        "arrow": (255, 255, 255),
        "path": (255, 0, 255),
    }
    
    # normalize colors
    normalized_colors = {color: tuple(c / 255 for c in color) for color in colors.values()}

    # Initialize the corresponding renderer based on the chosen visualization type
    if type == "pygame":
        # Initialize Pygame renderer
        renderer = RendererPyGame(
            screen_width=800,
            screen_height=600,
            colors=colors
        )
    
    elif type == "opengl_2d":
        # Initialize OpenGL 2D renderer
        renderer = RendererOpenGL2D(
            screen_width=800,
            screen_height=600,
            colors=normalized_colors
        )
    elif type == "opengl_3d":
        # Initialize OpenGL 3D renderer
        renderer = RendererOpenGL3D(
            screen_width=800,
            screen_height=600,
            colors=normalized_colors
        )
    else:
        raise ValueError(f"Unknown visualization type: {type}")
    # Return the selected renderer
    return renderer

def init_environment(num_agents: int, renderer_type: str, time_limit: float):
    # Initialize the environment
    renderer = visualization_type(renderer_type)
    simulator = Simulator(num_agents=num_agents, visualization=renderer)
    env = BinPickingPathFindingEnv(simulator=simulator, time_limit=time_limit)
    return env

def get_model(model_name: str):
    """Maps a string model_name to the corresponding model class and initializes it."""
    model_mapping = {
        "PPO": PPO,
        "TD3": TD3,
        "A2C": A2C,
        "DDPG": DDPG,
        "SAC": SAC,
    }
    if model_name not in model_mapping:
        raise ValueError(f"Model '{model_name}' not recognized. Choose from {list(model_mapping.keys())}.")
    
    # Initialize the model using the loaded config
    return model_mapping[model_name]

def load_parameters(yaml_file):
    with open(yaml_file, 'r') as f:
        return yaml.safe_load(f)

def get_parameters():    
    hyper_param_file = f"{FILE_DIR}/{HYPERPARAMS_FILE}"
    global_param_file = f"{FILE_DIR}/{GLOBALPARAMS_FILE}"
    # Extract parameters from the YAML file
    globalparams = load_parameters(global_param_file)
    hyperparams = load_parameters(hyper_param_file)
    return globalparams, hyperparams