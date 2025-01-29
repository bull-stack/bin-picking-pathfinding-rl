from scene.visualization.renderer_pygame import RendererPyGame
from scene.visualization.renderer_opengl2d import RendererOpenGL2D
from scene.visualization.renderer_opengl3d import RendererOpenGL3D
from stable_baselines3 import PPO, TD3, A2C, DQN, DDPG, SAC
from scene.simulator import Simulator
from env.bin_picking_path_finding_envS import BinPickingPathFindingEnv
def visualization_type(type: str="pygame"):
    # Check which visualization type is chosen and initialize the corresponding renderer
    if type == "pygame":
        # Initialize Pygame renderer
        renderer = RendererPyGame(
            screen_width=800,
            screen_height=600,
            colors={
                "background": (30, 30, 30),
                "table": (50, 50, 150),
                "agent": (0, 255, 0),
                "bin": (255, 0, 0),
                "bin_reached": (255, 255, 0),
                "arrow": (255, 255, 255),
                "path": (255, 0, 255),
            }
        )
    
    elif type == "opengl_2d":
        # Initialize OpenGL 2D renderer
        renderer = RendererOpenGL2D(
            screen_width=800,
            screen_height=600,
            colors={
                "background": (0.1, 0.1, 0.1),
                "table": (0.5, 0.5, 0.9),
                "agent": (0.0, 1.0, 0.0),
                "bin": (1.0, 0.0, 0.0),
                "bin_reached": (1.0, 1.0, 0.0),
                "arrow": (1.0, 1.0, 1.0),
                "path": (1.0, 0.0, 1.0),
            }
        )
    
    elif type == "opengl_3d":
        # Initialize OpenGL 3D renderer
        renderer = RendererOpenGL3D(
            screen_width=800,
            screen_height=600,
            colors={
                "background": (0.1, 0.1, 0.1),
                "table": (0.5, 0.5, 0.9),
                "agent": (0.0, 1.0, 0.0),
                "bin": (1.0, 0.0, 0.0),
                "bin_reached": (1.0, 1.0, 0.0),
                "arrow": (1.0, 1.0, 1.0),
                "path": (1.0, 0.0, 1.0),
            }
        )
    
    else:
        raise ValueError(f"Unknown visualization type: {type}")

    # Return the selected renderer
    return renderer

def get_model(model_name: str):
    """
    Maps a string model_name to the corresponding model class and initializes it.
    """
    model_mapping = {
        "PPO": PPO,
        "TD3": TD3,
        "A2C": A2C,
        "DQN": DQN,
        "DDPG": DDPG,
        "SAC": SAC,
    }
    if model_name not in model_mapping:
        raise ValueError(f"Model '{model_name}' not recognized. Choose from {list(model_mapping.keys())}.")
    
    return model_mapping[model_name]



def init_environment(num_agents: int):
    # Initialize the environment
    renderer = visualization_type("pygame")
    simulator = Simulator(num_agents=num_agents, visualization=renderer)
    return BinPickingPathFindingEnv(simulator=simulator)

print(init_environment(4).action_space.sample())