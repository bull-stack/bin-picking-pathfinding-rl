from setuptools import setup, find_packages

setup(
    name="bin_picking_rl",
    version="0.1.0",
    description="A reinforcement learning environment for bin picking tasks",
    author="Andreas Bull Enger",
    author_email="abullenger88@gmail.com",
    url="https://github.com/bull-stack/bin-picking-pathfinding-rl",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "gymnasium",
        "stable-baselines3",
        "optuna",
        "pyyaml",
        "torch",
        "torchvision",
        "torchaudio",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10.15',
)