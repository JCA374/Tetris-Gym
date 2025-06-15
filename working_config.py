
# Working config.py with manual registration
import gymnasium as gym
from gymnasium.envs.registration import register

# Register the environment
register(
    id="TetrisManual-v0",
    entry_point="tetris_gymnasium.envs.tetris:Tetris",
)

ENV_NAME = "TetrisManual-v0"

def make_env(env_name=None, render_mode=None, **kwargs):
    if env_name is None:
        env_name = ENV_NAME
    return gym.make(env_name, render_mode=render_mode or "rgb_array", **kwargs)
