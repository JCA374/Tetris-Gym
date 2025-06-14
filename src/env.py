import gym
import gym_tetris
import nes_py


def make_env(name):
    env = gym.make(name)
    # apply wrappers: resize, grayscale, frame-stack, etc.
    return env
