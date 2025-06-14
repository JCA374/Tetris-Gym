import random
from collections import deque


class Agent:
    def __init__(self, obs_space, action_space):
        # init nets, optimizer, replay buffer...

    def select_action(self, state, eval_mode=False):
        # ε-greedy or deterministic

    def remember(self, *transition):
        # push to replay buffer

    def learn(self):
        # sample batch, compute loss, backprop

    def save_checkpoint(self, episode):
        # torch.save(...)

    def load_checkpoint(self, latest=False, path=None):
        # load weights
