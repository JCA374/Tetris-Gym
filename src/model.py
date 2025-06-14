import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, obs_space, action_space):
        super().__init__()
        # define conv layers, FC layers...

    def forward(self, x):
        # return Q-values
