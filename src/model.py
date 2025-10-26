# src/model.py
import torch
import torch.nn as nn
import numpy as np


def _to_nchw(x: torch.Tensor) -> torch.Tensor:
    """
    Accepterar (B,H,W), (B,H,W,C) eller (B,C,H,W) och returnerar (B,C,H,W).
    Normaliserar till float32 [0,1].
    """
    if x.dim() == 3:                 # (B,H,W)
        x = x.unsqueeze(1)           # -> (B,1,H,W)
    elif x.dim() == 4 and x.shape[-1] in (1, 3):  # (B,H,W,C)
        x = x.permute(0, 3, 1, 2)    # -> (B,C,H,W)
    # annars antar vi redan (B,C,H,W)
    if x.dtype != torch.float32:
        x = x.float()
    # Om dina obs är 0..255
    if x.max() > 1.0:
        x = x / 255.0
    return x


def _infer_input_shape(obs_space):
    """
    Returns (C,H,W) as integers from either a Gym Box space or a raw tuple.
    Falls back to (1,20,10) if unclear.
    """
    shape = getattr(obs_space, "shape", None) or obs_space
    if shape is None:
        return (1, 20, 10)

    # common cases: (H,W,C) or (C,H,W)
    if len(shape) == 3:
        H,W,C = shape
        # If last dim is channels (e.g. 1), move to first
        if C <= 16:
            return (C, H, W)
        # Otherwise assume first is channels
        C,H,W = shape
        return (C,H,W)
    elif len(shape) == 2:
        H,W = shape
        return (1, H, W)
    else:
        # Fallback
        flat = int(np.prod(shape))
        return (1, flat, 1)


class ConvDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        C, H, W = input_shape

        self.features = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((20, 10)),   # <-- NYTT: fixera spatial storlek
            nn.Flatten(),
        )
        flat = 64 * 20 * 10                   # 12800

        self.head = nn.Sequential(
            nn.Linear(flat, 256), nn.ReLU(),
            nn.Linear(256, n_actions),
        )


    def forward(self, x):
        x = _to_nchw(x)
        return self.head(self.features(x))


class MLPDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        C,H,W = input_shape
        in_dim = C*H*W
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, n_actions),
        )
    def forward(self, x):
        return self.net(x)

def create_model(obs_space, action_space, model_type="dqn"):
    """
    Factory used by Agent:
      - obs_space: Gym space (Box) or shape-like
      - action_space: Gym Discrete (needs .n)
      - model_type: "dqn" (conv) or "mlp"
    """
    def _n_actions_from(action_space):
        # Gym/Gymnasium Discrete
        if hasattr(action_space, "n"):
            return int(action_space.n)
        # Already an int-like
        try:
            return int(action_space)
        except Exception as e:
            raise TypeError(
                f"Cannot determine n_actions from: {type(action_space)}"
            ) from e

    n_actions = _n_actions_from(action_space)



    C,H,W = _infer_input_shape(obs_space)

    if model_type in ("mlp", "dense"):
        return MLPDQN((C,H,W), n_actions)
    # default conv
    return ConvDQN((C,H,W), n_actions)
