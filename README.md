Here’s a minimal, modular layout that keeps things clear and lets each file hold a single responsibility:

```
tetris_ai/
├── README.md
├── requirements.txt
├── .gitignore
├── config.py            # global hyper-parameters & paths
├── train.py             # entry-point: training loop
├── evaluate.py          # entry-point: evaluation loop
│
├── src/                 # all your library code lives here
│   ├── __init__.py
│   ├── env.py           # wraps Gym/Tetris environment
│   ├── model.py         # defines your neural-net architecture
│   ├── agent.py         # agent class: select_action, remember, learn
│   └── utils.py         # logging, checkpoint save/load, plotting
│
├── models/              # saved weights/checkpoints
└── logs/                # TensorBoard logs or plain .csv/.txt histories
```

**File breakdown**

* **README.md**
  High-level project overview and quickstart (how to set up venv, install, run `train.py`, etc.).

* **requirements.txt**
  Pin your dependencies (e.g. `tensorflow`, `torch`, `gym-tetris`, `nes-py`, `numpy`, …).

* **.gitignore**
  Ignore `venv/`, `__pycache__/`, `models/`, `logs/`, etc.

* **config.py**

  ```python
  # Example
  ENV_NAME     = "ALE/Tetris-v5"
  LR           = 1e-4
  GAMMA        = 0.99
  BATCH_SIZE   = 32
  MAX_EPISODES = 500
  MODEL_DIR    = "models/"
  LOG_DIR      = "logs/"
  ```

  Centralize hyper-parameters, folder names, random seeds, etc.

* **train.py**

  ```python
  from config import *
  from src.env   import make_env
  from src.agent import Agent

  def main():
      env = make_env(ENV_NAME)
      agent = Agent(env.observation_space, env.action_space)
      for ep in range(MAX_EPISODES):
          state = env.reset()
          done = False
          while not done:
              action = agent.select_action(state)
              next_state, reward, done, _ = env.step(action)
              agent.remember(state, action, reward, next_state, done)
              agent.learn()
              state = next_state
          agent.save_checkpoint(ep)
  if __name__ == "__main__":
      main()
  ```

* **evaluate.py**

  ```python
  from config import *
  from src.env   import make_env
  from src.agent import Agent

  def main():
      env   = make_env(ENV_NAME)
      agent = Agent(env.observation_space, env.action_space)
      agent.load_checkpoint(latest=True)
      for _ in range(10):
          state, done = env.reset(), False
          total_reward = 0
          while not done:
              action = agent.select_action(state, eval_mode=True)
              state, reward, done, _ = env.step(action)
              total_reward += reward
          print("Episode reward:", total_reward)
  if __name__ == "__main__":
      main()
  ```

* **src/env.py**

  ```python
  import gym
  import gym_tetris
  import nes_py

  def make_env(name):
      env = gym.make(name)
      # apply wrappers: resize, grayscale, frame-stack, etc.
      return env
  ```

* **src/model.py**

  ```python
  import torch.nn as nn

  class DQN(nn.Module):
      def __init__(self, obs_space, action_space):
          super().__init__()
          # define conv layers, FC layers...
      def forward(self, x):
          # return Q-values
  ```

* **src/agent.py**

  ```python
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
  ```

* **src/utils.py**

  ```python
  import os
  def make_dir(path):
      os.makedirs(path, exist_ok=True)

  def plot_rewards(reward_list, save_path):
      # simple matplotlib plot and save
  ```

* **models/**
  Will hold your `.pth`/`.pt` or TensorFlow checkpoints.

* **logs/**
  For TensorBoard event files or your own CSV logs of reward vs. episode.

---

This structure keeps each concern isolated, makes it easy to find and test pieces independently, and is straightforward to extend when you add things like prioritized replay, different agents, or more advanced logging.
