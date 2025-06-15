import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import os
from src.model import create_model
from src.utils import make_dir


class Agent:
    def __init__(self, obs_space, action_space, lr=1e-4, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 memory_size=10000, batch_size=32, target_update=1000,
                 model_type="dqn"):
        """
        DQN Agent optimized for Tetris Gymnasium
        
        Args:
            obs_space: Observation space from environment
            action_space: Action space from environment
            lr: Learning rate
            gamma: Discount factor
            epsilon_start: Starting exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Exploration decay rate
            memory_size: Replay buffer size
            batch_size: Batch size for training
            target_update: Steps between target network updates
            model_type: "dqn" or "dueling_dqn"
        """
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.obs_space = obs_space
        self.action_space = action_space
        self.n_actions = action_space.n

        # Hyperparameters
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update

        # Networks
        self.q_network = create_model(
            obs_space, action_space, model_type).to(self.device)
        self.target_network = create_model(
            obs_space, action_space, model_type).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Initialize target network
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Replay buffer
        self.memory = deque(maxlen=memory_size)

        # Training tracking
        self.steps_done = 0
        self.episodes_done = 0
        self.total_rewards = []

        print(f"Agent initialized with {self.n_actions} actions")
        print(
            f"Q-network: {sum(p.numel() for p in self.q_network.parameters())} parameters")

    def select_action(self, state, eval_mode=False):
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current observation
            eval_mode: If True, use greedy policy (no exploration)
        """
        if eval_mode or random.random() > self.epsilon:
            with torch.no_grad():
                state_tensor = self._preprocess_state(state)
                q_values = self.q_network(state_tensor)
                return q_values.max(1)[1].item()
        else:
            return random.randrange(self.n_actions)

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        """Perform one step of learning"""
        if len(self.memory) < self.batch_size:
            return

        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.stack([self._preprocess_state(s)
                             for s in states]).squeeze(1)
        actions = torch.tensor(actions, device=self.device, dtype=torch.long)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float)
        next_states = torch.stack([self._preprocess_state(s)
                                  for s in next_states]).squeeze(1)
        dones = torch.tensor(dones, device=self.device, dtype=torch.bool)

        # Current Q values
        current_q_values = self.q_network(
            states).gather(1, actions.unsqueeze(1))

        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        # Compute loss
        loss = nn.functional.mse_loss(
            current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)

        self.optimizer.step()

        # Update target network
        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            print(f"Target network updated at step {self.steps_done}")

        # Decay epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

    def save_checkpoint(self, episode, model_dir="models/"):
        """Save model checkpoint"""
        make_dir(model_dir)

        checkpoint = {
            'episode': episode,
            'steps_done': self.steps_done,
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'total_rewards': self.total_rewards,
        }

        # Save latest checkpoint
        latest_path = os.path.join(model_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, latest_path)

        # Save periodic checkpoint
        if episode % 100 == 0:
            episode_path = os.path.join(
                model_dir, f'checkpoint_episode_{episode}.pth')
            torch.save(checkpoint, episode_path)
            print(f"Checkpoint saved: {episode_path}")

    def load_checkpoint(self, latest=False, path=None, model_dir="models/"):
        """Load model checkpoint"""
        if latest:
            path = os.path.join(model_dir, 'latest_checkpoint.pth')

        if path and os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)

            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(
                checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            self.episodes_done = checkpoint['episode']
            self.steps_done = checkpoint['steps_done']
            self.epsilon = checkpoint['epsilon']
            self.total_rewards = checkpoint.get('total_rewards', [])

            print(f"Checkpoint loaded: {path}")
            print(
                f"Resuming from episode {self.episodes_done}, epsilon={self.epsilon:.4f}")
            return True
        else:
            print(f"No checkpoint found at {path}")
            return False

    def _preprocess_state(self, state):
        """Convert state to tensor and normalize"""
        if isinstance(state, np.ndarray):
            # Ensure correct shape and type
            if state.dtype != np.float32:
                state = state.astype(np.float32)

            # Normalize to [0, 1] if needed
            if state.max() > 1.0:
                state = state / 255.0

            # Handle different input shapes
            if len(state.shape) == 3:  # (H, W, C)
                state = state.transpose(2, 0, 1)  # (C, H, W)
                state = np.expand_dims(state, axis=0)  # (1, C, H, W)
            elif len(state.shape) == 2:  # (H, W)
                state = np.expand_dims(state, axis=0)  # (1, H, W)
                state = np.expand_dims(state, axis=0)  # (1, 1, H, W)
            elif len(state.shape) == 1:  # Feature vector
                state = np.expand_dims(state, axis=0)  # (1, features)

            return torch.tensor(state, device=self.device, dtype=torch.float)
        else:
            raise ValueError(f"Unsupported state type: {type(state)}")

    def get_stats(self):
        """Get training statistics"""
        if not self.total_rewards:
            return {}

        return {
            'episodes': len(self.total_rewards),
            'steps': self.steps_done,
            'epsilon': self.epsilon,
            # Last 100 episodes
            'avg_reward': np.mean(self.total_rewards[-100:]),
            'max_reward': np.max(self.total_rewards),
            'min_reward': np.min(self.total_rewards),
        }

    def add_episode_reward(self, reward):
        """Add episode reward for tracking"""
        self.total_rewards.append(reward)
        self.episodes_done += 1

        # Print progress
        if self.episodes_done % 10 == 0:
            stats = self.get_stats()
            print(f"Episode {self.episodes_done}: "
                  f"Reward={reward:.1f}, "
                  f"Avg={stats['avg_reward']:.1f}, "
                  f"Epsilon={self.epsilon:.4f}")
