# src/agent.py
"""DQN Agent for Tetris - FIXED with proper action mapping and exploration"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from src.model import create_model
from config import (
    ACTION_NOOP,
    ACTION_LEFT, 
    ACTION_RIGHT, 
    ACTION_DOWN, 
    ACTION_ROTATE_CW,
    ACTION_ROTATE_CCW,
    ACTION_HARD_DROP,
    ACTION_SWAP
)


class Agent:
    def __init__(self, obs_space, action_space, lr=0.001, gamma=0.99, 
                 batch_size=32, memory_size=10000, target_update=1000,
                 model_type="dqn", epsilon_start=1.0, epsilon_end=0.01,
                 epsilon_decay=0.9999, epsilon_decay_method="exponential",
                 reward_shaping="balanced", max_episodes=10000, device=None):
        
        # Environment
        self.obs_space = obs_space
        self.action_space = action_space
        self.n_actions = action_space.n
        
        # Device
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Agent using device: {self.device}")
        
        # Hyperparameters
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        
        # Epsilon parameters
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        self.epsilon_decay_method = epsilon_decay_method
        self.max_episodes = max_episodes
        
        # Create adaptive schedule if needed
        if epsilon_decay_method == "adaptive":
            self.epsilon_schedule = self._create_adaptive_schedule(max_episodes)
        
        # Networks
        self.q_network = create_model(obs_space, action_space, model_type).to(self.device)
        self.target_network = create_model(obs_space, action_space, model_type).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Initialize target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Replay buffer
        self.memory = deque(maxlen=memory_size)
        
        # Reward shaping
        self.reward_shaping_type = reward_shaping
        print(f"Reward shaping: {reward_shaping}")
        
        # Tracking
        self.steps_done = 0
        self.episodes_done = 0
        self.total_rewards = []
        self.episode_metrics = []
        self._expected_shape = None
        
        # Exploration strategy tracking
        self._explore_direction = 0  # -1 = left, +1 = right, 0 = none
        self._explore_steps_remaining = 0
        self._column_visit_count = np.zeros(10)  # Track which columns we've visited
        
        print(f"Agent initialized for {max_episodes} episodes")
        print(f"Epsilon method: {self.epsilon_decay_method}")
    
    def _create_adaptive_schedule(self, max_episodes):
        """Create adaptive epsilon schedule optimized for Tetris learning"""
        schedule = []
        
        # Phase 1: High exploration (0-20% of episodes)
        phase1_end = int(0.2 * max_episodes)
        schedule.append({
            'start_episode': 0,
            'end_episode': phase1_end,
            'start_epsilon': 1.0,
            'end_epsilon': 0.3,
            'description': 'Discovery phase - find line clearing'
        })
        
        # Phase 2: Medium exploration (20-60% of episodes)
        phase2_end = int(0.6 * max_episodes)
        schedule.append({
            'start_episode': phase1_end,
            'end_episode': phase2_end,
            'start_epsilon': 0.3,
            'end_epsilon': 0.1,
            'description': 'Strategy phase - learn patterns'
        })
        
        # Phase 3: Low exploration (60-90% of episodes)
        phase3_end = int(0.9 * max_episodes)
        schedule.append({
            'start_episode': phase2_end,
            'end_episode': phase3_end,
            'start_epsilon': 0.1,
            'end_epsilon': 0.03,
            'description': 'Refinement phase - advanced play'
        })
        
        # Phase 4: Minimal exploration (90-100% of episodes)
        schedule.append({
            'start_episode': phase3_end,
            'end_episode': max_episodes,
            'start_epsilon': 0.03,
            'end_epsilon': 0.01,
            'description': 'Optimization phase - master play'
        })
        
        return schedule

    def select_action(self, state, training=True):
        """
        Select action using epsilon-greedy strategy with ALL 8 actions.
        
        Args:
            state: Current game observation
            training: If True, use epsilon-greedy; if False, always exploit
        
        Returns:
            action: Integer action ID (0-7)
        """
        from config import (ACTION_NOOP, ACTION_LEFT, ACTION_RIGHT, ACTION_DOWN,
                        ACTION_ROTATE_CW, ACTION_ROTATE_CCW, ACTION_HARD_DROP, ACTION_SWAP)
        
        if training and random.random() < self.epsilon:
            # ===================================================================
            # EXPLORATION: Random action selection
            # ===================================================================
            
            exploration_roll = random.random()
            
            if self.episodes_done < 1000:
                # ---------------------------------------------------------------
                # EARLY TRAINING: Prioritize basic movement and line clearing
                # ---------------------------------------------------------------
                if exploration_roll < 0.35:
                    # 35% - Horizontal movement (critical for line clearing!)
                    return random.choice([ACTION_LEFT, ACTION_RIGHT])
                
                elif exploration_roll < 0.50:
                    # 15% - Both rotation directions
                    return random.choice([ACTION_ROTATE_CW, ACTION_ROTATE_CCW])
                
                elif exploration_roll < 0.55:
                    # 5% - SWAP (try holding pieces)
                    return ACTION_SWAP
                
                elif exploration_roll < 0.75:
                    # 20% - Soft drop (controlled descent)
                    return ACTION_DOWN
                
                elif exploration_roll < 0.95:
                    # 20% - Hard drop (fast placement)
                    return ACTION_HARD_DROP
                
                else:
                    # 5% - NOOP (sometimes waiting is strategic)
                    return ACTION_NOOP
            
            else:
                # ---------------------------------------------------------------
                # LATER TRAINING: More balanced exploration
                # ---------------------------------------------------------------
                if exploration_roll < 0.25:
                    # 25% - Horizontal movement
                    return random.choice([ACTION_LEFT, ACTION_RIGHT])
                
                elif exploration_roll < 0.40:
                    # 15% - Both rotation directions
                    return random.choice([ACTION_ROTATE_CW, ACTION_ROTATE_CCW])
                
                elif exploration_roll < 0.50:
                    # 10% - SWAP
                    return ACTION_SWAP
                
                elif exploration_roll < 0.60:
                    # 10% - Soft drop
                    return ACTION_DOWN
                
                elif exploration_roll < 0.95:
                    # 35% - Hard drop (faster games)
                    return ACTION_HARD_DROP
                
                else:
                    # 5% - NOOP
                    return ACTION_NOOP
        
        else:
            # ===================================================================
            # EXPLOITATION: Use Q-network to select best action
            # ===================================================================
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done, info=None, original_reward=None):
        """Store experience with shape validation"""
        # NOTE: train.py already applies reward shaping, so we store the reward as-is
        
        # Validate shapes
        if self._expected_shape is None:
            self._expected_shape = state.shape
        
        # Store experience (reward is already shaped by train.py)
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        """Perform one step of learning"""
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Preprocess states if needed
        states = self._preprocess_batch(states)
        next_states = self._preprocess_batch(next_states)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values (from target network)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Loss and optimization
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Update steps counter
        self.steps_done += 1
        
        # Update target network
        if self.steps_done % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return {"loss": loss.item()}
    
    def _preprocess_state(self, state):
        """Preprocess single state for network"""
        # Ensure state is a numpy array
        if not isinstance(state, np.ndarray):
            state = np.array(state)
        
        # Add batch dimension
        state = state[np.newaxis, ...]
        
        # Convert to tensor
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        return state_tensor
    
    def _preprocess_batch(self, batch):
        """Preprocess batch of states"""
        # If already properly shaped, return as is
        if len(batch.shape) == 4:  # (B, H, W, C) or (B, C, H, W)
            # Check if needs channel rearrangement
            if batch.shape[-1] <= 4:  # Last dim is channels
                batch = batch.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        
        return batch
    
    def update_epsilon(self):
        """Update epsilon based on selected decay method"""
        if self.epsilon_decay_method == "exponential":
            # Exponential decay
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            
        elif self.epsilon_decay_method == "linear":
            # Linear decay
            decay_amount = (self.epsilon_start - self.epsilon_end) / self.max_episodes
            self.epsilon = max(self.epsilon_end, self.epsilon - decay_amount)
            
        elif self.epsilon_decay_method == "adaptive":
            # Adaptive schedule based on episode
            for phase in self.epsilon_schedule:
                if phase['start_episode'] <= self.episodes_done < phase['end_episode']:
                    # Linear interpolation within phase
                    progress = (self.episodes_done - phase['start_episode']) / (phase['end_episode'] - phase['start_episode'])
                    self.epsilon = phase['start_epsilon'] + progress * (phase['end_epsilon'] - phase['start_epsilon'])
                    break
    
    def end_episode(self, episode_reward, episode_steps, lines_cleared, original_reward=None):
        """Called at the end of each episode"""
        self.episodes_done += 1
        self.total_rewards.append(episode_reward)
        
        # Update epsilon
        self.update_epsilon()
        
        # Reset exploration state
        self._explore_direction = 0
        self._explore_steps_remaining = 0
        
        # Store episode metrics
        self.episode_metrics.append({
            'episode': self.episodes_done,
            'reward': episode_reward,
            'steps': episode_steps,
            'lines': lines_cleared,
            'epsilon': self.epsilon,
            'original_reward': original_reward
        })
    
    def save_checkpoint(self, episode, model_dir="models/"):
        """Save model checkpoint"""
        import os
        os.makedirs(model_dir, exist_ok=True)
        
        checkpoint = {
            'episode': episode,
            'q_network_state': self.q_network.state_dict(),
            'target_network_state': self.target_network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episodes_done': self.episodes_done,
            'steps_done': self.steps_done,
        }
        
        path = os.path.join(model_dir, f'checkpoint_ep{episode}.pth')
        torch.save(checkpoint, path)
        
        # Also save as latest
        latest_path = os.path.join(model_dir, 'checkpoint_latest.pth')
        torch.save(checkpoint, latest_path)
        
        return path
    
    def load_checkpoint(self, path=None, latest=False, model_dir="models/"):
        """Load model checkpoint"""
        import os
        
        if latest:
            path = os.path.join(model_dir, 'checkpoint_latest.pth')
        
        if path and os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            
            self.q_network.load_state_dict(checkpoint['q_network_state'])
            self.target_network.load_state_dict(checkpoint['target_network_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.epsilon = checkpoint['epsilon']
            self.episodes_done = checkpoint['episodes_done']
            self.steps_done = checkpoint['steps_done']
            
            print(f"Loaded checkpoint from {path}")
            print(f"  Episode: {checkpoint['episode']}")
            print(f"  Epsilon: {self.epsilon:.4f}")
            
            return True
        
        return False