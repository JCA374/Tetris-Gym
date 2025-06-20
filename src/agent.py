import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import os
import pickle
from .model import create_model
from .utils import make_dir


class Agent:
    """Fixed agent with robust checkpoint saving"""

    def __init__(self, obs_space, action_space, lr=1e-4, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 memory_size=10000, batch_size=32, target_update=1000,
                 model_type="dqn", reward_shaping="none", shaping_config=None):

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

        # Replay buffer - store as numpy arrays to avoid tensor issues
        self.memory = deque(maxlen=memory_size)

        # Simple reward shaping
        self.reward_shaping_type = reward_shaping
        print(f"Reward shaping: {reward_shaping}")

        # Tracking
        self.steps_done = 0
        self.episodes_done = 0
        self.total_rewards = []
        self.episode_metrics = []

        # Track expected observation shape
        self._expected_shape = None

        print(f"Agent initialized with {self.n_actions} actions")

    def _apply_reward_shaping(self, reward, done, info):
        """Apply simple reward shaping"""
        if self.reward_shaping_type == "none":
            return reward

        shaped_reward = reward

        # Basic penalties and bonuses
        if done:
            shaped_reward -= 20  # Game over penalty
        else:
            shaped_reward += 0.01  # Small survival bonus

        # Line clear bonus
        lines_cleared = info.get('lines_cleared', 0)
        if lines_cleared > 0:
            shaped_reward += lines_cleared * 10
            if lines_cleared == 4:  # Tetris bonus
                shaped_reward += 40

        return shaped_reward

    def select_action(self, state, eval_mode=False):
        """Select action using epsilon-greedy policy"""
        if eval_mode or random.random() > self.epsilon:
            with torch.no_grad():
                state_tensor = self._preprocess_state(state)
                q_values = self.q_network(state_tensor)
                return q_values.max(1)[1].item()
        else:
            return random.randrange(self.n_actions)

    def remember(self, state, action, reward, next_state, done, info=None, original_reward=None):
        """Store experience in replay buffer with shape validation"""
        # Apply reward shaping
        shaped_reward = self._apply_reward_shaping(reward, done, info or {})

        # Convert to consistent numpy format
        state_np = self._to_numpy_consistent(state)
        next_state_np = self._to_numpy_consistent(next_state)

        # Validate shapes
        if self._expected_shape is None:
            self._expected_shape = state_np.shape
            print(f"Expected observation shape set to: {self._expected_shape}")

        if state_np.shape != self._expected_shape:
            print(
                f"Warning: State shape mismatch: {state_np.shape} vs {self._expected_shape}")
            return  # Skip this experience

        if next_state_np.shape != self._expected_shape:
            print(
                f"Warning: Next state shape mismatch: {next_state_np.shape} vs {self._expected_shape}")
            return  # Skip this experience

        # Store experience
        self.memory.append(
            (state_np, action, shaped_reward, next_state_np, done))

    def _to_numpy_consistent(self, state):
        """Convert state to consistent numpy format for storage"""
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        elif isinstance(state, np.ndarray):
            state = state.copy()
        else:
            raise ValueError(f"Unsupported state type: {type(state)}")

        # Ensure float32
        if state.dtype != np.float32:
            state = state.astype(np.float32)

        # Normalize if needed
        if state.max() > 1.0:
            state = state / 255.0

        # Ensure consistent shape format (C, H, W) for storage
        if len(state.shape) == 3:  # (H, W, C)
            state = state.transpose(2, 0, 1)  # (C, H, W)
        elif len(state.shape) == 4:  # (1, H, W, C) or (1, C, H, W)
            if state.shape[0] == 1:  # Remove batch dimension
                state = state.squeeze(0)
                # (H, W, C)
                if len(state.shape) == 3 and state.shape[-1] <= 16:
                    state = state.transpose(2, 0, 1)  # (C, H, W)

        return state

    def learn(self):
        """Learn from replay buffer with robust error handling"""
        if len(self.memory) < self.batch_size:
            return None

        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Validate batch shapes
        state_shapes = [s.shape for s in states]
        next_state_shapes = [s.shape for s in next_states]

        unique_state_shapes = set(state_shapes)
        unique_next_shapes = set(next_state_shapes)

        if len(unique_state_shapes) > 1:
            print(
                f"Error: Inconsistent state shapes in batch: {unique_state_shapes}")
            return None

        if len(unique_next_shapes) > 1:
            print(
                f"Error: Inconsistent next state shapes in batch: {unique_next_shapes}")
            return None

        try:
            # Convert to tensors
            states_tensor = torch.tensor(
                np.stack(states), device=self.device, dtype=torch.float32)
            actions_tensor = torch.tensor(
                actions, device=self.device, dtype=torch.long)
            rewards_tensor = torch.tensor(
                rewards, device=self.device, dtype=torch.float32)
            next_states_tensor = torch.tensor(
                np.stack(next_states), device=self.device, dtype=torch.float32)
            dones_tensor = torch.tensor(
                dones, device=self.device, dtype=torch.bool)

        except Exception as e:
            print(f"Error creating tensors: {e}")
            print(f"State shapes in batch: {[s.shape for s in states[:3]]}")
            return None

        # Current Q values
        current_q_values = self.q_network(states_tensor).gather(
            1, actions_tensor.unsqueeze(1))

        # Next Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states_tensor).max(1)[0]
            target_q_values = rewards_tensor + \
                (self.gamma * next_q_values * ~dones_tensor)

        # Compute loss
        loss = nn.functional.mse_loss(
            current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
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

        return {
            'loss': loss.item(),
            'mean_q_value': current_q_values.mean().item(),
            'epsilon': self.epsilon
        }

    def _preprocess_state(self, state):
        """Convert state to tensor for network input"""
        state_np = self._to_numpy_consistent(state)

        # Add batch dimension
        state_np = np.expand_dims(state_np, axis=0)  # (1, C, H, W)

        # Convert to tensor
        state_tensor = torch.tensor(
            state_np, device=self.device, dtype=torch.float32)

        return state_tensor

    def end_episode(self, episode_reward, episode_length, lines_cleared, original_reward=None):
        """End episode tracking"""
        self.total_rewards.append(episode_reward)
        self.episodes_done += 1

        episode_data = {
            'episode': self.episodes_done,
            'reward': episode_reward,
            'length': episode_length,
            'lines_cleared': lines_cleared,
            'epsilon': self.epsilon,
        }
        self.episode_metrics.append(episode_data)

    def save_checkpoint(self, episode, model_dir="models/"):
        """Save checkpoint with robust error handling"""
        make_dir(model_dir)

        try:
            # Prepare checkpoint data
            checkpoint_data = {
                'episode': episode,
                'steps_done': self.steps_done,
                'epsilon': self.epsilon,
                'total_rewards': self.total_rewards,
                'episode_metrics': self.episode_metrics,
                'reward_shaping_type': self.reward_shaping_type,
            }

            # Save model weights separately for better compatibility
            latest_path = os.path.join(model_dir, 'latest_checkpoint.pth')
            
            # Method 1: Try standard torch.save
            try:
                # Prepare full checkpoint
                full_checkpoint = {
                    **checkpoint_data,
                    'q_network_state_dict': self.q_network.state_dict(),
                    'target_network_state_dict': self.target_network.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }
                
                torch.save(full_checkpoint, latest_path)
                print(f"✅ Checkpoint saved successfully: {latest_path}")
                
            except Exception as torch_error:
                print(f"❌ PyTorch save failed: {torch_error}")
                print("🔄 Trying alternative save method...")
                
                # Method 2: Save components separately
                try:
                    # Save model weights only
                    model_path = os.path.join(model_dir, 'latest_model.pth')
                    torch.save({
                        'q_network': self.q_network.state_dict(),
                        'target_network': self.target_network.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                    }, model_path)
                    
                    # Save training data with pickle
                    data_path = os.path.join(model_dir, 'latest_training_data.pkl')
                    with open(data_path, 'wb') as f:
                        pickle.dump(checkpoint_data, f)
                    
                    print(f"✅ Alternative save successful:")
                    print(f"   Model: {model_path}")
                    print(f"   Data: {data_path}")
                    
                except Exception as alt_error:
                    print(f"❌ Alternative save also failed: {alt_error}")
                    print("⚠️  Continuing training without saving...")

            # Save periodic checkpoint
            if episode % 100 == 0:
                try:
                    episode_path = os.path.join(model_dir, f'checkpoint_episode_{episode}.pth')
                    torch.save({
                        'q_network_state_dict': self.q_network.state_dict(),
                        'episode': episode,
                        'epsilon': self.epsilon,
                    }, episode_path)
                    print(f"📦 Episode checkpoint saved: {episode_path}")
                except Exception as e:
                    print(f"⚠️  Episode checkpoint failed: {e}")

        except Exception as e:
            print(f"❌ Critical error in save_checkpoint: {e}")
            print("⚠️  Training will continue without saving...")

    def load_checkpoint(self, latest=False, path=None, model_dir="models/"):
        """Load checkpoint with robust error handling"""
        try:
            if latest:
                path = os.path.join(model_dir, 'latest_checkpoint.pth')

            if path and os.path.exists(path):
                # Try standard loading
                try:
                    checkpoint = torch.load(
                        path, map_location=self.device, weights_only=False)

                    self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
                    self.target_network.load_state_dict(
                        checkpoint['target_network_state_dict'])
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

                    self.episodes_done = checkpoint['episode']
                    self.steps_done = checkpoint['steps_done']
                    self.epsilon = checkpoint['epsilon']
                    self.total_rewards = checkpoint.get('total_rewards', [])
                    self.episode_metrics = checkpoint.get('episode_metrics', [])

                    print(f"✅ Checkpoint loaded: {path}")
                    print(
                        f"Resuming from episode {self.episodes_done}, epsilon={self.epsilon:.4f}")
                    return True
                    
                except Exception as torch_error:
                    print(f"❌ Standard loading failed: {torch_error}")
                    
                    # Try alternative loading
                    model_path = os.path.join(model_dir, 'latest_model.pth')
                    data_path = os.path.join(model_dir, 'latest_training_data.pkl')
                    
                    if os.path.exists(model_path) and os.path.exists(data_path):
                        try:
                            # Load model weights
                            model_data = torch.load(model_path, map_location=self.device, weights_only=False)
                            self.q_network.load_state_dict(model_data['q_network'])
                            self.target_network.load_state_dict(model_data['target_network'])
                            self.optimizer.load_state_dict(model_data['optimizer'])
                            
                            # Load training data
                            with open(data_path, 'rb') as f:
                                training_data = pickle.load(f)
                            
                            self.episodes_done = training_data['episode']
                            self.steps_done = training_data['steps_done']
                            self.epsilon = training_data['epsilon']
                            self.total_rewards = training_data.get('total_rewards', [])
                            self.episode_metrics = training_data.get('episode_metrics', [])
                            
                            print(f"✅ Alternative loading successful")
                            print(f"Resuming from episode {self.episodes_done}, epsilon={self.epsilon:.4f}")
                            return True
                            
                        except Exception as alt_error:
                            print(f"❌ Alternative loading failed: {alt_error}")
                    
            else:
                print(f"❌ No checkpoint found at {path}")
                return False
                
        except Exception as e:
            print(f"❌ Critical error in load_checkpoint: {e}")
            
        return False

    def get_stats(self):
        """Get training statistics"""
        if not self.total_rewards:
            return {}

        return {
            'episodes': len(self.total_rewards),
            'steps': self.steps_done,
            'epsilon': self.epsilon,
            'avg_reward': np.mean(self.total_rewards[-100:]),
            'max_reward': np.max(self.total_rewards),
            'min_reward': np.min(self.total_rewards),
        }

    def get_shaping_analysis(self):
        """Compatibility method for training script"""
        return {}