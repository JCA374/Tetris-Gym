# src/agent.py - OPTIMIZED FOR 25,000+ EPISODE TRAINING

import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import os
import pickle
import math
from .model import create_model
from .utils import make_dir


class Agent:
    """Agent optimized for very long training runs (25,000+ episodes)"""

    def __init__(self, obs_space, action_space, lr=1e-4, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 memory_size=10000, batch_size=32, target_update=1000,
                 model_type="dqn", reward_shaping="none", shaping_config=None,
                 max_episodes=25000):

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
        self.batch_size = batch_size
        self.target_update = target_update
        self.max_episodes = max_episodes

        # 🔥 CALCULATE PROPER EPSILON DECAY FOR LONG TRAINING
        self.epsilon_decay_method = "adaptive"  # or "exponential", "linear", "step"
        
        if self.epsilon_decay_method == "exponential":
            # Calculate decay rate so epsilon reaches epsilon_end at 80% of max episodes
            decay_episodes = int(0.8 * max_episodes)  # 20,000 for 25,000 total
            self.epsilon_decay = math.pow(epsilon_end / epsilon_start, 1.0 / decay_episodes)
            print(f"Exponential decay: {self.epsilon_decay:.6f} over {decay_episodes} episodes")
            
        elif self.epsilon_decay_method == "linear":
            # Linear decay: epsilon decreases by constant amount each episode
            decay_episodes = int(0.8 * max_episodes)
            self.epsilon_linear_step = (epsilon_start - epsilon_end) / decay_episodes
            print(f"Linear decay: -{self.epsilon_linear_step:.6f} per episode over {decay_episodes} episodes")
            
        elif self.epsilon_decay_method == "adaptive":
            # Adaptive schedule optimized for Tetris learning phases
            self.epsilon_schedule = self._create_adaptive_schedule(max_episodes)
            print(f"Adaptive schedule with {len(self.epsilon_schedule)} phases")
            
        else:  # step decay
            self.epsilon_decay = epsilon_decay  # Use provided value
            
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

        # Reward shaping
        self.reward_shaping_type = reward_shaping
        print(f"Reward shaping: {reward_shaping}")

        # Tracking
        self.steps_done = 0
        self.episodes_done = 0
        self.total_rewards = []
        self.episode_metrics = []
        self._expected_shape = None

        print(f"Agent initialized for {max_episodes} episodes")
        print(f"Epsilon method: {self.epsilon_decay_method}")

    def _create_adaptive_schedule(self, max_episodes):
        """Create adaptive epsilon schedule optimized for Tetris learning"""
        schedule = []
        
        # Phase 1: High exploration (0-20% of episodes)
        # Need to discover basic line clearing
        phase1_end = int(0.2 * max_episodes)  # First 5,000 episodes
        schedule.append({
            'start_episode': 0,
            'end_episode': phase1_end,
            'start_epsilon': 1.0,
            'end_epsilon': 0.3,
            'description': 'Discovery phase - find line clearing'
        })
        
        # Phase 2: Medium exploration (20-60% of episodes) 
        # Learn strategic patterns
        phase2_end = int(0.6 * max_episodes)  # Episodes 5,000-15,000
        schedule.append({
            'start_episode': phase1_end,
            'end_episode': phase2_end,
            'start_epsilon': 0.3,
            'end_epsilon': 0.1,
            'description': 'Strategy phase - learn patterns'
        })
        
        # Phase 3: Low exploration (60-90% of episodes)
        # Refine advanced techniques
        phase3_end = int(0.9 * max_episodes)  # Episodes 15,000-22,500
        schedule.append({
            'start_episode': phase2_end,
            'end_episode': phase3_end,
            'start_epsilon': 0.1,
            'end_epsilon': 0.03,
            'description': 'Refinement phase - advanced play'
        })
        
        # Phase 4: Minimal exploration (90-100% of episodes)
        # Final optimization
        schedule.append({
            'start_episode': phase3_end,
            'end_episode': max_episodes,
            'start_epsilon': 0.03,
            'end_epsilon': 0.01,
            'description': 'Optimization phase - master play'
        })
        
        return schedule

    def _get_scheduled_epsilon(self, episode):
        """Get epsilon value based on adaptive schedule"""
        if self.epsilon_decay_method != "adaptive":
            return None
            
        for phase in self.epsilon_schedule:
            if phase['start_episode'] <= episode <= phase['end_episode']:
                # Linear interpolation within phase
                progress = (episode - phase['start_episode']) / max(1, phase['end_episode'] - phase['start_episode'])
                epsilon = phase['start_epsilon'] + progress * (phase['end_epsilon'] - phase['start_epsilon'])
                return max(self.epsilon_end, epsilon)
                
        return self.epsilon_end

    def _apply_reward_shaping(self, reward, done, info):
        """Apply reward shaping optimized for long training"""
        if self.reward_shaping_type == "none":
            return reward

        shaped_reward = reward

        # Penalties and bonuses
        if done:
            shaped_reward -= 20
        else:
            shaped_reward += 0.01

        # Line clear bonuses (stronger for long training)
        lines_cleared = info.get('lines_cleared', 0)
        if lines_cleared > 0:
            # Progressive bonus: more valuable in later training
            episode_bonus_multiplier = min(2.0, 1.0 + (self.episodes_done / 10000))
            shaped_reward += lines_cleared * 10 * episode_bonus_multiplier
            
            if lines_cleared == 4:  # Tetris bonus
                shaped_reward += 40 * episode_bonus_multiplier

        return shaped_reward

    def select_action(self, state, eval_mode=False):
        """Select action with proper epsilon handling"""
        if eval_mode or random.random() > self.epsilon:
            with torch.no_grad():
                state_tensor = self._preprocess_state(state)
                q_values = self.q_network(state_tensor)
                return q_values.max(1)[1].item()
        else:
            return random.randrange(self.n_actions)

    def remember(self, state, action, reward, next_state, done, info=None, original_reward=None):
        """Store experience with shape validation"""
        shaped_reward = self._apply_reward_shaping(reward, done, info or {})

        state_np = self._to_numpy_consistent(state)
        next_state_np = self._to_numpy_consistent(next_state)

        if self._expected_shape is None:
            self._expected_shape = state_np.shape
            print(f"Expected observation shape set to: {self._expected_shape}")

        if state_np.shape != self._expected_shape or next_state_np.shape != self._expected_shape:
            return  # Skip inconsistent experiences

        self.memory.append((state_np, action, shaped_reward, next_state_np, done))

    def _to_numpy_consistent(self, state):
        """Convert state to consistent numpy format"""
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        elif isinstance(state, np.ndarray):
            state = state.copy()
        else:
            raise ValueError(f"Unsupported state type: {type(state)}")

        if state.dtype != np.float32:
            state = state.astype(np.float32)

        if state.max() > 1.0:
            state = state / 255.0

        # Ensure consistent shape format
        if len(state.shape) == 3:  # (H, W, C)
            state = state.transpose(2, 0, 1)  # (C, H, W)
        elif len(state.shape) == 4 and state.shape[0] == 1:
            state = state.squeeze(0)
            if len(state.shape) == 3 and state.shape[-1] <= 16:
                state = state.transpose(2, 0, 1)

        return state

    def learn(self):
        """Learn from replay buffer (epsilon decay removed from here)"""
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Validate shapes
        state_shapes = [s.shape for s in states]
        if len(set(state_shapes)) > 1:
            return None

        try:
            states_tensor = torch.tensor(np.stack(states), device=self.device, dtype=torch.float32)
            actions_tensor = torch.tensor(actions, device=self.device, dtype=torch.long)
            rewards_tensor = torch.tensor(rewards, device=self.device, dtype=torch.float32)
            next_states_tensor = torch.tensor(np.stack(next_states), device=self.device, dtype=torch.float32)
            dones_tensor = torch.tensor(dones, device=self.device, dtype=torch.bool)
        except Exception as e:
            print(f"Tensor creation error: {e}")
            return None

        current_q_values = self.q_network(states_tensor).gather(1, actions_tensor.unsqueeze(1))

        with torch.no_grad():
            next_q_values = self.target_network(next_states_tensor).max(1)[0]
            target_q_values = rewards_tensor + (self.gamma * next_q_values * ~dones_tensor)

        loss = nn.functional.mse_loss(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return {
            'loss': loss.item(),
            'mean_q_value': current_q_values.mean().item(),
            'epsilon': self.epsilon
        }

    def end_episode(self, episode_reward, episode_length, lines_cleared, original_reward=None):
        """End episode with proper epsilon decay for long training"""
        self.total_rewards.append(episode_reward)
        self.episodes_done += 1

        # 🔥 PROPER EPSILON DECAY FOR LONG TRAINING
        old_epsilon = self.epsilon
        
        if self.epsilon_decay_method == "exponential":
            if self.epsilon > self.epsilon_end:
                self.epsilon *= self.epsilon_decay
                
        elif self.epsilon_decay_method == "linear":
            self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_linear_step)
            
        elif self.epsilon_decay_method == "adaptive":
            self.epsilon = self._get_scheduled_epsilon(self.episodes_done)
            
        else:  # step decay (original method)
            if self.epsilon > self.epsilon_end:
                self.epsilon *= self.epsilon_decay

        # Log epsilon changes at key milestones
        if (self.episodes_done <= 10 or 
            self.episodes_done % 1000 == 0 or 
            abs(self.epsilon - old_epsilon) > 0.01):
            print(f"Episode {self.episodes_done}: Epsilon = {self.epsilon:.4f}")
            
            # Show which phase we're in for adaptive
            if self.epsilon_decay_method == "adaptive":
                current_phase = None
                for phase in self.epsilon_schedule:
                    if phase['start_episode'] <= self.episodes_done <= phase['end_episode']:
                        current_phase = phase['description']
                        break
                if current_phase:
                    print(f"  Phase: {current_phase}")

        episode_data = {
            'episode': self.episodes_done,
            'reward': episode_reward,
            'length': episode_length,
            'lines_cleared': lines_cleared,
            'epsilon': self.epsilon,
        }
        self.episode_metrics.append(episode_data)

    def _preprocess_state(self, state):
        """Convert state to tensor for network input"""
        state_np = self._to_numpy_consistent(state)
        state_np = np.expand_dims(state_np, axis=0)
        return torch.tensor(state_np, device=self.device, dtype=torch.float32)

    def save_checkpoint(self, episode, model_dir="models/"):
        """Save checkpoint with robust error handling"""
        make_dir(model_dir)

        try:
            checkpoint_data = {
                'episode': episode,
                'steps_done': self.steps_done,
                'epsilon': self.epsilon,
                'total_rewards': self.total_rewards,
                'episode_metrics': self.episode_metrics,
                'reward_shaping_type': self.reward_shaping_type,
                'epsilon_decay_method': self.epsilon_decay_method,
                'max_episodes': self.max_episodes,
            }

            latest_path = os.path.join(model_dir, 'latest_checkpoint.pth')
            
            try:
                full_checkpoint = {
                    **checkpoint_data,
                    'q_network_state_dict': self.q_network.state_dict(),
                    'target_network_state_dict': self.target_network.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }
                torch.save(full_checkpoint, latest_path)
                print(f"✅ Checkpoint saved: {latest_path}")
                
            except Exception as torch_error:
                print(f"❌ PyTorch save failed: {torch_error}")
                
                # Alternative save method
                model_path = os.path.join(model_dir, 'latest_model.pth')
                torch.save({
                    'q_network': self.q_network.state_dict(),
                    'target_network': self.target_network.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }, model_path)
                
                data_path = os.path.join(model_dir, 'latest_training_data.pkl')
                with open(data_path, 'wb') as f:
                    pickle.dump(checkpoint_data, f)
                
                print(f"✅ Alternative save successful")

            # Periodic checkpoints
            if episode % 1000 == 0:  # Every 1000 episodes for long training
                try:
                    milestone_path = os.path.join(model_dir, f'checkpoint_episode_{episode}.pth')
                    torch.save({
                        'q_network_state_dict': self.q_network.state_dict(),
                        'episode': episode,
                        'epsilon': self.epsilon,
                    }, milestone_path)
                    print(f"📦 Milestone checkpoint: {milestone_path}")
                except Exception as e:
                    print(f"⚠️  Milestone checkpoint failed: {e}")

        except Exception as e:
            print(f"❌ Critical save error: {e}")

    def load_checkpoint(self, latest=False, path=None, model_dir="models/"):
        """Load checkpoint with robust error handling"""
        try:
            if latest:
                path = os.path.join(model_dir, 'latest_checkpoint.pth')

            if path and os.path.exists(path):
                try:
                    checkpoint = torch.load(path, map_location=self.device, weights_only=False)

                    self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
                    self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

                    self.episodes_done = checkpoint['episode']
                    self.steps_done = checkpoint['steps_done']
                    self.epsilon = checkpoint['epsilon']
                    self.total_rewards = checkpoint.get('total_rewards', [])
                    self.episode_metrics = checkpoint.get('episode_metrics', [])
                    
                    # Load epsilon method if available
                    if 'epsilon_decay_method' in checkpoint:
                        self.epsilon_decay_method = checkpoint['epsilon_decay_method']
                    if 'max_episodes' in checkpoint:
                        self.max_episodes = checkpoint['max_episodes']

                    print(f"✅ Checkpoint loaded: {path}")
                    print(f"Resuming from episode {self.episodes_done}, epsilon={self.epsilon:.4f}")
                    return True
                    
                except Exception as torch_error:
                    print(f"❌ Standard loading failed: {torch_error}")
                    
                    # Try alternative loading
                    model_path = os.path.join(model_dir, 'latest_model.pth')
                    data_path = os.path.join(model_dir, 'latest_training_data.pkl')
                    
                    if os.path.exists(model_path) and os.path.exists(data_path):
                        try:
                            model_data = torch.load(model_path, map_location=self.device, weights_only=False)
                            self.q_network.load_state_dict(model_data['q_network'])
                            self.target_network.load_state_dict(model_data['target_network'])
                            self.optimizer.load_state_dict(model_data['optimizer'])
                            
                            with open(data_path, 'rb') as f:
                                training_data = pickle.load(f)
                            
                            self.episodes_done = training_data['episode']
                            self.steps_done = training_data['steps_done']
                            self.epsilon = training_data['epsilon']
                            self.total_rewards = training_data.get('total_rewards', [])
                            self.episode_metrics = training_data.get('episode_metrics', [])
                            
                            print(f"✅ Alternative loading successful")
                            return True
                            
                        except Exception as alt_error:
                            print(f"❌ Alternative loading failed: {alt_error}")
                    
            else:
                print(f"❌ No checkpoint found")
                return False
                
        except Exception as e:
            print(f"❌ Critical loading error: {e}")
            
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
        """Compatibility method"""
        return {}

    def print_epsilon_schedule(self):
        """Print the complete epsilon schedule for verification"""
        if self.epsilon_decay_method == "adaptive":
            print("\n📊 EPSILON SCHEDULE FOR 25,000 EPISODES:")
            print("=" * 60)
            for i, phase in enumerate(self.epsilon_schedule, 1):
                print(f"Phase {i}: Episodes {phase['start_episode']:,}-{phase['end_episode']:,}")
                print(f"  Epsilon: {phase['start_epsilon']:.3f} → {phase['end_epsilon']:.3f}")
                print(f"  Goal: {phase['description']}")
                print()
        else:
            print(f"\nEpsilon method: {self.epsilon_decay_method}")
            print(f"Decay rate: {getattr(self, 'epsilon_decay', 'N/A')}")