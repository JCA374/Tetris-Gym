import random
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import os
from .model import create_model
from .utils import make_dir


class Agent:
    def __init__(self, obs_space, action_space, lr=1e-4, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 memory_size=10000, batch_size=32, target_update=1000,
                 model_type="dqn", reward_shaping="none", shaping_config=None):
        """
        Enhanced DQN Agent with reward shaping support
        
        Args:
            reward_shaping: "none", "simple", or "full"
            shaping_config: Configuration dict for reward shaper
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

        # Reward shaping setup
        self.reward_shaping_type = reward_shaping
        if reward_shaping != "none":
            # Import here to avoid circular import
            try:
                from .reward_shaping import create_reward_shaper
                config = shaping_config or {}
                self.reward_shaper = create_reward_shaper(
                    reward_shaping, **config)
                print(f"Reward shaping enabled: {reward_shaping}")
            except ImportError:
                print("Warning: reward_shaping module not found, using simple shaping")
                self.reward_shaper = self._create_simple_shaper(
                    shaping_config or {})
        else:
            self.reward_shaper = None
            print("No reward shaping")

        # Enhanced tracking
        self.steps_done = 0
        self.episodes_done = 0
        self.total_rewards = []
        self.shaped_rewards = []  # Track shaped vs original rewards
        self.episode_metrics = []  # Store detailed metrics per episode

        # Ablation study data
        self.ablation_data = {
            'original_rewards': [],
            'shaped_rewards': [],
            'shaping_components': [],
            'episode_lengths': [],
            'lines_cleared': []
        }

        print(f"Agent initialized with {self.n_actions} actions")
        print(
            f"Q-network: {sum(p.numel() for p in self.q_network.parameters())} parameters")

    def _create_simple_shaper(self, config):
        """Create a simple reward shaper if main module not available"""
        class SimpleShaper:
            def __init__(self, height_weight=-0.5, hole_weight=-2.0):
                self.height_weight = config.get('height_weight', height_weight)
                self.hole_weight = config.get('hole_weight', hole_weight)

            def calculate_reward(self, obs, action, base_reward, done, info):
                board = self._extract_board(obs)
                if board is None:
                    return base_reward

                max_height = self._get_max_height(board) / board.shape[0]
                holes = self._count_holes(board)

                shaped_reward = base_reward
                shaped_reward += max_height * self.height_weight
                shaped_reward += holes * self.hole_weight

                if done:
                    shaped_reward -= 20
                else:
                    shaped_reward += 0.01

                return shaped_reward

            def _extract_board(self, obs):
                if isinstance(obs, dict) and 'board' in obs:
                    return obs['board']
                elif isinstance(obs, np.ndarray) and len(obs.shape) == 2:
                    return obs
                return None

            def _get_max_height(self, board):
                max_height = 0
                for col in range(board.shape[1]):
                    for row in range(board.shape[0]):
                        if board[row, col] != 0:
                            height = board.shape[0] - row
                            max_height = max(max_height, height)
                            break
                return max_height

            def _count_holes(self, board):
                holes = 0
                for col in range(board.shape[1]):
                    found_filled = False
                    for row in range(board.shape[0]):
                        if board[row, col] != 0:
                            found_filled = True
                        elif found_filled and board[row, col] == 0:
                            holes += 1
                return holes

            def reset_episode(self):
                pass

            def episode_end(self, *args):
                pass

        return SimpleShaper()

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
        """
        Store experience with reward shaping
        
        Args:
            original_reward: Store original reward for ablation studies
        """
        # Apply reward shaping (may return float OR (float, metrics))
        if self.reward_shaper is not None and info is not None:
            result = self.reward_shaper.calculate_reward(
                state, action, reward, done, info)
            if isinstance(result, tuple):
                shaped_reward, shaping_metrics = result
            else:
                shaped_reward, shaping_metrics = result, None
        else:
            shaped_reward, shaping_metrics = reward, None

        # Store experience with shaped reward
        self.memory.append((state, action, shaped_reward, next_state, done))

        # Track for ablation studies
        if original_reward is not None:
            self.ablation_data['original_rewards'].append(original_reward)
            self.ablation_data['shaped_rewards'].append(shaped_reward)
            if shaping_metrics:
                self.ablation_data['shaping_components'].append({
                    'max_height': getattr(shaping_metrics, 'max_height', 0),
                    'holes': getattr(shaping_metrics, 'holes', 0),
                    'bumpiness': getattr(shaping_metrics, 'bumpiness', 0),
                    'lines_cleared': getattr(shaping_metrics, 'lines_cleared', 0),
                    'tetris_bonus': getattr(shaping_metrics, 'tetris_bonus', False)
                })

    def learn(self):
        """Enhanced learning with shaping metrics tracking"""
        if len(self.memory) < self.batch_size:
            return None

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

    def end_episode(self, episode_reward, episode_length, lines_cleared, original_reward=None):
        """
        Enhanced episode ending with comprehensive tracking
        
        Args:
            original_reward: Original (unshaped) episode reward for comparison
        """
        self.total_rewards.append(episode_reward)
        self.episodes_done += 1

        # Track shaping effectiveness
        if original_reward is not None:
            self.shaped_rewards.append(episode_reward)
            self.ablation_data['episode_lengths'].append(episode_length)
            self.ablation_data['lines_cleared'].append(lines_cleared)

        # Update reward shaper
        if self.reward_shaper is not None:
            self.reward_shaper.episode_end(
                episode_reward, episode_length, lines_cleared)
            self.reward_shaper.reset_episode()

        # Store episode metrics
        episode_data = {
            'episode': self.episodes_done,
            'reward': episode_reward,
            'length': episode_length,
            'lines_cleared': lines_cleared,
            'epsilon': self.epsilon,
            'original_reward': original_reward
        }
        self.episode_metrics.append(episode_data)

        # Print progress with shaping info
        if self.episodes_done % 10 == 0:
            stats = self.get_stats()
            shaping_info = ""
            if original_reward is not None:
                improvement = episode_reward - original_reward
                shaping_info = f", Shaping: +{improvement:.1f}"

            print(f"Episode {self.episodes_done}: "
                  f"Reward={episode_reward:.1f}{shaping_info}, "
                  f"Avg={stats['avg_reward']:.1f}, "
                  f"Lines={lines_cleared}, "
                  f"Epsilon={self.epsilon:.4f}")

    def get_shaping_analysis(self):
        """Get detailed analysis of reward shaping effectiveness"""
        if not self.ablation_data['original_rewards']:
            return {}

        original = np.array(self.ablation_data['original_rewards'])
        shaped = np.array(self.ablation_data['shaped_rewards'])

        analysis = {
            'mean_original_reward': np.mean(original),
            'mean_shaped_reward': np.mean(shaped),
            'reward_improvement': np.mean(shaped - original),
            'improvement_std': np.std(shaped - original),
            'correlation': np.corrcoef(original, shaped)[0, 1] if len(original) > 1 else 0,
        }

        # Component analysis
        if self.ablation_data['shaping_components']:
            components = self.ablation_data['shaping_components']
            analysis['avg_height_penalty'] = np.mean(
                [c['max_height'] for c in components])
            analysis['avg_holes'] = np.mean([c['holes'] for c in components])
            analysis['tetris_bonus_frequency'] = np.mean(
                [c['tetris_bonus'] for c in components])

        # Get reward shaper ablation data if available
        if self.reward_shaper and hasattr(self.reward_shaper, 'get_ablation_data'):
            shaper_data = self.reward_shaper.get_ablation_data()
            analysis.update(shaper_data)

        return analysis

    def save_checkpoint(self, episode, model_dir="models/"):
        """Enhanced checkpoint saving with shaping data"""
        make_dir(model_dir)

        checkpoint = {
            'episode': episode,
            'steps_done': self.steps_done,
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'total_rewards': self.total_rewards,
            'episode_metrics': self.episode_metrics,
            'reward_shaping_type': self.reward_shaping_type,
            'shaping_analysis': self.get_shaping_analysis(),
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
        """Enhanced checkpoint loading with shaping data"""
        if latest:
            path = os.path.join(model_dir, 'latest_checkpoint.pth')

        if path and os.path.exists(path):
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

            print(f"Checkpoint loaded: {path}")
            print(
                f"Resuming from episode {self.episodes_done}, epsilon={self.epsilon:.4f}")

            # Print shaping analysis if available
            shaping_analysis = checkpoint.get('shaping_analysis', {})
            if shaping_analysis:
                print(f"Reward shaping analysis:")
                for key, value in shaping_analysis.items():
                    if isinstance(value, (int, float)):
                        print(f"  {key}: {value:.3f}")

            return True
        else:
            print(f"No checkpoint found at {path}")
            return False

    def _preprocess_state(self, state):
        """Convert state to tensor and normalize"""
        if isinstance(state, np.ndarray):
            if state.dtype != np.float32:
                state = state.astype(np.float32)

            if state.max() > 1.0:
                state = state / 255.0

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
        """Get training statistics with shaping analysis"""
        if not self.total_rewards:
            return {}

        stats = {
            'episodes': len(self.total_rewards),
            'steps': self.steps_done,
            'epsilon': self.epsilon,
            'avg_reward': np.mean(self.total_rewards[-100:]),
            'max_reward': np.max(self.total_rewards),
            'min_reward': np.min(self.total_rewards),
        }

        # Add shaping statistics
        if self.ablation_data['original_rewards']:
            shaping_analysis = self.get_shaping_analysis()
            stats.update({
                'shaping_improvement': shaping_analysis.get('reward_improvement', 0),
                'avg_lines_cleared': np.mean(self.ablation_data['lines_cleared'][-100:]) if self.ablation_data['lines_cleared'] else 0
            })

        return stats

# Modified train.py - Integration with reward shaping
"""
Enhanced training script with reward shaping integration and ablation studies
"""


# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


def parse_args():
    """Enhanced argument parsing with reward shaping options"""
    parser = argparse.ArgumentParser(
        description='Train Tetris AI with Reward Shaping')

    # Standard training arguments
    parser.add_argument('--episodes', type=int, default=MAX_EPISODES,
                        help=f'Number of episodes to train (default: {MAX_EPISODES})')
    parser.add_argument('--lr', type=float, default=LR,
                        help=f'Learning rate (default: {LR})')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help=f'Batch size (default: {BATCH_SIZE})')
    parser.add_argument('--gamma', type=float, default=GAMMA,
                        help=f'Discount factor (default: {GAMMA})')
    parser.add_argument('--model_type', type=str, default='dqn', choices=['dqn', 'dueling_dqn'],
                        help='Model architecture type')

    # Reward shaping arguments
    parser.add_argument('--reward_shaping', type=str, default='full',
                        choices=['none', 'simple', 'full'],
                        help='Type of reward shaping to use')
    parser.add_argument('--height_weight', type=float, default=-1.0,
                        help='Weight for height penalty (simple shaping)')
    parser.add_argument('--hole_weight', type=float, default=-2.0,
                        help='Weight for hole penalty (simple shaping)')

    # Training control
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from latest checkpoint')
    parser.add_argument('--no_render', action='store_true',
                        help='Disable rendering during training')
    parser.add_argument('--save_freq', type=int, default=100,
                        help='Save model every N episodes')
    parser.add_argument('--log_freq', type=int, default=10,
                        help='Log progress every N episodes')
    parser.add_argument('--eval_freq', type=int, default=50,
                        help='Evaluate model every N episodes')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name for this experiment')

    # Ablation study options
    parser.add_argument('--ablation_study', action='store_true',
                        help='Run ablation study comparing different shaping methods')

    return parser.parse_args()


def run_ablation_study(args):
    """Run comprehensive ablation study"""
    print("Running Reward Shaping Ablation Study")
    print("=" * 60)

    shaping_methods = [
        ('none', 'No reward shaping'),
        ('simple', 'Height + Hole penalties only'),
        ('full', 'Full multi-objective shaping')
    ]

    results = {}

    for method, description in shaping_methods:
        print(f"\nTesting: {description}")
        print("-" * 40)

        # Override shaping method
        args.reward_shaping = method
        args.episodes = 200  # Shorter for ablation
        args.experiment_name = f"ablation_{method}"

        # Run training
        result = train_single_configuration(args)
        results[method] = result

        print(f"Results for {method}:")
        print(f"  Final avg reward: {result['final_avg_reward']:.2f}")
        print(f"  Max reward: {result['max_reward']:.2f}")
        print(f"  Avg lines cleared: {result['avg_lines_cleared']:.2f}")
        print(f"  Training time: {result['training_time']:.1f}s")

    # Save ablation results
    ablation_file = os.path.join(
        LOG_DIR, f"ablation_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(ablation_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("ABLATION STUDY COMPLETE")
    print("=" * 60)
    print("Results summary:")
    for method, result in results.items():
        print(f"{method:10s}: Avg={result['final_avg_reward']:6.1f}, "
              f"Max={result['max_reward']:6.1f}, "
              f"Lines={result['avg_lines_cleared']:4.1f}")
    print(f"\nDetailed results saved to: {ablation_file}")


def train_single_configuration(args):
    """Train with a single configuration and return results"""
    start_time = time.time()

    # Create environment
    render_mode = None if args.no_render else "rgb_array"
    env = make_env(ENV_NAME, render_mode=render_mode)

    # Setup reward shaping configuration
    shaping_config = {}
    if args.reward_shaping == 'simple':
        shaping_config = {
            'height_weight': args.height_weight,
            'hole_weight': args.hole_weight
        }

    # Initialize agent with reward shaping
    agent = Agent(
        obs_space=env.observation_space,
        action_space=env.action_space,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        model_type=args.model_type,
        reward_shaping=args.reward_shaping,
        shaping_config=shaping_config
    )

    # Resume if requested
    start_episode = 0
    if args.resume:
        if agent.load_checkpoint(latest=True, model_dir=MODEL_DIR):
            start_episode = agent.episodes_done

    # Initialize logger
    experiment_name = args.experiment_name or f"tetris_{args.reward_shaping}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = TrainingLogger(LOG_DIR, experiment_name)

    # Training metrics tracking
    episode_rewards = []
    episode_lines_cleared = []
    original_rewards = []  # For comparison

    # Training loop
    for episode in range(start_episode, args.episodes):
        obs, info = env.reset()
        episode_reward = 0
        original_episode_reward = 0  # Track original reward
        episode_steps = 0
        lines_cleared_this_episode = 0

        done = False
        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Track original reward
            original_episode_reward += reward

            # Store experience with info for reward shaping
            agent.remember(obs, action, reward, next_obs, done, info, reward)

            # Learn
            learning_metrics = agent.learn()

            # Update tracking
            episode_reward += reward  # This will be shaped if shaping is enabled
            episode_steps += 1
            lines_cleared_this_episode += info.get('lines_cleared', 0)

            obs = next_obs

        # Episode end
        agent.end_episode(episode_reward, episode_steps,
                          lines_cleared_this_episode, original_episode_reward)

        # Track metrics
        episode_rewards.append(episode_reward)
        episode_lines_cleared.append(lines_cleared_this_episode)
        original_rewards.append(original_episode_reward)

        # Log episode
        logger.log_episode(
            episode=episode + 1,
            reward=episode_reward,
            steps=episode_steps,
            epsilon=agent.epsilon,
            lines_cleared=lines_cleared_this_episode,
            original_reward=original_episode_reward,
            shaping_improvement=episode_reward - original_episode_reward
        )

        # Periodic logging
        if (episode + 1) % args.log_freq == 0:
            stats = agent.get_stats()
            shaping_analysis = agent.get_shaping_analysis()

            print(f"Episode {episode+1:4d} | "
                  f"Reward: {episode_reward:6.1f} | "
                  f"Orig: {original_episode_reward:6.1f} | "
                  f"Avg: {stats['avg_reward']:6.1f} | "
                  f"Lines: {lines_cleared_this_episode:2d} | "
                  f"Epsilon: {agent.epsilon:.4f}")

            if shaping_analysis:
                improvement = shaping_analysis.get('reward_improvement', 0)
                print(f"         | Shaping improvement: {improvement:+5.1f}")

        # Periodic saves
        if (episode + 1) % args.save_freq == 0:
            agent.save_checkpoint(episode + 1, MODEL_DIR)
            logger.save_logs()
            logger.plot_progress()

    # Final save and cleanup
    agent.save_checkpoint(episode + 1, MODEL_DIR)
    logger.save_logs()
    logger.plot_progress()

    training_time = time.time() - start_time
    env.close()

    # Return results for ablation study
    return {
        'final_avg_reward': np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else np.mean(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'avg_lines_cleared': np.mean(episode_lines_cleared),
        'training_time': training_time,
        'shaping_analysis': agent.get_shaping_analysis(),
        'total_episodes': len(episode_rewards)
    }


def train(args):
    """Main training function with reward shaping"""
    print("Starting Tetris AI Training with Reward Shaping")
    print("=" * 60)
    print(f"Reward shaping method: {args.reward_shaping}")

    # Print system information
    print_system_info()

    # Create directories
    make_dir(MODEL_DIR)
    make_dir(LOG_DIR)

    if args.ablation_study:
        run_ablation_study(args)
    else:
        result = train_single_configuration(args)

        print("\n" + "=" * 60)
        print("TRAINING COMPLETED")
        print("=" * 60)
        print(f"Final average reward: {result['final_avg_reward']:.2f}")
        print(f"Maximum reward achieved: {result['max_reward']:.2f}")
        print(
            f"Average lines cleared per episode: {result['avg_lines_cleared']:.2f}")
        print(f"Total training time: {result['training_time']/3600:.2f} hours")

        # Print shaping analysis
        shaping_analysis = result['shaping_analysis']
        if shaping_analysis and args.reward_shaping != 'none':
            print("\nReward Shaping Analysis:")
            improvement = shaping_analysis.get('reward_improvement', 0)
            print(f"  Average reward improvement: {improvement:+.2f}")
            correlation = shaping_analysis.get('correlation', 0)
            print(f"  Original-Shaped correlation: {correlation:.3f}")


def main():
    """Main entry point"""
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
