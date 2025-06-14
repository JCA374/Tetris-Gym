import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from datetime import datetime
from pathlib import Path
import csv


def make_dir(path):
    """Create directory if it doesn't exist"""
    Path(path).mkdir(parents=True, exist_ok=True)


def save_config(config_dict, save_path):
    """Save configuration dictionary as JSON"""
    make_dir(os.path.dirname(save_path))
    with open(save_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"Configuration saved: {save_path}")


def load_config(config_path):
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def plot_rewards(reward_list, save_path, window_size=100, title="Training Progress"):
    """
    Plot training rewards with moving average
    
    Args:
        reward_list: List of episode rewards
        save_path: Path to save the plot
        window_size: Window size for moving average
        title: Plot title
    """
    make_dir(os.path.dirname(save_path))

    episodes = range(1, len(reward_list) + 1)

    # Calculate moving average
    if len(reward_list) >= window_size:
        moving_avg = []
        for i in range(len(reward_list)):
            start_idx = max(0, i - window_size + 1)
            moving_avg.append(np.mean(reward_list[start_idx:i+1]))
    else:
        moving_avg = reward_list

    # Create plot
    plt.figure(figsize=(12, 6))

    # Plot raw rewards
    plt.subplot(1, 2, 1)
    plt.plot(episodes, reward_list, alpha=0.6,
             color='lightblue', label='Episode Reward')
    plt.plot(episodes, moving_avg, color='darkblue', linewidth=2,
             label=f'Moving Average ({window_size} episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'{title} - Rewards')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot reward distribution
    plt.subplot(1, 2, 2)
    plt.hist(reward_list, bins=30, alpha=0.7, color='green', edgecolor='black')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.title('Reward Distribution')
    plt.axvline(np.mean(reward_list), color='red', linestyle='--',
                label=f'Mean: {np.mean(reward_list):.1f}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Reward plot saved: {save_path}")


def plot_training_metrics(metrics_dict, save_path, title="Training Metrics"):
    """
    Plot multiple training metrics
    
    Args:
        metrics_dict: Dictionary with metric names as keys and lists as values
        save_path: Path to save the plot
        title: Plot title
    """
    make_dir(os.path.dirname(save_path))

    n_metrics = len(metrics_dict)
    if n_metrics == 0:
        return

    # Calculate subplot layout
    cols = min(3, n_metrics)
    rows = (n_metrics + cols - 1) // cols

    plt.figure(figsize=(5*cols, 4*rows))

    for i, (metric_name, values) in enumerate(metrics_dict.items(), 1):
        plt.subplot(rows, cols, i)
        episodes = range(1, len(values) + 1)
        plt.plot(episodes, values, linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel(metric_name)
        plt.title(f'{metric_name} over Time')
        plt.grid(True, alpha=0.3)

        # Add statistics
        mean_val = np.mean(values)
        plt.axhline(mean_val, color='red', linestyle='--', alpha=0.7,
                    label=f'Mean: {mean_val:.3f}')
        plt.legend()

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Training metrics plot saved: {save_path}")


def save_training_log(log_data, save_path):
    """
    Save training log as CSV
    
    Args:
        log_data: List of dictionaries with training data
        save_path: Path to save CSV file
    """
    make_dir(os.path.dirname(save_path))

    if not log_data:
        return

    # Get all unique keys from all log entries
    fieldnames = set()
    for entry in log_data:
        fieldnames.update(entry.keys())
    fieldnames = sorted(list(fieldnames))

    with open(save_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(log_data)

    print(f"Training log saved: {save_path}")


def load_training_log(log_path):
    """Load training log from CSV"""
    log_data = []
    with open(log_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric strings back to numbers
            processed_row = {}
            for key, value in row.items():
                try:
                    processed_row[key] = float(value)
                except ValueError:
                    processed_row[key] = value
            log_data.append(processed_row)

    return log_data


class TrainingLogger:
    """Comprehensive training logger"""

    def __init__(self, log_dir, experiment_name=None):
        self.log_dir = Path(log_dir)
        make_dir(self.log_dir)

        if experiment_name is None:
            experiment_name = f"tetris_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.experiment_name = experiment_name
        self.experiment_dir = self.log_dir / experiment_name
        make_dir(self.experiment_dir)

        # Initialize log storage
        self.episode_logs = []
        self.step_logs = []

        print(f"Training logger initialized: {self.experiment_dir}")

    def log_episode(self, episode, reward, steps, epsilon, **kwargs):
        """Log episode-level metrics"""
        log_entry = {
            'episode': episode,
            'reward': reward,
            'steps': steps,
            'epsilon': epsilon,
            'timestamp': datetime.now().isoformat(),
            **kwargs
        }
        self.episode_logs.append(log_entry)

    def log_step(self, step, loss, q_value, **kwargs):
        """Log step-level metrics"""
        log_entry = {
            'step': step,
            'loss': loss,
            'q_value': q_value,
            'timestamp': datetime.now().isoformat(),
            **kwargs
        }
        self.step_logs.append(log_entry)

    def save_logs(self):
        """Save all logs to files"""
        # Save episode logs
        if self.episode_logs:
            episode_log_path = self.experiment_dir / "episode_log.csv"
            save_training_log(self.episode_logs, episode_log_path)

        # Save step logs (save only recent ones to avoid huge files)
        if self.step_logs:
            recent_steps = self.step_logs[-10000:]  # Keep last 10k steps
            step_log_path = self.experiment_dir / "step_log.csv"
            save_training_log(recent_steps, step_log_path)

    def plot_progress(self):
        """Generate and save progress plots"""
        if not self.episode_logs:
            return

        # Extract rewards
        rewards = [log['reward'] for log in self.episode_logs]

        # Plot rewards
        reward_plot_path = self.experiment_dir / "reward_progress.png"
        plot_rewards(rewards, reward_plot_path,
                     title=f"Training Progress - {self.experiment_name}")

        # Plot other metrics
        metrics = {}
        for key in ['epsilon', 'steps']:
            if key in self.episode_logs[0]:
                metrics[key] = [log[key] for log in self.episode_logs]

        if metrics:
            metrics_plot_path = self.experiment_dir / "training_metrics.png"
            plot_training_metrics(metrics, metrics_plot_path,
                                  title=f"Training Metrics - {self.experiment_name}")

    def get_summary(self):
        """Get training summary statistics"""
        if not self.episode_logs:
            return {}

        rewards = [log['reward'] for log in self.episode_logs]

        summary = {
            'total_episodes': len(self.episode_logs),
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'max_reward': np.max(rewards),
            'min_reward': np.min(rewards),
            'final_epsilon': self.episode_logs[-1].get('epsilon', 'N/A'),
        }

        # Recent performance (last 100 episodes)
        if len(rewards) >= 100:
            recent_rewards = rewards[-100:]
            summary['recent_mean_reward'] = np.mean(recent_rewards)
            summary['recent_std_reward'] = np.std(recent_rewards)

        return summary


def print_system_info():
    """Print system information for debugging"""
    print("=" * 50)
    print("SYSTEM INFORMATION")
    print("=" * 50)

    # PyTorch info
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(
            f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Environment info
    try:
        import gymnasium
        print(f"Gymnasium version: {gymnasium.__version__}")
    except ImportError:
        print("Gymnasium not installed")

    try:
        import tetris_gymnasium
        print("Tetris Gymnasium: Available")
    except ImportError:
        print("Tetris Gymnasium: Not installed")

    print("=" * 50)


def benchmark_environment(env, n_steps=1000):
    """Benchmark environment performance"""
    print(f"Benchmarking environment for {n_steps} steps...")

    import time

    # Reset timing
    start_time = time.time()
    obs, info = env.reset()
    reset_time = time.time() - start_time

    # Step timing
    step_times = []
    for i in range(n_steps):
        action = env.action_space.sample()

        step_start = time.time()
        obs, reward, terminated, truncated, info = env.step(action)
        step_time = time.time() - step_start

        step_times.append(step_time)

        if terminated or truncated:
            env.reset()

    avg_step_time = np.mean(step_times)

    print(f"Environment Benchmark Results:")
    print(f"  Reset time: {reset_time*1000:.2f} ms")
    print(f"  Average step time: {avg_step_time*1000:.2f} ms")
    print(f"  Steps per second: {1/avg_step_time:.0f}")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Action space size: {env.action_space.n}")

    return {
        'reset_time': reset_time,
        'avg_step_time': avg_step_time,
        'steps_per_second': 1/avg_step_time,
        'obs_shape': obs.shape,
        'action_space_size': env.action_space.n
    }


if __name__ == "__main__":
    # Test utilities
    print("Testing utilities...")

    # Test plotting
    dummy_rewards = np.random.randn(200).cumsum() + 100
    plot_rewards(dummy_rewards, "test_plots/reward_test.png")

    # Test logger
    logger = TrainingLogger("test_logs", "test_experiment")
    for i in range(10):
        logger.log_episode(i, dummy_rewards[i], 100, 0.5)
    logger.save_logs()
    logger.plot_progress()

    print("✅ Utilities test completed!")
