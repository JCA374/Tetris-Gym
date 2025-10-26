# src/training_logger.py
"""Training logger for Tetris AI"""

import os
import json
import csv
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from src.utils import make_dir, moving_average


class TrainingLogger:
    """Logger for training metrics and visualization"""
    
    def __init__(self, log_dir, experiment_name=None):
        """Initialize logger"""
        self.log_dir = make_dir(log_dir)
        
        # Create experiment folder
        if experiment_name is None:
            experiment_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.experiment_name = experiment_name
        self.experiment_dir = make_dir(os.path.join(log_dir, experiment_name))
        
        # File paths
        self.metrics_file = os.path.join(self.experiment_dir, "metrics.json")
        self.csv_file = os.path.join(self.experiment_dir, "episodes.csv")
        self.plot_file = os.path.join(self.experiment_dir, "training_curves.png")
        self.config_file = os.path.join(self.experiment_dir, "config.json")
        
        # Data storage
        self.episode_data = []
        self.config = {}
        
        # Initialize CSV file
        self._init_csv()
        
        print(f"📊 Logging to: {self.experiment_dir}")
    
    def _init_csv(self):
        """Initialize CSV file with headers"""
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'episode', 'reward', 'steps', 'epsilon', 
                'lines_cleared', 'total_lines', 'avg_reward',
                'avg_steps', 'avg_lines', 'timestamp'
            ])
    
    def log_config(self, config):
        """Log training configuration"""
        self.config = config
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def log_episode(self, episode, reward, steps, epsilon, lines_cleared=0,
                    original_reward=None, total_lines=0, shaped_reward_used=False):
        """Log episode data"""
        # Calculate moving averages
        recent_episodes = self.episode_data[-100:] if len(self.episode_data) >= 100 else self.episode_data
        
        avg_reward = np.mean([e['reward'] for e in recent_episodes]) if recent_episodes else reward
        avg_steps = np.mean([e['steps'] for e in recent_episodes]) if recent_episodes else steps
        avg_lines = np.mean([e.get('lines', 0) for e in recent_episodes]) if recent_episodes else lines_cleared
        
        # Create episode record
        episode_record = {
            'episode': episode,
            'reward': float(reward),
            'steps': int(steps),
            'epsilon': float(epsilon),
            'lines': int(lines_cleared),
            'total_lines': int(total_lines),
            'avg_reward': float(avg_reward),
            'avg_steps': float(avg_steps),
            'avg_lines': float(avg_lines),
            'timestamp': datetime.now().isoformat(),
            'shaped_reward_used': shaped_reward_used
        }
        
        if original_reward is not None:
            episode_record['original_reward'] = float(original_reward)
        
        # Store data
        self.episode_data.append(episode_record)
        
        # Write to CSV
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                episode, reward, steps, epsilon, lines_cleared,
                total_lines, avg_reward, avg_steps, avg_lines,
                episode_record['timestamp']
            ])
    
    def save_logs(self):
        """Save all logs to disk"""
        # Save metrics
        with open(self.metrics_file, 'w') as f:
            json.dump({
                'experiment_name': self.experiment_name,
                'config': self.config,
                'episodes': self.episode_data,
                'summary': self._get_summary()
            }, f, indent=2)
    
    def _get_summary(self):
        """Get training summary statistics"""
        if not self.episode_data:
            return {}
        
        all_rewards = [e['reward'] for e in self.episode_data]
        all_steps = [e['steps'] for e in self.episode_data]
        all_lines = [e.get('lines', 0) for e in self.episode_data]
        
        # Get recent performance (last 100 episodes)
        recent = self.episode_data[-100:] if len(self.episode_data) >= 100 else self.episode_data
        recent_rewards = [e['reward'] for e in recent]
        recent_steps = [e['steps'] for e in recent]
        recent_lines = [e.get('lines', 0) for e in recent]
        
        summary = {
            'total_episodes': len(self.episode_data),
            'total_lines_cleared': sum(all_lines),
            
            'all_time': {
                'best_reward': max(all_rewards),
                'worst_reward': min(all_rewards),
                'avg_reward': np.mean(all_rewards),
                'std_reward': np.std(all_rewards),
                'best_lines': max(all_lines),
                'avg_lines': np.mean(all_lines),
                'avg_steps': np.mean(all_steps),
            },
            
            'recent_100': {
                'avg_reward': np.mean(recent_rewards) if recent_rewards else 0,
                'std_reward': np.std(recent_rewards) if recent_rewards else 0,
                'avg_lines': np.mean(recent_lines) if recent_lines else 0,
                'avg_steps': np.mean(recent_steps) if recent_steps else 0,
            }
        }
        
        # Find first line clear
        for i, e in enumerate(self.episode_data):
            if e.get('lines', 0) > 0:
                summary['first_line_episode'] = e['episode']
                break
        
        return summary
    
    def plot_progress(self, save=True):
        """Plot training progress"""
        if len(self.episode_data) < 2:
            return
        
        episodes = [d['episode'] for d in self.episode_data]
        rewards = [d['reward'] for d in self.episode_data]
        lines = [d.get('lines', 0) for d in self.episode_data]
        steps = [d.get('steps', 0) for d in self.episode_data]
        epsilon = [d.get('epsilon', 0) for d in self.episode_data]
        
        # Create figure
        fig = plt.figure(figsize=(15, 10))
        
        # Rewards subplot
        ax1 = plt.subplot(2, 3, 1)
        ax1.plot(episodes, rewards, alpha=0.3, color='blue')
        if len(rewards) >= 100:
            ax1.plot(episodes[99:], moving_average(rewards, 100), 
                    color='blue', linewidth=2, label='MA-100')
        ax1.set_title('Episode Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Lines cleared subplot
        ax2 = plt.subplot(2, 3, 2)
        ax2.plot(episodes, lines, alpha=0.3, color='green')
        if len(lines) >= 100:
            ax2.plot(episodes[99:], moving_average(lines, 100),
                    color='green', linewidth=2, label='MA-100')
        ax2.set_title('Lines Cleared')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Lines')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Steps per episode subplot
        ax3 = plt.subplot(2, 3, 3)
        ax3.plot(episodes, steps, alpha=0.3, color='orange')
        if len(steps) >= 100:
            ax3.plot(episodes[99:], moving_average(steps, 100),
                    color='orange', linewidth=2, label='MA-100')
        ax3.set_title('Episode Length')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Steps')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Epsilon subplot
        ax4 = plt.subplot(2, 3, 4)
        ax4.plot(episodes, epsilon, color='red', linewidth=2)
        ax4.set_title('Exploration Rate')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Epsilon')
        ax4.grid(True, alpha=0.3)
        
        # Cumulative lines subplot
        ax5 = plt.subplot(2, 3, 5)
        cumulative_lines = np.cumsum(lines)
        ax5.plot(episodes, cumulative_lines, color='purple', linewidth=2)
        ax5.set_title('Total Lines Cleared')
        ax5.set_xlabel('Episode')
        ax5.set_ylabel('Cumulative Lines')
        ax5.grid(True, alpha=0.3)
        
        # Performance over time subplot
        ax6 = plt.subplot(2, 3, 6)
        if len(self.episode_data) >= 100:
            # Calculate lines per 100 episodes
            lines_per_100 = []
            for i in range(100, len(self.episode_data) + 1, 100):
                chunk_lines = sum(d.get('lines', 0) for d in self.episode_data[i-100:i])
                lines_per_100.append(chunk_lines)
            
            if lines_per_100:
                x_vals = list(range(100, len(lines_per_100) * 100 + 1, 100))
                ax6.bar(x_vals, lines_per_100, width=80, color='teal', alpha=0.7)
                ax6.set_title('Lines per 100 Episodes')
                ax6.set_xlabel('Episode')
                ax6.set_ylabel('Lines Cleared')
                ax6.grid(True, alpha=0.3)
        
        plt.suptitle(f'Training Progress - {self.experiment_name}', fontsize=14)
        plt.tight_layout()
        
        if save:
            plt.savefig(self.plot_file, dpi=150, bbox_inches='tight')
            print(f"  📈 Plot saved to {self.plot_file}")
        else:
            plt.show()
        
        plt.close()
    
    def print_summary(self):
        """Print training summary"""
        summary = self._get_summary()
        
        if not summary:
            print("No data to summarize")
            return
        
        print("\n" + "="*60)
        print(f"TRAINING SUMMARY - {self.experiment_name}")
        print("="*60)
        
        print(f"\nTotal Episodes: {summary['total_episodes']}")
        print(f"Total Lines Cleared: {summary['total_lines_cleared']}")
        
        if 'first_line_episode' in summary:
            print(f"First Line Cleared: Episode {summary['first_line_episode']}")
        
        print(f"\nAll-Time Performance:")
        print(f"  Best Reward: {summary['all_time']['best_reward']:.1f}")
        print(f"  Avg Reward: {summary['all_time']['avg_reward']:.1f}")
        print(f"  Best Lines: {summary['all_time']['best_lines']}")
        print(f"  Avg Lines: {summary['all_time']['avg_lines']:.2f}")
        
        if summary['total_episodes'] >= 100:
            print(f"\nRecent Performance (last 100):")
            print(f"  Avg Reward: {summary['recent_100']['avg_reward']:.1f}")
            print(f"  Avg Lines: {summary['recent_100']['avg_lines']:.2f}")
            print(f"  Avg Steps: {summary['recent_100']['avg_steps']:.1f}")