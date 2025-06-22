#!/usr/bin/env python3
"""
Enhanced Tetris AI Diagnostics with Professional Analysis Tools
Implements all the suggested improvements for robust debugging
"""

from src.agent import Agent
from config import make_env
import sys
import os
import argparse
import logging
import json
import random
import time
from datetime import datetime
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Setup logging


def setup_logging(level=logging.INFO):
    """Setup structured logging"""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    )
    return logging.getLogger(__name__)


def parse_args():
    """Enhanced argument parsing for diagnostics"""
    parser = argparse.ArgumentParser(
        description='Enhanced Tetris AI Diagnostics')

    # Core parameters
    parser.add_argument("--episodes", type=int, default=50,
                        help="Number of episodes to analyze")
    parser.add_argument("--max-steps", type=int, default=5000,
                        help="Maximum steps per episode")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    # Environment options
    parser.add_argument("--render", action="store_true",
                        help="Enable visual rendering")
    parser.add_argument("--reward-shaping", choices=["none", "simple", "full"],
                        default="simple", help="Reward shaping method")
    parser.add_argument("--model-type", choices=["dqn", "dueling_dqn"],
                        default="dqn", help="Model architecture")

    # Analysis options
    parser.add_argument("--analyze-board", action="store_true",
                        help="Detailed board state analysis")
    parser.add_argument("--analyze-actions", action="store_true",
                        help="Action distribution analysis")
    parser.add_argument("--analyze-q-values", action="store_true",
                        help="Q-value statistics analysis")

    # Output options
    parser.add_argument("--output-dir", type=str, default="diagnostics_output",
                        help="Output directory for results")
    parser.add_argument("--output-csv", type=str, default="tetris_diagnostics.csv",
                        help="CSV output filename")
    parser.add_argument("--save-plots", action="store_true",
                        help="Save diagnostic plots")

    # Logging
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")

    return parser.parse_args()


def set_seeds(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class TetrisBoardAnalyzer:
    """Analyze Tetris board states for diagnostic insights"""

    def __init__(self):
        self.board_height = 20
        self.board_width = 10

    def analyze_board_state(self, board):
        """Compute comprehensive board metrics"""
        if isinstance(board, dict) and 'board' in board:
            board = board['board']

        # Convert to numpy if needed
        if hasattr(board, 'cpu'):
            board = board.cpu().numpy()
        board = np.array(board)

        # Handle different board formats
        if len(board.shape) > 2:
            # Extract 2D board from processed observations
            if board.shape[-1] == 1:  # (H, W, 1)
                board = board.squeeze(-1)
            elif len(board.shape) == 3:  # Frame stacked
                board = board[:, :, -1]  # Use most recent frame

        # Ensure we have a 2D board
        if len(board.shape) != 2:
            return self._empty_metrics()

        try:
            # Column heights
            heights = []
            for col in range(min(board.shape[1], self.board_width)):
                height = 0
                for row in range(board.shape[0]):
                    if board[row, col] != 0:
                        height = board.shape[0] - row
                        break
                heights.append(height)

            # Pad if needed
            while len(heights) < self.board_width:
                heights.append(0)
            heights = heights[:self.board_width]

            # Calculate metrics
            max_height = max(heights) if heights else 0
            avg_height = np.mean(heights) if heights else 0
            height_variance = np.var(heights) if heights else 0

            # Holes (empty cells with filled cells above)
            holes = 0
            for col in range(min(board.shape[1], self.board_width)):
                found_filled = False
                for row in range(board.shape[0]):
                    if board[row, col] != 0:
                        found_filled = True
                    elif found_filled:
                        holes += 1

            # Bumpiness (height differences)
            bumpiness = sum(abs(heights[i] - heights[i+1])
                            for i in range(len(heights)-1))

            # Wells (deep single-column gaps)
            wells = 0
            for i in range(len(heights)):
                left_height = heights[i-1] if i > 0 else heights[i]
                right_height = heights[i +
                                       1] if i < len(heights)-1 else heights[i]
                well_depth = min(left_height, right_height) - heights[i]
                wells += max(0, well_depth)

            return {
                'max_height': max_height,
                'avg_height': avg_height,
                'height_variance': height_variance,
                'holes': holes,
                'bumpiness': bumpiness,
                'wells': wells,
                'filled_cells': np.sum(board != 0),
                'board_fullness': np.sum(board != 0) / board.size
            }

        except Exception as e:
            logging.warning(f"Board analysis failed: {e}")
            return self._empty_metrics()

    def _empty_metrics(self):
        """Return empty metrics dict"""
        return {
            'max_height': 0, 'avg_height': 0, 'height_variance': 0,
            'holes': 0, 'bumpiness': 0, 'wells': 0,
            'filled_cells': 0, 'board_fullness': 0
        }


class ActionAnalyzer:
    """Analyze action distributions and patterns"""

    ACTION_NAMES = {
        0: 'NO-OP', 1: 'RIGHT', 2: 'LEFT', 3: 'DOWN',
        4: 'ROTATE_CW', 5: 'ROTATE_CCW', 6: 'HARD_DROP', 7: 'HOLD'
    }

    def __init__(self):
        self.action_counts = defaultdict(int)
        self.action_sequences = []
        self.recent_actions = deque(maxlen=10)

    def record_action(self, action):
        """Record an action for analysis"""
        self.action_counts[action] += 1
        self.recent_actions.append(action)

        # Check for repetitive patterns
        if len(self.recent_actions) == 10:
            self.action_sequences.append(list(self.recent_actions))

    def get_action_distribution(self):
        """Get action distribution statistics"""
        total_actions = sum(self.action_counts.values())
        if total_actions == 0:
            return {}

        distribution = {}
        for action, count in self.action_counts.items():
            action_name = self.ACTION_NAMES.get(action, f"ACTION_{action}")
            distribution[action_name] = {
                'count': count,
                'percentage': (count / total_actions) * 100
            }

        return distribution

    def detect_patterns(self):
        """Detect repetitive action patterns"""
        patterns = defaultdict(int)

        for sequence in self.action_sequences:
            # Look for repeated actions
            for i in range(len(sequence) - 2):
                if sequence[i] == sequence[i+1] == sequence[i+2]:
                    action_name = self.ACTION_NAMES.get(
                        sequence[i], f"ACTION_{sequence[i]}")
                    patterns[f"Triple_{action_name}"] += 1

        return dict(patterns)


def run_diagnostic_episode(env, agent, board_analyzer, action_analyzer,
                           episode_num, max_steps, analyze_q_values, log):
    """Run a single diagnostic episode with comprehensive analysis"""

    obs, info = env.reset()
    episode_metrics = {
        'episode': episode_num,
        'total_reward': 0,
        'original_reward': 0,
        'steps': 0,
        'lines_cleared': 0,
        'pieces_placed': 0,
        'game_over': False
    }

    # Q-value tracking
    q_values_list = []

    # Board state tracking
    board_states = []

    step = 0
    done = False

    while not done and step < max_steps:
        # Action selection and Q-value analysis
        if analyze_q_values and agent:
            with torch.no_grad():
                state_tensor = agent._preprocess_state(obs)
                q_values = agent.q_network(state_tensor)
                q_values_np = q_values.cpu().numpy().flatten()
                q_values_list.append(q_values_np)

                action = q_values_np.argmax()
        else:
            action = env.action_space.sample()

        # Record action
        action_analyzer.record_action(action)

        # Take step
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Track rewards separately
        episode_metrics['total_reward'] += reward
        episode_metrics['original_reward'] += reward  # Same for now

        # Board analysis
        board_metrics = board_analyzer.analyze_board_state(obs)
        board_metrics['step'] = step
        board_states.append(board_metrics)

        # Game metrics
        episode_metrics['lines_cleared'] += info.get('lines_cleared', 0)
        if info.get('lines_cleared', 0) > 0:
            episode_metrics['pieces_placed'] += 1

        obs = next_obs
        step += 1

    episode_metrics['steps'] = step
    episode_metrics['game_over'] = done

    # Calculate episode-level board metrics
    if board_states:
        final_board = board_states[-1]
        episode_metrics.update({
            'final_max_height': final_board['max_height'],
            'final_holes': final_board['holes'],
            'final_bumpiness': final_board['bumpiness'],
            'avg_board_fullness': np.mean([bs['board_fullness'] for bs in board_states])
        })

    # Q-value statistics
    if q_values_list:
        all_q_values = np.concatenate(q_values_list)
        episode_metrics.update({
            'q_mean': float(all_q_values.mean()),
            'q_std': float(all_q_values.std()),
            'q_max': float(all_q_values.max()),
            'q_min': float(all_q_values.min()),
        })

    # Log episode summary
    log.info(f"Episode {episode_num:3d} | "
             f"Reward: {episode_metrics['total_reward']:6.1f} | "
             f"Steps: {episode_metrics['steps']:4d} | "
             f"Lines: {episode_metrics['lines_cleared']:2d} | "
             f"Height: {episode_metrics.get('final_max_height', 0):2.0f} | "
             f"Holes: {episode_metrics.get('final_holes', 0):2d}")

    return episode_metrics, board_states


def create_diagnostic_plots(results_df, action_analyzer, output_dir, save_plots):
    """Create comprehensive diagnostic plots"""
    if not save_plots:
        return

    os.makedirs(output_dir, exist_ok=True)

    # 1. Training progress plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Tetris AI Diagnostic Analysis', fontsize=16)

    # Rewards over time
    axes[0, 0].plot(results_df['episode'], results_df['total_reward'])
    axes[0, 0].set_title('Reward Progress')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].grid(True, alpha=0.3)

    # Lines cleared over time
    axes[0, 1].plot(results_df['episode'], results_df['lines_cleared'])
    axes[0, 1].set_title('Lines Cleared Progress')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Lines Cleared')
    axes[0, 1].grid(True, alpha=0.3)

    # Board health metrics
    if 'final_max_height' in results_df.columns:
        axes[0, 2].plot(results_df['episode'], results_df['final_max_height'],
                        label='Max Height', alpha=0.7)
        axes[0, 2].plot(results_df['episode'], results_df['final_holes'],
                        label='Holes', alpha=0.7)
        axes[0, 2].set_title('Board Health')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Metric Value')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

    # Q-value statistics (if available)
    if 'q_mean' in results_df.columns:
        axes[1, 0].plot(results_df['episode'], results_df['q_mean'])
        axes[1, 0].fill_between(results_df['episode'],
                                results_df['q_mean'] - results_df['q_std'],
                                results_df['q_mean'] + results_df['q_std'],
                                alpha=0.3)
        axes[1, 0].set_title('Q-Value Distribution')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Q-Value')
        axes[1, 0].grid(True, alpha=0.3)

    # Episode length distribution
    axes[1, 1].hist(results_df['steps'], bins=20, alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Episode Length Distribution')
    axes[1, 1].set_xlabel('Steps')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)

    # Performance correlation
    if results_df['lines_cleared'].sum() > 0:
        axes[1, 2].scatter(results_df['total_reward'],
                           results_df['lines_cleared'], alpha=0.6)
        axes[1, 2].set_title('Reward vs Lines Cleared')
        axes[1, 2].set_xlabel('Total Reward')
        axes[1, 2].set_ylabel('Lines Cleared')
        axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tetris_diagnostics_overview.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Action distribution plot
    action_dist = action_analyzer.get_action_distribution()
    if action_dist:
        fig, ax = plt.subplots(figsize=(10, 6))

        actions = list(action_dist.keys())
        counts = [action_dist[action]['count'] for action in actions]
        percentages = [action_dist[action]['percentage'] for action in actions]

        bars = ax.bar(actions, counts)
        ax.set_title('Action Distribution')
        ax.set_xlabel('Action')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)

        # Add percentage labels
        for bar, pct in zip(bars, percentages):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{pct:.1f}%', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'action_distribution.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()


def main():
    """Main diagnostic function"""
    args = parse_args()

    # Setup
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log = setup_logging(log_level)

    log.info("🔍 Starting Enhanced Tetris AI Diagnostics")
    log.info(f"Configuration: {vars(args)}")

    # Set seeds for reproducibility
    set_seeds(args.seed)
    log.info(f"Random seed set to: {args.seed}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize environment
    try:
        render_mode = "human" if args.render else "rgb_array"
        env = make_env(render_mode=render_mode, frame_stack=4)
        log.info(f"Environment created successfully")
        log.info(f"Observation space: {env.observation_space}")
        log.info(f"Action space: {env.action_space}")
    except Exception as e:
        log.error(f"Failed to create environment: {e}")
        return 1

    # Initialize agent (if available)
    agent = None
    if args.analyze_q_values:
        try:
            agent = Agent(
                obs_space=env.observation_space,
                action_space=env.action_space,
                model_type=args.model_type,
                reward_shaping=args.reward_shaping
            )
            log.info(f"Agent initialized for Q-value analysis")
        except Exception as e:
            log.warning(f"Could not initialize agent: {e}")
            args.analyze_q_values = False

    # Initialize analyzers
    board_analyzer = TetrisBoardAnalyzer()
    action_analyzer = ActionAnalyzer()

    # Run diagnostic episodes
    log.info(f"Running {args.episodes} diagnostic episodes...")

    all_results = []
    start_time = time.time()

    for episode in range(args.episodes):
        episode_metrics, board_states = run_diagnostic_episode(
            env, agent, board_analyzer, action_analyzer,
            episode + 1, args.max_steps, args.analyze_q_values, log
        )

        all_results.append(episode_metrics)

    total_time = time.time() - start_time

    # Analysis and output
    log.info(f"Diagnostic run completed in {total_time:.2f} seconds")

    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)

    # Save CSV
    csv_path = os.path.join(args.output_dir, args.output_csv)
    results_df.to_csv(csv_path, index=False)
    log.info(f"Results saved to: {csv_path}")

    # Generate summary statistics
    summary_stats = {
        'total_episodes': len(all_results),
        'total_time': total_time,
        'avg_reward': results_df['total_reward'].mean(),
        'std_reward': results_df['total_reward'].std(),
        'max_reward': results_df['total_reward'].max(),
        'min_reward': results_df['total_reward'].min(),
        'total_lines_cleared': results_df['lines_cleared'].sum(),
        'avg_lines_per_episode': results_df['lines_cleared'].mean(),
        'episodes_with_lines': (results_df['lines_cleared'] > 0).sum(),
        'avg_episode_length': results_df['steps'].mean(),
        'seed_used': args.seed,
        'configuration': vars(args)
    }

    # Add Q-value stats if available
    if 'q_mean' in results_df.columns:
        summary_stats.update({
            'avg_q_mean': results_df['q_mean'].mean(),
            'avg_q_std': results_df['q_std'].mean(),
            'overall_q_range': results_df['q_max'].max() - results_df['q_min'].min()
        })

    # Action analysis
    action_dist = action_analyzer.get_action_distribution()
    patterns = action_analyzer.detect_patterns()

    summary_stats.update({
        'action_distribution': action_dist,
        'action_patterns': patterns
    })

    # Save summary
    summary_path = os.path.join(args.output_dir, 'diagnostic_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary_stats, f, indent=2, default=str)

    # Print summary
    log.info("=" * 60)
    log.info("DIAGNOSTIC SUMMARY")
    log.info("=" * 60)
    log.info(f"Episodes analyzed: {summary_stats['total_episodes']}")
    log.info(
        f"Average reward: {summary_stats['avg_reward']:.2f} ± {summary_stats['std_reward']:.2f}")
    log.info(f"Total lines cleared: {summary_stats['total_lines_cleared']}")
    log.info(
        f"Episodes with lines: {summary_stats['episodes_with_lines']}/{summary_stats['total_episodes']}")
    log.info(
        f"Average episode length: {summary_stats['avg_episode_length']:.1f} steps")

    if action_dist:
        log.info(
            f"Most common action: {max(action_dist.items(), key=lambda x: x[1]['count'])[0]}")

    if args.analyze_q_values and 'avg_q_mean' in summary_stats:
        log.info(f"Average Q-value: {summary_stats['avg_q_mean']:.3f}")

    # Generate plots
    create_diagnostic_plots(results_df, action_analyzer,
                            args.output_dir, args.save_plots)

    # Cleanup
    env.close()

    log.info(f"All results saved to: {args.output_dir}")
    log.info("🎉 Diagnostic analysis complete!")

    return 0


if __name__ == "__main__":
    exit(main())
