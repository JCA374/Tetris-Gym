# src/utils.py
"""Utility functions for Tetris AI training"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def make_dir(path):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)
    return path


def save_json(data, filepath):
    """Save data to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(filepath):
    """Load data from JSON file"""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None


def plot_training_curves(episode_data, save_path=None):
    """Plot training curves"""
    if not episode_data:
        print("No data to plot")
        return
    
    episodes = [d['episode'] for d in episode_data]
    rewards = [d['reward'] for d in episode_data]
    lines = [d.get('lines', 0) for d in episode_data]
    steps = [d.get('steps', 0) for d in episode_data]
    epsilon = [d.get('epsilon', 0) for d in episode_data]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Rewards
    axes[0, 0].plot(episodes, rewards, alpha=0.6)
    axes[0, 0].plot(episodes, moving_average(rewards, 100), linewidth=2)
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True)
    axes[0, 0].legend(['Raw', 'Moving Avg (100)'])
    
    # Lines cleared
    axes[0, 1].plot(episodes, lines, alpha=0.6)
    axes[0, 1].plot(episodes, moving_average(lines, 100), linewidth=2)
    axes[0, 1].set_title('Lines Cleared per Episode')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Lines')
    axes[0, 1].grid(True)
    axes[0, 1].legend(['Raw', 'Moving Avg (100)'])
    
    # Steps per episode
    axes[1, 0].plot(episodes, steps, alpha=0.6)
    axes[1, 0].plot(episodes, moving_average(steps, 100), linewidth=2)
    axes[1, 0].set_title('Steps per Episode')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Steps')
    axes[1, 0].grid(True)
    axes[1, 0].legend(['Raw', 'Moving Avg (100)'])
    
    # Epsilon
    axes[1, 1].plot(episodes, epsilon)
    axes[1, 1].set_title('Exploration Rate (Epsilon)')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Epsilon')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def moving_average(values, window):
    """Calculate moving average"""
    if len(values) < window:
        return values
    
    weights = np.ones(window) / window
    return np.convolve(values, weights, mode='valid').tolist()


def format_time(seconds):
    """Format time in seconds to human-readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def get_timestamp():
    """Get current timestamp string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def print_board(board, clear=True):
    """Print the Tetris board in a nice format"""
    if clear:
        os.system('clear' if os.name == 'posix' else 'cls')
    
    print("┌" + "─" * (board.shape[1] * 2) + "┐")
    
    for row in board:
        print("│", end="")
        for cell in row:
            if cell > 0:
                print("██", end="")
            else:
                print("  ", end="")
        print("│")
    
    print("└" + "─" * (board.shape[1] * 2) + "┘")


def analyze_board_state(board):
    """Analyze and return statistics about the board state"""
    from src.reward_shaping import (
        get_column_heights,
        count_holes,
        calculate_bumpiness,
        get_max_height,
        get_horizontal_distribution
    )
    
    stats = {
        'max_height': get_max_height(board),
        'column_heights': get_column_heights(board),
        'holes': count_holes(board),
        'bumpiness': calculate_bumpiness(board),
        'distribution': get_horizontal_distribution(board),
        'filled_cells': np.sum(board > 0),
        'empty_cells': np.sum(board == 0),
    }
    
    # Check for full rows
    full_rows = []
    for i, row in enumerate(board):
        if np.all(row > 0):
            full_rows.append(i)
    stats['full_rows'] = full_rows
    
    return stats