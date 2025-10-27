# src/reward_shaping.py
"""Fixed reward shaping - ignores environment's negative base rewards"""

import numpy as np


def extract_board_from_obs(obs):
    """Extract and normalize board from observation"""
    if len(obs.shape) == 3:
        board = obs[:, :, 0]
    else:
        board = obs
    
    # Normalize to 0-1
    board = (board > 0).astype(np.float32)
    return board


def get_column_heights(board):
    """Get height of each column"""
    heights = []
    for col in range(board.shape[1]):
        occupied = np.where(board[:, col] > 0)[0]
        height = board.shape[0] - occupied[0] if len(occupied) > 0 else 0
        heights.append(height)
    return heights


def count_holes(board):
    """Count holes (empty cells below filled cells)"""
    holes = 0
    for col in range(board.shape[1]):
        column = board[:, col]
        found_block = False
        for cell in column:
            if cell > 0:
                found_block = True
            elif found_block:
                holes += 1
    return holes


def calculate_bumpiness(board):
    """Calculate bumpiness (height differences between adjacent columns)"""
    heights = get_column_heights(board)
    return sum(abs(heights[i] - heights[i+1]) for i in range(len(heights)-1))


def get_max_height(board):
    """Get maximum height"""
    heights = get_column_heights(board)
    return max(heights) if heights else 0


def get_horizontal_distribution(board):
    """Measure horizontal distribution (lower variance = better)"""
    heights = get_column_heights(board)
    if not heights or max(heights) == 0:
        return 1.0
    
    variance = np.var(heights)
    max_possible_variance = (board.shape[0] / 2) ** 2
    return 1.0 - min(variance / max_possible_variance, 1.0)


def balanced_reward_shaping(obs, action, reward, done, info):
    """
    Fixed balanced reward shaping - prevents rotation exploitation
    
    Key changes:
    1. NO per-step rewards (no reward just for rotating)
    2. Only reward actual progress (lines cleared)
    3. Penalize time-wasting (very small step penalty)
    4. Death penalty only if no progress made
    """
    from src.reward_shaping import (
        extract_board_from_obs, count_holes, get_column_heights,
        calculate_bumpiness, get_horizontal_distribution
    )
    import numpy as np
    
    # Start with ZERO reward (not the base reward!)
    shaped_reward = 0.0
    
    # Extract board
    board = extract_board_from_obs(obs)
    if board is None or len(board.shape) != 2:
        return shaped_reward
    
    # Get board metrics
    heights = get_column_heights(board)
    holes = count_holes(board)
    max_height = max(heights) if heights else 0
    bumpiness = calculate_bumpiness(heights)
    distribution = get_horizontal_distribution(board)
    
    # ========================================================================
    # 1. LINE CLEARING (Primary Objective) - HUGE rewards
    # ========================================================================
    lines = (info.get('lines_cleared', 0) or 
             info.get('number_of_lines', 0) or 
             info.get('lines', 0) or 0)
    
    if lines > 0:
        # Exponential rewards for more lines
        line_rewards = {
            1: 1000,   # Single line
            2: 3000,   # Double
            3: 6000,   # Triple
            4: 12000   # Tetris!
        }
        shaped_reward += line_rewards.get(lines, lines * 1000)
    


def aggressive_reward_shaping(obs, action, reward, done, info):
    """Aggressive - heavy penalties, IGNORES base reward"""
    shaped_reward = 0
    board = extract_board_from_obs(obs)
    
    ## shaped_reward += 2.0 if not done else -100 ## endless loop of rotation?
    
    lines = info.get('lines_cleared', 0) or info.get('number_of_lines', 0)
    if lines > 0:
        shaped_reward += [1000, 3000, 6000, 20000][lines-1]
    
    shaped_reward -= get_max_height(board) * 5
    shaped_reward -= count_holes(board) * 10
    shaped_reward -= calculate_bumpiness(board) * 2
    shaped_reward += get_horizontal_distribution(board) * 20
    
    return max(shaped_reward, -200)


def positive_reward_shaping(obs, action, reward, done, info):
    """Positive-focused - minimal penalties, IGNORES base reward"""
    shaped_reward = 0
    board = extract_board_from_obs(obs)
    
    # High survival bonus
    shaped_reward += 10.0 if not done else -10
    
    # Line clear bonuses
    lines = info.get('lines_cleared', 0) or info.get('number_of_lines', 0)
    if lines > 0:
        shaped_reward += [300, 900, 1800, 6000][lines-1]
    
    # Gentle penalties
    max_height = get_max_height(board)
    if max_height > 15:
        shaped_reward -= (max_height - 15) * 1
    
    holes = count_holes(board)
    if holes > 5:
        shaped_reward -= (holes - 5) * 1
    
    shaped_reward += get_horizontal_distribution(board) * 15
    
    # Bonus for low board
    if max_height < 5:
        shaped_reward += 15
    elif max_height < 10:
        shaped_reward += 5
    
    return max(shaped_reward, 0)