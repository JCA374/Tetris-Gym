# src/reward_shaping.py
"""Reward shaping functions for Tetris - FIXED with reasonable coefficients"""

import numpy as np


def extract_board_from_obs(obs):
    """Extract and normalize the board from observation"""
    if len(obs.shape) == 3:
        # 3D observation (height, width, channels)
        board = obs[:, :, 0] if obs.shape[2] == 1 else obs[:, :, 0]
    else:
        # 2D observation
        board = obs
    
    # Ensure binary values (0 or 1)
    board = (board > 0).astype(np.float32)
    
    return board


def get_column_heights(board):
    """Get the height of each column"""
    heights = []
    for col in range(board.shape[1]):
        column = board[:, col]
        # Find highest occupied cell
        occupied = np.where(column > 0)[0]
        if len(occupied) > 0:
            height = board.shape[0] - occupied[0]
        else:
            height = 0
        heights.append(height)
    return heights


def count_holes(board):
    """Count the number of holes (empty cells below filled cells)"""
    holes = 0
    for col in range(board.shape[1]):
        column = board[:, col]
        found_block = False
        for row in range(board.shape[0]):
            if column[row] > 0:
                found_block = True
            elif found_block and column[row] == 0:
                holes += 1
    return holes


def calculate_bumpiness(board):
    """Calculate the bumpiness (sum of height differences between adjacent columns)"""
    heights = get_column_heights(board)
    if len(heights) < 2:
        return 0
    
    bumpiness = 0
    for i in range(len(heights) - 1):
        bumpiness += abs(heights[i] - heights[i + 1])
    
    return bumpiness


def get_max_height(board):
    """Get the maximum height on the board"""
    heights = get_column_heights(board)
    return max(heights) if heights else 0


def get_horizontal_distribution(board):
    """Measure how evenly pieces are distributed horizontally"""
    heights = get_column_heights(board)
    if not heights or max(heights) == 0:
        return 0
    
    # Calculate variance in column heights
    mean_height = np.mean(heights)
    variance = np.var(heights)
    
    # Lower variance = better distribution
    # Normalize to 0-1 where 1 is perfect distribution
    max_possible_variance = (board.shape[0] / 2) ** 2
    distribution_score = 1.0 - min(variance / max_possible_variance, 1.0)
    
    return distribution_score


def balanced_reward_shaping(obs, action, reward, done, info):
    """Balanced reward shaping - encourages clearing lines and staying alive"""
    shaped_reward = reward
    
    # Extract board
    board = extract_board_from_obs(obs)
    
    # Basic survival bonus
    if not done:
        shaped_reward += 2.0  # Survival bonus per step
    else:
        shaped_reward -= 10  # Death penalty (reduced from 50)
    
    # Line clear bonuses - MOST IMPORTANT
    lines_cleared = info.get('lines_cleared', 0) or info.get('number_of_lines', 0)
    if lines_cleared > 0:
        if lines_cleared == 1:
            shaped_reward += 500  # Single line
        elif lines_cleared == 2:
            shaped_reward += 1500  # Double
        elif lines_cleared == 3:
            shaped_reward += 3000  # Triple
        elif lines_cleared == 4:
            shaped_reward += 10000  # Tetris!
    
    # Height penalty (gentle)
    max_height = get_max_height(board)
    if max_height > 10:
        shaped_reward -= (max_height - 10) * 5  # Reduced from 10
    
    # Hole penalty (gentle)
    holes = count_holes(board)
    shaped_reward -= holes * 2  # Reduced from 5
    
    # Bumpiness penalty (very gentle)
    bumpiness = calculate_bumpiness(board)
    shaped_reward -= bumpiness * 0.5  # Reduced from 2
    
    # Horizontal distribution bonus (encourage spreading pieces)
    distribution = get_horizontal_distribution(board)
    shaped_reward += distribution * 10
    
    # Clamp to prevent extreme negative rewards
    shaped_reward = max(shaped_reward, -100)
    
    return shaped_reward


def aggressive_reward_shaping(obs, action, reward, done, info):
    """Aggressive reward shaping - heavily penalizes poor placement"""
    shaped_reward = reward
    
    # Extract board
    board = extract_board_from_obs(obs)
    
    # Survival/death
    if not done:
        shaped_reward += 1.0  # Small survival bonus
    else:
        shaped_reward -= 50  # Significant death penalty
    
    # Line clear bonuses (HUGE rewards for clearing)
    lines_cleared = info.get('lines_cleared', 0) or info.get('number_of_lines', 0)
    if lines_cleared > 0:
        if lines_cleared == 1:
            shaped_reward += 1000  # Single line
        elif lines_cleared == 2:
            shaped_reward += 3000  # Double
        elif lines_cleared == 3:
            shaped_reward += 6000  # Triple
        elif lines_cleared == 4:
            shaped_reward += 20000  # Tetris!
    
    # Strong height penalty
    max_height = get_max_height(board)
    shaped_reward -= max_height * 10
    
    # Average height penalty
    heights = get_column_heights(board)
    avg_height = np.mean(heights) if heights else 0
    shaped_reward -= avg_height * 5
    
    # Strong hole penalty
    holes = count_holes(board)
    shaped_reward -= holes * 10
    
    # Bumpiness penalty
    bumpiness = calculate_bumpiness(board)
    shaped_reward -= bumpiness * 2
    
    # Horizontal distribution bonus
    distribution = get_horizontal_distribution(board)
    shaped_reward += distribution * 20
    
    # Clamp rewards
    shaped_reward = max(shaped_reward, -200)
    
    return shaped_reward


def positive_reward_shaping(obs, action, reward, done, info):
    """Positive-focused reward shaping - minimal penalties, focus on rewards"""
    shaped_reward = reward
    
    # Extract board
    board = extract_board_from_obs(obs)
    
    # Strong survival bonus
    if not done:
        shaped_reward += 5.0  # Good survival bonus
    else:
        shaped_reward -= 5  # Minimal death penalty
    
    # Line clear bonuses (main source of reward)
    lines_cleared = info.get('lines_cleared', 0) or info.get('number_of_lines', 0)
    if lines_cleared > 0:
        if lines_cleared == 1:
            shaped_reward += 200  # Single line
        elif lines_cleared == 2:
            shaped_reward += 600  # Double
        elif lines_cleared == 3:
            shaped_reward += 1200  # Triple
        elif lines_cleared == 4:
            shaped_reward += 5000  # Tetris!
    
    # Very gentle penalties
    max_height = get_max_height(board)
    if max_height > 15:  # Only penalize very high stacks
        shaped_reward -= (max_height - 15) * 2
    
    # Minimal hole penalty
    holes = count_holes(board)
    if holes > 5:  # Only penalize many holes
        shaped_reward -= (holes - 5) * 1
    
    # Horizontal distribution bonus (encourage spreading)
    distribution = get_horizontal_distribution(board)
    shaped_reward += distribution * 15
    
    # Bonus for keeping the board low
    if max_height < 5:
        shaped_reward += 10
    elif max_height < 10:
        shaped_reward += 5
    
    # No negative clamping - keep it positive!
    shaped_reward = max(shaped_reward, 0)
    
    return shaped_reward