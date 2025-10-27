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


def calculate_bumpiness(heights):
    """
    Calculate bumpiness (sum of height differences)
    
    Args:
        heights: List of column heights (NOT the board!)
    """
    if not heights or len(heights) < 2:
        return 0
    
    bumpiness = sum(abs(heights[i] - heights[i+1]) for i in range(len(heights) - 1))
    return bumpiness


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
    
    CRITICAL FIXES:
    1. Removed per-step survival bonus that caused rotation exploit
    2. Added -1 penalty for steps without progress
    3. Always returns a float (never None)
    
    Args:
        obs: Current observation
        action: Action taken
        reward: Original environment reward (not used in shaping)
        done: Episode termination flag
        info: Environment info dict
    
    Returns:
        shaped_reward: Modified reward value (always a float)
    """
    import numpy as np
    
    # Start from zero (don't use base reward)
    shaped_reward = 0.0
    
    # ========================================================================
    # EXTRACT BOARD STATE
    # ========================================================================
    board = extract_board_from_obs(obs)
    
    # Safety check - if board extraction fails, return penalty
    if board is None:
        return -1.0
    
    if len(board.shape) != 2:
        return -1.0
    
    # ========================================================================
    # CALCULATE BOARD METRICS
    # ========================================================================
    heights = get_column_heights(board)
    
    # Safety check for heights
    if not heights:
        return -1.0
    
    holes = count_holes(board)
    max_height = max(heights)
    bumpiness = calculate_bumpiness(heights)  # Pass heights, not board!
    distribution = get_horizontal_distribution(board)
    
    # ========================================================================
    # 1. LINE CLEARING REWARDS (Primary Objective)
    # ========================================================================
    lines = (info.get('lines_cleared', 0) or 
             info.get('number_of_lines', 0) or 
             info.get('lines', 0) or 0)
    
    if lines > 0:
        # Big rewards for clearing lines - exponential scaling
        line_rewards = {
            1: 1000,   # Single line
            2: 3000,   # Double
            3: 6000,   # Triple
            4: 12000   # Tetris!
        }
        shaped_reward += line_rewards.get(lines, lines * 1000)
    else:
        # CRITICAL: Penalty for not clearing lines
        # This prevents rotation exploit!
        shaped_reward -= 1.0
    
    # ========================================================================
    # 2. BOARD STATE PENALTIES
    # ========================================================================
    
    # Holes are very bad (hard to recover from)
    shaped_reward -= holes * 5.0
    
    # Height penalty (keep board low for survival)
    shaped_reward -= max_height * 1.0
    
    # Bumpiness penalty (smooth surface is easier to manage)
    shaped_reward -= bumpiness * 0.5
    
    # ========================================================================
    # 3. STRATEGIC BONUSES
    # ========================================================================
    
    # Horizontal distribution bonus (spread pieces across board)
    shaped_reward += distribution * 5.0
    
    # Low height bonus (encourages keeping board low)
    if max_height < 10:
        shaped_reward += 5.0
    
    # ========================================================================
    # 4. DEATH PENALTY
    # ========================================================================
    
    if done:
        if lines == 0:
            # Big penalty for dying without clearing any lines
            shaped_reward -= 50.0
        else:
            # Smaller penalty if at least cleared some lines
            shaped_reward -= 10.0
    
    # ========================================================================
    # 5. CLAMP AND RETURN
    # ========================================================================
    
    # Prevent extreme values
    shaped_reward = np.clip(shaped_reward, -500.0, 15000.0)
    
    # Always return a float
    return float(shaped_reward)


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