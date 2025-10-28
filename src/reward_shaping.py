# src/reward_shaping.py
"""
IMPROVED Reward Shaping for Tetris with Stronger Anti-Stacking Penalties
"""

import numpy as np


def extract_board_from_obs(obs):
    """
    Extract and normalize board from observation.
    
    Args:
        obs: Observation from environment (dict or array)
    
    Returns:
        board: 2D numpy array (H, W) with values in [0, 1]
    """
    if isinstance(obs, dict):
        if 'board' in obs:
            board = obs['board']
        elif 'observation' in obs:
            board = obs['observation']
        else:
            # Fallback: use first array-like value
            for v in obs.values():
                if hasattr(v, 'shape') and len(v.shape) >= 2:
                    board = v
                    break
    else:
        board = obs
    
    # Handle channel dimension
    if len(board.shape) == 3:
        if board.shape[0] <= 4:  # (C, H, W)
            board = board[0]
        elif board.shape[2] <= 4:  # (H, W, C)
            board = board[:, :, 0]
    
    # Ensure 2D
    if len(board.shape) != 2:
        raise ValueError(f"Cannot extract 2D board from shape {board.shape}")
    
    # Normalize to [0, 1]
    board = board.astype(np.float32)
    if board.max() > 1:
        board = board / 255.0
    
    return board


def get_column_heights(board):
    """
    Calculate height of each column (number of filled cells from bottom).
    
    Args:
        board: 2D array (H, W)
    
    Returns:
        heights: List of column heights
    """
    H, W = board.shape
    heights = []
    
    for col in range(W):
        height = 0
        for row in range(H):
            if board[row, col] > 0:
                height = H - row
                break
        heights.append(height)
    
    return heights


def count_holes(board):
    """
    Count holes (empty cells with filled cells above them).
    
    Args:
        board: 2D array (H, W)
    
    Returns:
        holes: Number of holes
    """
    H, W = board.shape
    holes = 0
    
    for col in range(W):
        found_block = False
        for row in range(H):
            if board[row, col] > 0:
                found_block = True
            elif found_block:
                holes += 1
    
    return holes


def calculate_bumpiness(heights):
    """
    Calculate bumpiness (sum of absolute height differences between adjacent columns).
    
    Args:
        heights: List of column heights
    
    Returns:
        bumpiness: Total bumpiness score
    """
    if len(heights) < 2:
        return 0
    
    bumpiness = 0
    for i in range(len(heights) - 1):
        bumpiness += abs(heights[i] - heights[i + 1])
    
    return bumpiness


def horizontal_distribution(board):
    """
    Calculate how evenly pieces are distributed horizontally.
    Rewards spreading pieces across all columns.
    
    Args:
        board: 2D array (H, W)
    
    Returns:
        distribution_score: Higher is better (range 0-1)
    """
    H, W = board.shape
    
    # Count non-empty cells per column
    col_counts = [np.sum(board[:, col] > 0) for col in range(W)]
    
    if sum(col_counts) == 0:
        return 1.0  # Empty board is perfectly distributed
    
    # Calculate entropy (higher entropy = more spread out)
    total = sum(col_counts)
    probs = [c / total for c in col_counts if c > 0]
    
    if len(probs) <= 1:
        return 0.0  # All in one column is worst
    
    entropy = -sum(p * np.log(p + 1e-10) for p in probs)
    max_entropy = np.log(W)
    
    return entropy / max_entropy if max_entropy > 0 else 0.0


def calculate_column_variance(heights):
    """
    NEW: Calculate variance in column heights.
    High variance means uneven stacking (bad).
    
    Args:
        heights: List of column heights
    
    Returns:
        variance: Standard deviation of heights
    """
    if len(heights) == 0:
        return 0
    return np.std(heights)


def count_wells(heights):
    """
    NEW: Count deep wells (columns significantly lower than neighbors).
    Wells make it hard to place pieces.
    
    Args:
        heights: List of column heights
    
    Returns:
        well_penalty: Penalty for having wells
    """
    if len(heights) < 3:
        return 0
    
    well_penalty = 0
    for i in range(1, len(heights) - 1):
        left = heights[i - 1]
        center = heights[i]
        right = heights[i + 1]
        
        # Check if center is significantly lower than both neighbors
        if center < left - 2 and center < right - 2:
            depth = min(left - center, right - center)
            well_penalty += depth ** 2  # Quadratic penalty for deep wells
    
    return well_penalty


def single_column_penalty(heights):
    """
    NEW: Strong penalty for stacking in a single column.
    This is the main issue from your diagnostic output.
    
    Args:
        heights: List of column heights
    
    Returns:
        penalty: Large penalty if using only one column
    """
    non_zero = [h for h in heights if h > 0]
    
    if len(non_zero) == 0:
        return 0
    
    if len(non_zero) == 1:
        # MASSIVE penalty for single column
        return -50.0 * non_zero[0]  # Gets worse as height increases
    
    if len(non_zero) <= 2:
        # Large penalty for using only 2 columns
        return -20.0 * max(non_zero)
    
    if len(non_zero) <= 3:
        # Medium penalty for using only 3 columns
        return -10.0 * max(non_zero)
    
    return 0


def balanced_reward_shaping(obs, action, reward, done, info):
    """
    IMPROVED balanced reward shaping with anti-single-column penalties.
    
    Args:
        obs: Current observation
        action: Action taken
        reward: Original reward from environment
        done: Whether episode ended
        info: Info dict from environment
    
    Returns:
        shaped_reward: Modified reward
    """
    # Extract board
    try:
        board = extract_board_from_obs(obs)
    except:
        return reward
    
    # Base reward
    shaped_reward = reward
    
    # Calculate metrics
    heights = get_column_heights(board)
    max_height = max(heights) if heights else 0
    avg_height = np.mean(heights) if heights else 0
    holes = count_holes(board)
    bumpiness = calculate_bumpiness(heights)
    distribution = horizontal_distribution(board)
    variance = calculate_column_variance(heights)
    wells = count_wells(heights)
    single_col = single_column_penalty(heights)
    
    # Line clear bonus (multiplicative to make it dominant)
    lines_keys = ['lines_cleared', 'cleared_lines', 'lines', 'n_lines', 'number_of_lines']
    lines_cleared = 0
    for key in lines_keys:
        if key in info:
            lines_cleared = info.get(key, 0) or 0
            break
    
    if lines_cleared > 0:
        shaped_reward += 100.0 * lines_cleared ** 2  # Quadratic bonus
    
    # Penalties (INCREASED)
    shaped_reward -= 2.0 * holes  # DOUBLED from 1.0
    shaped_reward -= 0.5 * max_height  # INCREASED from 0.2
    shaped_reward -= 0.3 * bumpiness  # INCREASED from 0.1
    shaped_reward -= 1.5 * variance  # NEW: Penalty for uneven heights
    shaped_reward -= 0.5 * wells  # NEW: Penalty for wells
    shaped_reward += single_col  # NEW: MASSIVE penalty for single column
    
    # Distribution bonus (INCREASED to encourage spreading)
    shaped_reward += 5.0 * distribution  # INCREASED from 2.0
    
    # Death penalty
    if done and reward <= 0:
        shaped_reward -= 50.0
    
    return shaped_reward


def aggressive_reward_shaping(obs, action, reward, done, info):
    """
    AGGRESSIVE reward shaping with even stronger penalties.
    Use this if balanced still produces bad behavior.
    """
    try:
        board = extract_board_from_obs(obs)
    except:
        return reward
    
    shaped_reward = reward
    
    heights = get_column_heights(board)
    max_height = max(heights) if heights else 0
    holes = count_holes(board)
    bumpiness = calculate_bumpiness(heights)
    distribution = horizontal_distribution(board)
    variance = calculate_column_variance(heights)
    wells = count_wells(heights)
    single_col = single_column_penalty(heights)
    
    # Line clears
    lines_keys = ['lines_cleared', 'cleared_lines', 'lines', 'n_lines', 'number_of_lines']
    lines_cleared = 0
    for key in lines_keys:
        if key in info:
            lines_cleared = info.get(key, 0) or 0
            break
    
    if lines_cleared > 0:
        shaped_reward += 200.0 * lines_cleared ** 2  # Even higher bonus
    
    # AGGRESSIVE penalties
    shaped_reward -= 5.0 * holes  # 5x original
    shaped_reward -= 1.0 * max_height  # 5x original
    shaped_reward -= 0.5 * bumpiness  # 5x original
    shaped_reward -= 3.0 * variance  # Strong variance penalty
    shaped_reward -= 1.0 * wells  # Strong well penalty
    shaped_reward += single_col * 2  # DOUBLE the single column penalty
    
    # Strong distribution bonus
    shaped_reward += 10.0 * distribution  # 5x original
    
    # Death penalty
    if done and reward <= 0:
        shaped_reward -= 100.0
    
    return shaped_reward


def positive_reward_shaping(obs, action, reward, done, info):
    """
    Positive-only reward shaping (no penalties, only bonuses).
    """
    try:
        board = extract_board_from_obs(obs)
    except:
        return reward
    
    shaped_reward = max(0, reward)  # No negative rewards
    
    heights = get_column_heights(board)
    distribution = horizontal_distribution(board)
    non_zero_cols = sum(1 for h in heights if h > 0)
    
    # Line clears
    lines_keys = ['lines_cleared', 'cleared_lines', 'lines', 'n_lines', 'number_of_lines']
    lines_cleared = 0
    for key in lines_keys:
        if key in info:
            lines_cleared = info.get(key, 0) or 0
            break
    
    if lines_cleared > 0:
        shaped_reward += 100.0 * lines_cleared ** 2
    
    # Positive bonuses only
    shaped_reward += 5.0 * distribution  # Reward spreading
    shaped_reward += 2.0 * non_zero_cols  # Reward using more columns
    shaped_reward += 0.5  # Survival bonus each step
    
    return shaped_reward


# Export all functions
__all__ = [
    'extract_board_from_obs',
    'get_column_heights',
    'count_holes',
    'calculate_bumpiness',
    'horizontal_distribution',
    'calculate_column_variance',
    'count_wells',
    'single_column_penalty',
    'balanced_reward_shaping',
    'aggressive_reward_shaping',
    'positive_reward_shaping'
]