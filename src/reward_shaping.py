# reward_shaping.py
"""
IMPROVED Tetris Reward Shaping - Addresses Zero Lines Cleared Problem

Key Changes from Original:
1. MUCH stronger line clearing rewards (exponential scaling)
2. Larger penalties for bad play (holes, height, single-column)
3. Strong incentive for board flatness and distribution
4. Step-by-step progress rewards (not just survival bonus)
5. NO exploitable patterns (no free rewards for doing nothing)
"""

import numpy as np

# reward_shaping.py — add this
import numpy as np

# ---- tiny helpers (you already have these in this file; reuse yours if present) ----
def extract_board_from_obs(obs):
    if isinstance(obs, dict):
        b = obs.get("board") or obs.get("observation")
        if b is None:
            for v in obs.values():
                if isinstance(v, np.ndarray):
                    b = v; break
            if b is None:
                return None
    else:
        b = obs
    b = np.asarray(b)
    if b.ndim == 3 and b.shape[-1] >= 1:
        b = b[..., 0]
    return (b > 0).astype(np.uint8)

def get_column_heights(board):
    H, W = board.shape
    heights = []
    for c in range(W):
        col = board[:, c]
        h = 0
        for r in range(H):
            if col[r]:
                h = H - r
                break
        heights.append(h)
    return heights

def count_holes(board):
    H, W = board.shape
    holes = 0
    for c in range(W):
        seen = False
        for r in range(H):
            if board[r, c]:
                seen = True
            elif seen and not board[r, c]:
                holes += 1
    return holes

def calculate_bumpiness(heights):
    return sum(abs(heights[i] - heights[i+1]) for i in range(len(heights)-1))

# ---- Potential Φ(s) and PBRS shaping ----
def _potential(board):
    """Lower is better. Scale small to keep values stable."""
    heights = get_column_heights(board)
    max_h   = max(heights) if heights else 0
    holes   = count_holes(board)
    bump    = calculate_bumpiness(heights)
    # weights chosen to rank states, not dwarf line rewards
    return -( 2.0*max_h + 6.0*holes + 0.8*bump )

def balanced_pbrs_reward(obs, action, base_reward, done, info, gamma=0.99):
    """
    Potential-Based Reward Shaping:
      r' = base_reward + gamma*Φ(s') - Φ(s)
    Keep line clears huge (from base env), keep step penalties near-zero.
    """
    # extract s (current) board
    board_s = extract_board_from_obs(obs)
    if board_s is None:
        return float(base_reward)

    phi_s = _potential(board_s)

    # We can’t see s' here; compute Φ(s') from info['next_obs'] if you pass it,
    # otherwise approximate using current obs after env.step in train loop.
    # Minimal invasive approach: allow caller to pass info['next_board'].
    next_board = info.get("next_board", None)
    if next_board is not None:
        board_sp = (np.asarray(next_board) > 0).astype(np.uint8)
        phi_sp = _potential(board_sp)
        shaped = base_reward + (gamma * phi_sp - phi_s)
    else:
        # Fallback: just add -phi_s to encourage moving toward better states.
        shaped = base_reward - 0.1 * phi_s  # small weight if s' is unknown

    # Optional tiny survival carrot; avoid big step penalties
    if not done:
        shaped += 0.1

    # Small death penalty only to break ties
    if done and info.get('lines_cleared', 0) == 0:
        shaped -= 5.0

    return float(shaped)



def horizontal_distribution(board):
    """
    Reward spreading pieces across columns (not stacking in one place).
    
    Returns:
        Score from 0 (all in one column) to 1 (evenly distributed)
    """
    if board is None or len(board.shape) != 2:
        return 0
    
    height, width = board.shape
    column_counts = np.sum(board, axis=0)
    
    if np.sum(column_counts) == 0:
        return 1.0
    
    # Calculate entropy-like distribution
    total = np.sum(column_counts)
    probs = column_counts / total
    probs = probs[probs > 0]  # Remove zeros
    
    if len(probs) == 0:
        return 0
    
    # Normalized entropy
    max_entropy = np.log(width)
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    
    return entropy / max_entropy


def calculate_column_variance(heights):
    """
    Calculate variance in column heights.
    
    Low variance = flat board = good.
    """
    if len(heights) == 0:
        return 0
    
    return float(np.var(heights))


def count_wells(heights):
    """
    Count "wells" (columns significantly lower than neighbors).
    
    Wells make it hard to place pieces efficiently.
    """
    if len(heights) < 3:
        return 0
    
    wells = 0
    
    for i in range(1, len(heights) - 1):
        if heights[i] < heights[i-1] - 2 and heights[i] < heights[i+1] - 2:
            wells += 1
    
    return wells


def single_column_penalty(heights):
    """
    MASSIVE penalty for stacking everything in one column.
    This is the main issue from your diagnostic output.
    """
    non_zero = [h for h in heights if h > 0]
    
    if len(non_zero) == 0:
        return 0.0
    
    if len(non_zero) == 1:
        # MASSIVE penalty for single column (scales with height)
        return -100.0 * non_zero[0]
    
    if len(non_zero) <= 2:
        # Large penalty for using only 2 columns
        return -40.0 * max(non_zero)
    
    if len(non_zero) <= 3:
        # Medium penalty for using only 3 columns
        return -20.0 * max(non_zero)
    
    return 0.0


def count_filled_rows(board):
    """
    Count rows that are completely or mostly filled.
    This gives intermediate reward for getting close to line clears.
    """
    if board is None or len(board.shape) != 2:
        return 0
    
    height, width = board.shape
    filled_count = 0
    
    for row in range(height):
        row_sum = np.sum(board[row, :])
        # Count as "filled" if >= 80% full
        if row_sum >= width * 0.8:
            filled_count += 1
    
    return filled_count


# ============================================================================
# REWARD SHAPING FUNCTIONS
# ============================================================================

def improved_balanced_reward_shaping(obs, action, reward, done, info):
    """
    IMPROVED balanced reward shaping designed to fix the zero-lines problem.
    
    Key improvements:
    1. Exponential line clearing rewards (1000-16000)
    2. Intermediate rewards for almost-full rows
    3. Much stronger penalties for holes and height
    4. Massive penalty for single-column stacking
    5. Strong distribution bonuses
    6. Small step penalty to encourage efficiency
    
    Args:
        obs: Current observation
        action: Action taken (can be ignored for state-based shaping)
        reward: Original environment reward
        done: Whether episode is finished
        info: Info dict with game stats
    
    Returns:
        float: Shaped reward
    """
    # Extract board
    try:
        board = extract_board_from_obs(obs)
        if board is None:
            return -1.0  # Small penalty if can't extract board
    except Exception:
        return -1.0
    
    shaped_reward = 0.0  # Start from 0, build up
    
    # ========================================
    # 1. LINE CLEARING REWARDS (DOMINANT)
    # ========================================
    lines_keys = ['lines_cleared', 'cleared_lines', 'lines', 'n_lines', 'number_of_lines']
    lines_cleared = 0
    for key in lines_keys:
        if key in info:
            val = info.get(key, 0)
            lines_cleared = val if val is not None else 0
            break
    
    if lines_cleared > 0:
        # EXPONENTIAL rewards: 1 line = 1000, 2 lines = 3000, 3 = 6000, 4 = 16000
        line_rewards = [1000, 3000, 6000, 16000]
        shaped_reward += line_rewards[min(lines_cleared - 1, 3)]
    
    # ========================================
    # 2. INTERMEDIATE PROGRESS REWARDS
    # ========================================
    # Reward for having rows that are ALMOST complete
    filled_rows = count_filled_rows(board)
    if filled_rows > 0:
        shaped_reward += 10.0 * filled_rows  # Reward progress toward line clears
    
    # ========================================
    # 3. CALCULATE BOARD METRICS
    # ========================================
    heights = get_column_heights(board)
    max_height = max(heights) if heights else 0
    avg_height = np.mean(heights) if heights else 0
    holes = count_holes(board)
    bumpiness = calculate_bumpiness(heights)
    distribution = horizontal_distribution(board)
    variance = calculate_column_variance(heights)
    wells = count_wells(heights)
    single_col = single_column_penalty(heights)
    
    # ========================================
    # 4. PENALTIES (MUCH STRONGER)
    # ========================================
    shaped_reward -= 5.0 * holes           # Holes are VERY bad (was 2.0)
    shaped_reward -= 1.0 * max_height      # Discourage high stacks (was 0.5)
    shaped_reward -= 0.5 * bumpiness       # Flat surface is good (was 0.3)
    shaped_reward -= 2.0 * variance        # Even heights (was 1.5)
    shaped_reward -= 2.0 * wells           # Wells trap pieces (was 0.5)
    shaped_reward += single_col            # MASSIVE penalty (up to -100 * height)
    
    # ========================================
    # 5. BONUSES FOR GOOD PLAY
    # ========================================
    shaped_reward += 10.0 * distribution   # Reward spreading pieces (was 5.0)
    
    # Bonus for keeping board low
    if max_height < 10:
        shaped_reward += 5.0
    elif max_height < 15:
        shaped_reward += 2.0
    
    # ========================================
    # 6. STEP PENALTY (FORCE PROGRESS)
    # ========================================
    # Small penalty each step to encourage efficient play
    # This prevents the agent from just moving pieces around forever
    shaped_reward -= 1.0
    
    # ========================================
    # 7. DEATH PENALTY
    # ========================================
    if done and lines_cleared == 0:
        shaped_reward -= 100.0  # Big penalty for dying without progress
    
    return float(shaped_reward)  # Ensure float return


def aggressive_reward_shaping(obs, action, reward, done, info):
    """
    AGGRESSIVE reward shaping with even stronger penalties.
    Use this if improved_balanced still doesn't work.
    
    This is more punishing and may lead to more conservative play.
    """
    try:
        board = extract_board_from_obs(obs)
        if board is None:
            return -1.0
    except Exception:
        return -1.0
    
    shaped_reward = 0.0
    
    # Line clearing (same as balanced)
    lines_keys = ['lines_cleared', 'cleared_lines', 'lines', 'n_lines', 'number_of_lines']
    lines_cleared = 0
    for key in lines_keys:
        if key in info:
            val = info.get(key, 0)
            lines_cleared = val if val is not None else 0
            break
    
    if lines_cleared > 0:
        line_rewards = [1000, 3000, 6000, 16000]
        shaped_reward += line_rewards[min(lines_cleared - 1, 3)]
    
    # Intermediate progress
    filled_rows = count_filled_rows(board)
    shaped_reward += 15.0 * filled_rows  # Higher bonus
    
    # Metrics
    heights = get_column_heights(board)
    max_height = max(heights) if heights else 0
    holes = count_holes(board)
    bumpiness = calculate_bumpiness(heights)
    distribution = horizontal_distribution(board)
    variance = calculate_column_variance(heights)
    wells = count_wells(heights)
    single_col = single_column_penalty(heights)
    
    # VERY STRONG penalties
    shaped_reward -= 10.0 * holes          # 2x stronger
    shaped_reward -= 2.0 * max_height      # 2x stronger
    shaped_reward -= 1.0 * bumpiness       # 2x stronger
    shaped_reward -= 3.0 * variance        # 1.5x stronger
    shaped_reward -= 3.0 * wells           # 1.5x stronger
    shaped_reward += single_col * 1.5      # 1.5x stronger (already massive)
    
    # Strong distribution bonus
    shaped_reward += 15.0 * distribution   # 1.5x stronger
    
    # Board height bonuses
    if max_height < 10:
        shaped_reward += 10.0
    elif max_height < 15:
        shaped_reward += 5.0
    
    # Step penalty
    shaped_reward -= 1.5  # Slightly higher
    
    # Death penalty
    if done and lines_cleared == 0:
        shaped_reward -= 150.0  # Harsher
    
    return float(shaped_reward)


def positive_reward_shaping(obs, action, reward, done, info):
    """
    Positive-focused reward shaping (fewer penalties, more bonuses).
    
    Use this if the agent becomes too conservative with aggressive shaping.
    """
    try:
        board = extract_board_from_obs(obs)
        if board is None:
            return 0.0
    except Exception:
        return 0.0
    
    shaped_reward = 0.0
    
    # Line clearing
    lines_keys = ['lines_cleared', 'cleared_lines', 'lines', 'n_lines', 'number_of_lines']
    lines_cleared = 0
    for key in lines_keys:
        if key in info:
            val = info.get(key, 0)
            lines_cleared = val if val is not None else 0
            break
    
    if lines_cleared > 0:
        line_rewards = [1000, 3000, 6000, 16000]
        shaped_reward += line_rewards[min(lines_cleared - 1, 3)]
    
    # Progress rewards
    filled_rows = count_filled_rows(board)
    shaped_reward += 20.0 * filled_rows  # Strong bonus for progress
    
    # Metrics
    heights = get_column_heights(board)
    max_height = max(heights) if heights else 0
    holes = count_holes(board)
    distribution = horizontal_distribution(board)
    single_col = single_column_penalty(heights)
    
    # Mild penalties (only for serious problems)
    if holes > 3:
        shaped_reward -= 3.0 * (holes - 3)  # Only penalize if >3 holes
    
    if max_height > 15:
        shaped_reward -= 1.0 * (max_height - 15)  # Only if very high
    
    shaped_reward += single_col * 0.5  # Reduced penalty
    
    # Strong positive bonuses
    shaped_reward += 15.0 * distribution  # Reward good distribution
    
    # Board state bonuses
    if max_height < 10:
        shaped_reward += 10.0
    elif max_height < 15:
        shaped_reward += 5.0
    
    # Survival bonus (small)
    if not done:
        shaped_reward += 0.5
    
    return float(shaped_reward)


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
    'count_filled_rows',
    'improved_balanced_reward_shaping',
    'aggressive_reward_shaping',
    'positive_reward_shaping'
]