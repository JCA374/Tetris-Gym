# reward_shaping_UPDATED.py
"""
Comprehensive Tetris Reward Shaping with Board Analysis Helper Functions

Based on research from successful Tetris AI implementations:
- 4 Core Heuristics: Aggregate Height, Holes, Bumpiness, Complete Lines
- Standard approach from Code My Road, Lucky's Notes, and academic papers
"""

import numpy as np


# =============================================================================
# BOARD ANALYSIS HELPER FUNCTIONS
# =============================================================================

def extract_board_from_obs(obs):
    """
    Extract 2D board array from observation
    
    Handles different observation formats:
    - Dict with 'board' key
    - Tensor that needs conversion
    - Already a numpy array
    """
    # If observation is a dict
    if isinstance(obs, dict):
        if 'board' in obs:
            board = obs['board']
        else:
            # Might be flattened, try to get the board channel
            board = obs
    else:
        board = obs
    
    # Convert tensor to numpy if needed
    if hasattr(board, 'cpu'):
        board = board.cpu().numpy()
    
    board = np.array(board)
    
    # Handle different array shapes
    if len(board.shape) > 2:
        # If shape is (H, W, C), take first channel
        if board.shape[-1] == 1:
            board = board.squeeze(-1)
        elif len(board.shape) == 3:
            # Frame stacking - use most recent
            board = board[:, :, -1]
    
    # Ensure 2D
    if len(board.shape) != 2:
        return None
        
    return board


def get_column_heights(board):
    """
    Calculate height of each column
    
    Height = distance from highest filled cell to bottom
    Empty column = height 0
    """
    if board is None:
        return []
    
    heights = []
    rows, cols = board.shape
    
    for col in range(cols):
        height = 0
        for row in range(rows):
            if board[row, col] != 0:
                # Found a filled cell - height is distance to bottom
                height = rows - row
                break
        heights.append(height)
    
    return heights


def calculate_aggregate_height(board):
    """
    Aggregate height = sum of all column heights
    
    MINIMIZE this: Lower board = more space = can survive longer
    """
    heights = get_column_heights(board)
    return sum(heights)


def count_holes(board):
    """
    Count holes: empty cells with at least one filled cell above them
    
    MINIMIZE this: Holes are very bad - hard to clear lines above them
    
    Example:
    . X .    <- filled cell
    . . .    <- this empty cell is a HOLE (has filled cell above)
    . X .
    """
    if board is None:
        return 0
    
    holes = 0
    rows, cols = board.shape
    
    for col in range(cols):
        found_filled = False
        for row in range(rows):
            cell = board[row, col]
            
            if cell != 0:
                found_filled = True
            elif found_filled and cell == 0:
                # Empty cell with filled cell above = hole
                holes += 1
    
    return holes


def calculate_bumpiness(board):
    """
    Bumpiness = sum of absolute height differences between adjacent columns
    
    MINIMIZE this: Smooth/flat top is better, easier to place pieces
    
    Example:
    Heights: [3, 5, 5, 4, 6, 5]
    Bumpiness = |3-5| + |5-5| + |5-4| + |4-6| + |6-5| = 2+0+1+2+1 = 6
    """
    heights = get_column_heights(board)
    
    if len(heights) < 2:
        return 0
    
    bumpiness = 0
    for i in range(len(heights) - 1):
        bumpiness += abs(heights[i] - heights[i + 1])
    
    return bumpiness


def calculate_wells(board):
    """
    Wells = deep single-column gaps (surrounded by taller columns)
    
    MINIMIZE this: Wells make clearing lines difficult
    
    Example:
    X X . X X   <- column 2 is a well (depth 2)
    X X . X X
    X X X X X
    """
    heights = get_column_heights(board)
    
    if len(heights) < 2:
        return 0
    
    wells = 0
    for i in range(len(heights)):
        left_height = heights[i-1] if i > 0 else heights[i]
        right_height = heights[i+1] if i < len(heights)-1 else heights[i]
        
        # Well depth = how much lower this column is than neighbors
        well_depth = min(left_height, right_height) - heights[i]
        wells += max(0, well_depth)
    
    return wells


def get_max_height(board):
    """Get maximum column height"""
    heights = get_column_heights(board)
    return max(heights) if heights else 0


def count_complete_lines(board, prev_board=None):
    """
    Count complete lines that were just cleared
    
    If prev_board provided, counts difference
    Otherwise just checks current board for full rows
    """
    if board is None:
        return 0
    
    complete_lines = 0
    rows, cols = board.shape
    
    for row in range(rows):
        if np.all(board[row, :] != 0):
            complete_lines += 1
    
    return complete_lines


# =============================================================================
# REWARD SHAPING FUNCTIONS
# =============================================================================

def aggressive_reward_shaping(obs, action, base_reward, done, info):
    """
    AGGRESSIVE reward shaping with board quality heuristics
    
    Based on research: Score = -w0×height + w1×lines - w2×holes - w3×bumpiness
    
    This version focuses HEAVILY on:
    1. Punishing bad board states (holes, height, bumpiness)
    2. Rewarding line clears enormously
    3. Small survival bonus to keep playing
    """
    shaped_reward = 0  # Start at zero for aggressive approach
    
    # Extract board
    board = extract_board_from_obs(obs)
    
    if board is not None:
        # Calculate board metrics
        aggregate_height = calculate_aggregate_height(board)
        holes = count_holes(board)
        bumpiness = calculate_bumpiness(board)
        max_height = get_max_height(board)
        wells = calculate_wells(board)
        
        # PENALTIES (negative rewards)
        shaped_reward -= aggregate_height * 0.51  # Penalize total height
        shaped_reward -= holes * 3.5               # HEAVILY penalize holes
        shaped_reward -= bumpiness * 0.35          # Penalize bumpy surface
        shaped_reward -= wells * 0.65              # Penalize wells
        
        # Extra penalty if dangerously high
        if max_height > 15:  # More than 75% full
            shaped_reward -= (max_height - 15) * 2.0
    
    # HUGE LINE CLEAR BONUSES
    lines = info.get('lines_cleared', 0)
    if lines > 0:
        line_rewards = {
            1: 1000,   # Single
            2: 3000,   # Double  
            3: 6000,   # Triple
            4: 20000   # TETRIS!!!
        }
        bonus = line_rewards.get(lines, lines * 1000)
        shaped_reward += bonus
        print(f"  🔥 {lines} LINES! +{bonus} MEGA BONUS! Shaped reward: {shaped_reward:.1f}")
    
    # Small survival bonus (so agent doesn't die immediately)
    if not done:
        shaped_reward += 1.0
    
    # Death penalty
    if done:
        shaped_reward -= 50.0
    
    return shaped_reward


def positive_reward_shaping(obs, action, base_reward, done, info):
    """
    POSITIVE reward shaping - more balanced approach
    
    Encourages both survival and line clearing with board quality guidance
    """
    shaped_reward = base_reward
    
    # Extract board
    board = extract_board_from_obs(obs)
    
    if board is not None:
        # Calculate board metrics
        aggregate_height = calculate_aggregate_height(board)
        holes = count_holes(board)
        bumpiness = calculate_bumpiness(board)
        max_height = get_max_height(board)
        
        # Gentler penalties
        shaped_reward -= aggregate_height * 0.2
        shaped_reward -= holes * 2.0
        shaped_reward -= bumpiness * 0.15
        
        # Danger zone penalty
        if max_height > 16:
            shaped_reward -= (max_height - 16) * 1.0
    
    # Survival bonus
    if not done:
        shaped_reward += 2.0
    
    # LINE CLEAR BONUSES
    lines = info.get('lines_cleared', 0)
    if lines > 0:
        line_rewards = {
            1: 500,
            2: 1500,
            3: 3000,
            4: 10000
        }
        bonus = line_rewards.get(lines, lines * 500)
        shaped_reward += bonus
        
        if lines == 1:
            print(f"  🎯 LINE! +{bonus}")
        elif lines == 2:
            print(f"  🎯🎯 DOUBLE! +{bonus}")
        elif lines == 3:
            print(f"  🎯🎯🎯 TRIPLE! +{bonus}")
        elif lines == 4:
            print(f"  🎯🎯🎯🎯 TETRIS!!! +{bonus}")
    
    # Small death penalty
    if done:
        shaped_reward -= 10.0
    
    return shaped_reward


def balanced_reward_shaping(obs, action, base_reward, done, info):
    """
    BALANCED reward shaping - middle ground
    
    Uses standard Tetris AI formula with proven weights
    """
    shaped_reward = base_reward
    
    # Extract board
    board = extract_board_from_obs(obs)
    
    if board is not None:
        # Calculate all metrics
        aggregate_height = calculate_aggregate_height(board)
        holes = count_holes(board)
        bumpiness = calculate_bumpiness(board)
        wells = calculate_wells(board)
        
        # Research-based weights (from successful Tetris AIs)
        shaped_reward -= aggregate_height * 0.35
        shaped_reward -= holes * 2.5
        shaped_reward -= bumpiness * 0.25
        shaped_reward -= wells * 0.4
    
    # Moderate survival bonus
    if not done:
        shaped_reward += 1.5
    
    # Line clear rewards (exponential-ish)
    lines = info.get('lines_cleared', 0)
    if lines > 0:
        line_rewards = {
            1: 750,
            2: 2000,
            3: 4500,
            4: 15000
        }
        bonus = line_rewards.get(lines, lines * 750)
        shaped_reward += bonus
        print(f"  ✨ {lines} lines cleared! +{bonus}")
    
    # Moderate death penalty
    if done:
        shaped_reward -= 30.0
    
    return shaped_reward


# =============================================================================
# DIAGNOSTIC / TESTING FUNCTIONS
# =============================================================================

def test_helper_functions():
    """Test all helper functions with example boards"""
    print("Testing Tetris Board Analysis Functions")
    print("="*60)
    
    # Create test board
    test_board = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Row 0 (top)
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Complete line
        [1, 1, 0, 1, 1, 1, 1, 1, 1, 1],  # Has a hole at col 2
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 0, 1, 1, 1, 1],  # Has a hole at col 5
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Row 9 (bottom)
    ], dtype=np.float32)
    
    print("\nTest Board (1=filled, 0=empty):")
    print(test_board)
    
    heights = get_column_heights(test_board)
    print(f"\nColumn Heights: {heights}")
    print(f"Aggregate Height: {calculate_aggregate_height(test_board)}")
    print(f"Max Height: {get_max_height(test_board)}")
    print(f"Holes: {count_holes(test_board)}")
    print(f"Bumpiness: {calculate_bumpiness(test_board)}")
    print(f"Wells: {calculate_wells(test_board)}")
    
    # Test with bumpy board
    bumpy_board = np.zeros((10, 10))
    for col in range(10):
        height = [2, 5, 3, 7, 2, 8, 3, 6, 2, 4][col]
        bumpy_board[-height:, col] = 1
    
    print("\n" + "="*60)
    print("Bumpy Board Test:")
    print(f"Heights: {get_column_heights(bumpy_board)}")
    print(f"Bumpiness: {calculate_bumpiness(bumpy_board)}")
    print(f"Wells: {calculate_wells(bumpy_board)}")


if __name__ == "__main__":
    # Run tests when file is executed directly
    test_helper_functions()
