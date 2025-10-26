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
    Extract and normalize the board from observation
    """
    # Handle different observation formats
    if isinstance(obs, dict):
        board = obs.get('board', np.zeros((20, 10)))
        if 'board' in obs:
            # Extract only the playable area (20x10) from the full board
            full_board = obs['board']
            if full_board.shape == (24, 18):
                # Tetris Gymnasium uses 24x18 with padding
                board = full_board[2:22, 4:14]  # Extract 20x10 playable area
            elif full_board.shape == (20, 10):
                board = full_board
            else:
                # Fallback: try to extract 20x10 from whatever shape
                h, w = full_board.shape[:2]
                h_start = max(0, (h - 20) // 2)
                w_start = max(0, (w - 10) // 2)
                board = full_board[h_start:h_start+20, w_start:w_start+10]
    elif hasattr(obs, 'shape'):
        if len(obs.shape) == 3 and obs.shape[2] == 1:
            board = obs[:, :, 0]
        elif len(obs.shape) == 2:
            board = obs
        else:
            board = obs
    else:
        board = np.array(obs)
    
    # Ensure board is 20x10
    if board.shape != (20, 10):
        # Resize if necessary
        target_board = np.zeros((20, 10))
        h, w = min(20, board.shape[0]), min(10, board.shape[1]) if len(board.shape) > 1 else 10
        target_board[:h, :w] = board[:h, :w] if len(board.shape) > 1 else board[:h].reshape(-1, 1)[:, 0]
        board = target_board
    
    # ✨ CRITICAL FIX: Force binary normalization
    # Convert any non-zero values to 1.0 for consistent rewards
    if board.size > 0:
        board = (board > 0).astype(float)  # 0.0 or 1.0 only!
    
    return board

def calculate_horizontal_distribution(board):
    """
    Calculate how horizontally distributed the pieces are.
    Returns a value between 0 (all in center) and 1 (well distributed).
    """
    if board.size == 0 or not board.any():
        return 0.5  # Neutral for empty board
    
    # Calculate center of mass for each row
    rows, cols = board.shape
    col_indices = np.arange(cols)
    
    # Calculate horizontal variance for occupied rows
    variances = []
    for row in board:
        if row.any():
            # Calculate center of mass for this row
            center = np.average(col_indices, weights=row)
            # Calculate variance from center
            variance = np.average((col_indices - center)**2, weights=row)
            # Normalize by max possible variance (corners)
            max_variance = (cols - 1)**2 / 4
            normalized_variance = variance / max_variance if max_variance > 0 else 0
            variances.append(normalized_variance)
    
    if variances:
        return np.mean(variances)
    return 0.5

def calculate_height_penalty(board):
    """Calculate penalty based on max column height"""
    col_heights = np.array([
        20 - np.argmax(board[:, col] > 0) if board[:, col].any() else 0
        for col in range(board.shape[1])
    ])
    max_height = np.max(col_heights)
    avg_height = np.mean(col_heights)
    
    # Reduced exponential penalty for height
    height_penalty = (max_height / 20) ** 2 * 10  # Reduced from 30
    avg_penalty = (avg_height / 20) * 5  # Reduced from 20
    
    return height_penalty + avg_penalty

def calculate_hole_penalty(board):
    """Calculate penalty for holes (empty cells below filled cells)"""
    holes = 0
    for col in range(board.shape[1]):
        column = board[:, col]
        # Find the first filled cell from top
        filled_indices = np.where(column > 0)[0]
        if len(filled_indices) > 0:
            first_filled = filled_indices[0]
            # Count empty cells below the first filled cell
            holes += np.sum(column[first_filled:] == 0)
    
    return holes * 2  # Reduced from 5 points penalty per hole

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
    """Calculate the bumpiness (height differences between adjacent columns)"""
    col_heights = np.array([
        20 - np.argmax(board[:, col] > 0) if board[:, col].any() else 0
        for col in range(board.shape[1])
    ])
    
    if len(col_heights) > 1:
        bumpiness = np.sum(np.abs(np.diff(col_heights)))
        return bumpiness * 0.5  # Reduced from 2 points penalty per height difference
    return 0

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

def balanced_reward_shaping(obs, reward, done, info):
    """
    Balanced reward shaping with horizontal distribution bonus
    """
    board = extract_board_from_obs(obs)
    
    # Base reward (from environment)
    shaped_reward = reward * 100  # Amplify line clear rewards
    
    # Calculate penalties (REDUCED)
    height_penalty = calculate_height_penalty(board) * 0.3  # Reduced from 1.0
    hole_penalty = calculate_hole_penalty(board) * 0.3  # Reduced from 1.0
    bumpiness_penalty = calculate_bumpiness(board) * 0.3  # Reduced from 1.0
    
    # Calculate bonuses
    horizontal_bonus = calculate_horizontal_distribution(board) * 5  # Reduced from 10
    
    # Survival bonus (increases with time)
    steps = info.get('steps', 0)
    survival_bonus = min(steps * 0.05, 5)  # Reduced from 0.1 and cap at 5
    
    # Line clear bonus
    lines_cleared = info.get('lines_cleared', 0)
    if lines_cleared > 0:
        shaped_reward += lines_cleared * 50  # Big bonus for clearing lines
    
    # Combine all factors
    shaped_reward += horizontal_bonus + survival_bonus
    shaped_reward -= height_penalty + hole_penalty + bumpiness_penalty
    
    # Death penalty
    if done:
        shaped_reward -= 50  # Reduced from 100
    
    # Clamp to reasonable range
    shaped_reward = np.clip(shaped_reward, -100, 500)  # Reduced lower bound
    
    return shaped_reward

def aggressive_reward_shaping(obs, reward, done, info):
    """
    More aggressive reward shaping that heavily penalizes height
    """
    board = extract_board_from_obs(obs)
    
    shaped_reward = reward * 150  # Higher amplification
    
    # Stronger penalties (but still reduced from original)
    height_penalty = calculate_height_penalty(board) * 0.6  # Reduced from 2.0
    hole_penalty = calculate_hole_penalty(board) * 0.5  # Reduced from 1.5
    bumpiness_penalty = calculate_bumpiness(board) * 0.5  # Reduced from 1.5
    
    # Bonuses
    horizontal_bonus = calculate_horizontal_distribution(board) * 8  # Reduced from 15
    survival_bonus = min(info.get('steps', 0) * 0.1, 10)  # Reduced from 0.2, cap at 10
    
    # Line clear mega bonus
    lines_cleared = info.get('lines_cleared', 0)
    if lines_cleared > 0:
        shaped_reward += lines_cleared * 100
    
    shaped_reward += horizontal_bonus + survival_bonus
    shaped_reward -= height_penalty + hole_penalty + bumpiness_penalty
    
    if done:
        shaped_reward -= 75  # Reduced from 150
    
    shaped_reward = np.clip(shaped_reward, -150, 1000)  # Reduced lower bound
    
    return shaped_reward

def positive_reward_shaping(obs, reward, done, info):
    """
    Emphasizes positive rewards over penalties
    """
    board = extract_board_from_obs(obs)
    
    shaped_reward = reward * 200  # Very high amplification for positive actions
    
    # Very mild penalties  
    height_penalty = calculate_height_penalty(board) * 0.15  # Reduced from 0.5
    hole_penalty = calculate_hole_penalty(board) * 0.15  # Reduced from 0.5
    bumpiness_penalty = calculate_bumpiness(board) * 0.15  # Reduced from 0.5
    
    # Strong bonuses
    horizontal_bonus = calculate_horizontal_distribution(board) * 10  # Reduced from 20
    survival_bonus = min(info.get('steps', 0) * 0.2, 20)  # Reduced from 0.5, cap at 20
    
    # Huge line clear bonus
    lines_cleared = info.get('lines_cleared', 0)
    if lines_cleared > 0:
        shaped_reward += lines_cleared * 200
    
    # Low placement bonus (keeping pieces low)
    avg_height = np.mean([20 - np.argmax(board[:, col] > 0) if board[:, col].any() else 0
                          for col in range(board.shape[1])])
    if avg_height < 5:
        shaped_reward += 5  # Reduced from 10
    
    shaped_reward += horizontal_bonus + survival_bonus
    shaped_reward -= height_penalty + hole_penalty + bumpiness_penalty
    
    if done:
        shaped_reward -= 25  # Reduced from 50 - smaller death penalty
    
    shaped_reward = np.clip(shaped_reward, -50, 2000)  # Reduced lower bound
    
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
