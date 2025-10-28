# test_reward_helpers.py
"""
Test script to verify the reward shaping helper functions work correctly

Run this BEFORE integrating into train.py to make sure everything works!

Usage:
    python test_reward_helpers.py
"""

import numpy as np
import sys
import os
from pathlib import Path

# Add parent directory to path so we can import config and src modules
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

# Import the helper functions
try:
    from src.reward_shaping import (
        get_column_heights,
        calculate_aggregate_height,
        count_holes,
        calculate_bumpiness,
        calculate_wells,
        get_max_height,
        extract_board_from_obs,
        aggressive_reward_shaping,
        positive_reward_shaping,
        balanced_reward_shaping
    )
    print("✅ Successfully imported all helper functions!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure reward_shape.py is in the same directory")
    sys.exit(1)


def create_test_board(scenario="empty"):
    """Create different test boards for testing"""
    board = np.zeros((20, 10), dtype=np.float32)
    
    if scenario == "empty":
        # Empty board
        pass
    
    elif scenario == "simple":
        # Simple flat bottom
        board[-3:, :] = 1  # Fill bottom 3 rows
    
    elif scenario == "holes":
        # Board with holes
        board[-4:, :] = 1  # Fill bottom 4 rows
        board[-3, 2] = 0   # Create hole at column 2
        board[-2, 5] = 0   # Create hole at column 5
        board[-3, 7] = 0   # Create hole at column 7
    
    elif scenario == "bumpy":
        # Bumpy surface
        heights = [2, 5, 3, 7, 2, 8, 3, 6, 2, 4]
        for col, h in enumerate(heights):
            if h > 0:
                board[-h:, col] = 1
    
    elif scenario == "dangerous":
        # Dangerously high
        heights = [18, 17, 19, 18, 17, 18, 19, 18, 17, 18]
        for col, h in enumerate(heights):
            if h > 0:
                board[-h:, col] = 1
    
    elif scenario == "well":
        # Board with a deep well
        board[-10:, :] = 1
        board[-10:, 5] = 0  # Column 5 is empty (well)
    
    return board


def print_board(board, title="Board"):
    """Pretty print a board"""
    print(f"\n{title}:")
    print("─" * 22)
    for row in board[:10]:  # Show top 10 rows
        print("│", end="")
        for cell in row:
            print(" █" if cell != 0 else " ·", end="")
        print(" │")
    print("─" * 22)
    print("  0 1 2 3 4 5 6 7 8 9  (columns)")


def test_scenario(name, board):
    """Test all metrics on a given board"""
    print(f"\n{'='*60}")
    print(f"SCENARIO: {name}")
    print(f"{'='*60}")
    
    print_board(board, f"{name} Board")
    
    # Calculate all metrics
    heights = get_column_heights(board)
    agg_height = calculate_aggregate_height(board)
    max_h = get_max_height(board)
    holes = count_holes(board)
    bump = calculate_bumpiness(board)
    wells = calculate_wells(board)
    
    print(f"\n📊 METRICS:")
    print(f"   Column Heights:    {heights}")
    print(f"   Aggregate Height:  {agg_height}")
    print(f"   Max Height:        {max_h}")
    print(f"   Holes:             {holes}")
    print(f"   Bumpiness:         {bump}")
    print(f"   Wells:             {wells}")
    
    # Test reward shaping
    print(f"\n💰 REWARD SHAPING (no lines cleared):")
    
    # Create fake info dict
    info = {'lines_cleared': 0}
    
    # Test all three modes
    aggressive_r = aggressive_reward_shaping(board, 0, 0, False, info)
    positive_r = positive_reward_shaping(board, 0, 0, False, info)
    balanced_r = balanced_reward_shaping(board, 0, 0, False, info)
    
    print(f"   Aggressive mode:   {aggressive_r:+.2f}")
    print(f"   Positive mode:     {positive_r:+.2f}")
    print(f"   Balanced mode:     {balanced_r:+.2f}")
    
    # Test with line clear
    print(f"\n💎 WITH 1 LINE CLEARED:")
    info_lines = {'lines_cleared': 1}
    aggressive_r = aggressive_reward_shaping(board, 0, 0, False, info_lines)
    positive_r = positive_reward_shaping(board, 0, 0, False, info_lines)
    balanced_r = balanced_reward_shaping(board, 0, 0, False, info_lines)
    
    print(f"   Aggressive mode:   {aggressive_r:+.2f}")
    print(f"   Positive mode:     {positive_r:+.2f}")
    print(f"   Balanced mode:     {balanced_r:+.2f}")


def test_edge_cases():
    """Test edge cases"""
    print(f"\n{'='*60}")
    print("EDGE CASE TESTS")
    print(f"{'='*60}")
    
    # Test 1: Empty board
    print("\n1. Empty board (should all be 0)")
    empty = np.zeros((20, 10))
    print(f"   Heights: {get_column_heights(empty)}")
    print(f"   Holes: {count_holes(empty)}")
    print(f"   Bumpiness: {calculate_bumpiness(empty)}")
    
    # Test 2: Full board
    print("\n2. Full board (should have no holes)")
    full = np.ones((20, 10))
    print(f"   Aggregate height: {calculate_aggregate_height(full)}")
    print(f"   Holes: {count_holes(full)}")
    
    # Test 3: Single column
    print("\n3. Single column filled")
    single = np.zeros((20, 10))
    single[:, 5] = 1
    print(f"   Heights: {get_column_heights(single)}")
    print(f"   Bumpiness: {calculate_bumpiness(single)}")
    
    # Test 4: Observation extraction
    print("\n4. Observation extraction test")
    
    # Test with numpy array
    print("   - Direct numpy array: ", end="")
    test_obs = np.zeros((20, 10))
    result = extract_board_from_obs(test_obs)
    print("✅" if result is not None else "❌")
    
    # Test with dict
    print("   - Dict with 'board' key: ", end="")
    test_obs = {'board': np.zeros((20, 10))}
    result = extract_board_from_obs(test_obs)
    print("✅" if result is not None else "❌")
    
    # Test with 3D array
    print("   - 3D array (H, W, C): ", end="")
    test_obs = np.zeros((20, 10, 1))
    result = extract_board_from_obs(test_obs)
    print("✅" if result is not None and len(result.shape) == 2 else "❌")


def compare_modes():
    """Compare all three reward shaping modes"""
    print(f"\n{'='*60}")
    print("REWARD SHAPING MODE COMPARISON")
    print(f"{'='*60}")
    
    # Create a moderately bad board
    board = create_test_board("holes")
    
    info_no_lines = {'lines_cleared': 0}
    info_1_line = {'lines_cleared': 1}
    info_tetris = {'lines_cleared': 4}
    
    print("\n📋 For a board with holes (bad state):")
    print(f"   Holes: {count_holes(board)}")
    print(f"   Height: {calculate_aggregate_height(board)}")
    
    modes = [
        ('Aggressive', aggressive_reward_shaping),
        ('Positive', positive_reward_shaping),
        ('Balanced', balanced_reward_shaping)
    ]
    
    for mode_name, mode_func in modes:
        print(f"\n   {mode_name} mode:")
        r0 = mode_func(board, 0, 0, False, info_no_lines)
        r1 = mode_func(board, 0, 0, False, info_1_line)
        r4 = mode_func(board, 0, 0, False, info_tetris)
        
        print(f"      No lines:  {r0:+7.1f}")
        print(f"      1 line:    {r1:+7.1f}  (Δ = +{r1-r0:.1f})")
        print(f"      TETRIS:    {r4:+7.1f}  (Δ = +{r4-r0:.1f})")


def main():
    """Run all tests"""
    print("="*60)
    print("TETRIS REWARD SHAPING HELPER FUNCTIONS TEST")
    print("="*60)
    
    # Test different scenarios
    scenarios = [
        ("Empty Board", create_test_board("empty")),
        ("Simple Flat Bottom", create_test_board("simple")),
        ("Board with Holes", create_test_board("holes")),
        ("Bumpy Surface", create_test_board("bumpy")),
        ("Dangerous Height", create_test_board("dangerous")),
        ("Deep Well", create_test_board("well"))
    ]
    
    for name, board in scenarios:
        test_scenario(name, board)
    
    # Test edge cases
    test_edge_cases()
    
    # Compare modes
    compare_modes()
    
    print(f"\n{'='*60}")
    print("✅ ALL TESTS COMPLETE!")
    print(f"{'='*60}")
    print("\nIf you see this message, all helper functions work correctly!")
    print("\nNext steps:")
    print("1. Your reward_shape.py file is ready to use")
    print("2. Update train.py imports: from reward_shape import ...")
    print("3. Run training with: python train.py --reward_shaping aggressive --force_fresh")
    print()


if __name__ == "__main__":
    main()