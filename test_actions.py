# test_actions.py
"""Test script to verify action mappings are working correctly"""

import numpy as np
from config import make_env, ACTION_LEFT, ACTION_RIGHT, ACTION_HARD_DROP, ACTION_MEANINGS
from src.reward_shaping import extract_board_from_obs, get_column_heights


def test_action_effects():
    """Test what each action actually does"""
    print("="*80)
    print("🔍 TESTING ACTION MAPPINGS")
    print("="*80)
    
    env = make_env(render_mode=None, use_complete_vision=True)
    
    print("\n📋 Discovered action mappings:")
    if ACTION_MEANINGS:
        for i, meaning in enumerate(ACTION_MEANINGS):
            print(f"   {i}: {meaning}")
    print(f"\n   ACTION_LEFT = {ACTION_LEFT}")
    print(f"   ACTION_RIGHT = {ACTION_RIGHT}")
    print(f"   ACTION_HARD_DROP = {ACTION_HARD_DROP}")
    
    print("\n" + "="*80)
    print("🧪 TESTING LEFT/RIGHT MOVEMENT")
    print("="*80)
    
    # Test 1: Does LEFT action move piece left?
    print("\n1️⃣ Testing LEFT action...")
    obs, _ = env.reset(seed=42)
    
    # Take several LEFT actions and drop
    for _ in range(4):
        obs, _, done, _, _ = env.step(ACTION_LEFT)
        if done:
            break
    
    # Drop the piece
    obs, _, done, _, _ = env.step(ACTION_HARD_DROP)
    
    board = extract_board_from_obs(obs)
    heights = get_column_heights(board)
    
    print(f"   After moving LEFT 4x and dropping:")
    print(f"   Column heights: {heights}")
    
    left_side_filled = any(heights[i] > 0 for i in range(3))
    if left_side_filled:
        print(f"   ✅ LEFT action works! Piece on left side.")
    else:
        print(f"   ❌ LEFT action may not be working!")
    
    # Test 2: Does RIGHT action move piece right?
    print("\n2️⃣ Testing RIGHT action...")
    obs, _ = env.reset(seed=43)
    
    # Take several RIGHT actions and drop
    for _ in range(4):
        obs, _, done, _, _ = env.step(ACTION_RIGHT)
        if done:
            break
    
    # Drop the piece
    obs, _, done, _, _ = env.step(ACTION_HARD_DROP)
    
    board = extract_board_from_obs(obs)
    heights = get_column_heights(board)
    
    print(f"   After moving RIGHT 4x and dropping:")
    print(f"   Column heights: {heights}")
    
    right_side_filled = any(heights[i] > 0 for i in range(7, 10))
    if right_side_filled:
        print(f"   ✅ RIGHT action works! Piece on right side.")
    else:
        print(f"   ❌ RIGHT action may not be working!")
    
    print("\n" + "="*80)
    print("🧪 TESTING BALANCED PLACEMENT")
    print("="*80)
    
    # Test 3: Can we place pieces across the board?
    print("\n3️⃣ Testing balanced placement strategy...")
    obs, _ = env.reset(seed=44)
    
    placement_actions = [
        ([], "Center"),
        ([ACTION_LEFT, ACTION_LEFT], "Left"),
        ([ACTION_RIGHT, ACTION_RIGHT], "Right"),
        ([ACTION_LEFT], "Center-Left"),
        ([ACTION_RIGHT], "Center-Right"),
    ]
    
    pieces_placed = 0
    done = False
    
    for moves, position in placement_actions * 4:  # Try 20 pieces
        if done:
            break
            
        # Execute movement
        for action in moves:
            obs, _, done, _, info = env.step(action)
            if done:
                break
        
        if not done:
            # Drop piece
            obs, _, done, _, info = env.step(ACTION_HARD_DROP)
            pieces_placed += 1
            
            # Check for lines
            lines = info.get('lines_cleared', 0) or info.get('number_of_lines', 0)
            if lines > 0:
                print(f"   🎉 LINE CLEARED after {pieces_placed} pieces!")
    
    board = extract_board_from_obs(obs)
    heights = get_column_heights(board)
    
    print(f"\n   Final column heights after {pieces_placed} pieces:")
    print(f"   {heights}")
    
    # Check distribution
    variance = np.var(heights)
    max_height = max(heights)
    min_height = min(h for h in heights if h > 0) if any(h > 0 for h in heights) else 0
    
    print(f"\n   Height variance: {variance:.2f}")
    print(f"   Max height: {max_height}, Min height: {min_height}")
    
    if variance < 20 and max_height > 0:
        print(f"   ✅ Balanced placement works! Good horizontal distribution.")
    elif variance > 50:
        print(f"   ⚠️  High variance - pieces not spreading evenly.")
        if heights[0] < 3 and heights[-1] < 3 and max(heights[3:7]) > 10:
            print(f"   ❌ CENTER STACKING DETECTED! Actions may be wrong.")
    
    env.close()
    
    print("\n" + "="*80)
    print("📊 CONCLUSION")
    print("="*80)
    
    if left_side_filled and right_side_filled:
        print("✅ Action mappings appear to be CORRECT!")
        print("   LEFT and RIGHT actions work as expected.")
        print("\n💡 If lines still aren't clearing during training:")
        print("   - Agent needs more exploration")
        print("   - Try training for more episodes")
        print("   - Check reward shaping incentives")
    else:
        print("❌ Action mappings may be INCORRECT!")
        print("\n💡 Debug suggestions:")
        print("   1. Check tetris-gymnasium version")
        print("   2. Manually test each action ID (0-7)")
        print("   3. Verify environment configuration")


if __name__ == "__main__":
    test_action_effects()