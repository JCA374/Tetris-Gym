# verify_actions.py
"""
Quick verification that LEFT and RIGHT actions work properly.
This will help diagnose if the single-column problem is environmental.
"""

import gymnasium as gym
import numpy as np
import tetris_gymnasium


def verify_horizontal_movement():
    """
    Test that LEFT and RIGHT actions actually move pieces horizontally.
    """
    
    print("="*70)
    print("🧪 VERIFYING HORIZONTAL MOVEMENT")
    print("="*70)
    
    env = gym.make("tetris_gymnasium/Tetris", render_mode=None)
    
    # Determine action IDs
    action_meanings = {}
    if hasattr(env.unwrapped, 'get_action_meanings'):
        meanings = env.unwrapped.get_action_meanings()
        action_meanings = {i: m for i, m in enumerate(meanings)}
    else:
        action_meanings = {
            0: "NOOP", 1: "LEFT", 2: "RIGHT", 3: "DOWN",
            4: "ROTATE_CW", 5: "ROTATE_CCW", 6: "HARD_DROP", 7: "SWAP"
        }
    
    # Find LEFT and RIGHT action IDs
    left_id = next((k for k, v in action_meanings.items() if "LEFT" in v), 1)
    right_id = next((k for k, v in action_meanings.items() if "RIGHT" in v), 2)
    hard_drop_id = next((k for k, v in action_meanings.items() if "HARD_DROP" in v), 6)
    
    print(f"\nAction mappings:")
    print(f"  LEFT = {left_id}")
    print(f"  RIGHT = {right_id}")
    print(f"  HARD_DROP = {hard_drop_id}")
    
    def extract_board(obs):
        if isinstance(obs, dict):
            board = obs.get('board', obs.get('observation', list(obs.values())[0]))
        else:
            board = obs
        
        if len(board.shape) == 3:
            board = board[0] if board.shape[0] <= 4 else board[:, :, 0]
        
        return (board / 255.0 if board.max() > 1 else board)
    
    def get_filled_columns(board):
        """Return set of column indices that have any filled cells"""
        H, W = board.shape
        filled = set()
        for col in range(W):
            if np.any(board[:, col] > 0):
                filled.add(col)
        return filled
    
    # Test 1: LEFT movement
    print("\n" + "="*70)
    print("TEST 1: Move LEFT and drop")
    print("="*70)
    
    obs, _ = env.reset()
    board_start = extract_board(obs)
    cols_start = get_filled_columns(board_start)
    print(f"Starting filled columns: {sorted(cols_start) if cols_start else 'none'}")
    
    # Move left 5 times then drop
    for _ in range(5):
        obs, _, _, _, _ = env.step(left_id)
    obs, _, _, _, _ = env.step(hard_drop_id)
    
    board_after_left = extract_board(obs)
    cols_after_left = get_filled_columns(board_after_left)
    print(f"After LEFT x5 + DROP: {sorted(cols_after_left) if cols_after_left else 'none'}")
    
    # Test 2: RIGHT movement
    print("\n" + "="*70)
    print("TEST 2: Move RIGHT and drop")
    print("="*70)
    
    obs, _ = env.reset()
    board_start = extract_board(obs)
    cols_start = get_filled_columns(board_start)
    print(f"Starting filled columns: {sorted(cols_start) if cols_start else 'none'}")
    
    # Move right 5 times then drop
    for _ in range(5):
        obs, _, _, _, _ = env.step(right_id)
    obs, _, _, _, _ = env.step(hard_drop_id)
    
    board_after_right = extract_board(obs)
    cols_after_right = get_filled_columns(board_after_right)
    print(f"After RIGHT x5 + DROP: {sorted(cols_after_right) if cols_after_right else 'none'}")
    
    # Test 3: Multiple pieces
    print("\n" + "="*70)
    print("TEST 3: Alternate LEFT/RIGHT for 10 pieces")
    print("="*70)
    
    obs, _ = env.reset()
    
    for i in range(10):
        # Alternate between left and right
        if i % 2 == 0:
            # Go left
            for _ in range(3):
                obs, _, terminated, truncated, _ = env.step(left_id)
                if terminated or truncated:
                    break
        else:
            # Go right
            for _ in range(3):
                obs, _, terminated, truncated, _ = env.step(right_id)
                if terminated or truncated:
                    break
        
        # Drop
        obs, _, terminated, truncated, _ = env.step(hard_drop_id)
        if terminated or truncated:
            print(f"  Episode ended after piece {i+1}")
            break
    
    board_final = extract_board(obs)
    cols_final = get_filled_columns(board_final)
    print(f"Final filled columns: {sorted(cols_final) if cols_final else 'none'}")
    print(f"Number of columns used: {len(cols_final)}/10")
    
    # Verdict
    print("\n" + "="*70)
    print("📊 VERDICT")
    print("="*70)
    
    results = []
    
    # Check if LEFT creates different columns than RIGHT
    if cols_after_left != cols_after_right:
        print("✅ LEFT and RIGHT create different patterns (GOOD)")
        results.append(True)
    else:
        print("❌ LEFT and RIGHT create same pattern (BAD - actions not working)")
        results.append(False)
    
    # Check if multiple columns are used
    if len(cols_final) >= 5:
        print(f"✅ Multiple columns used: {len(cols_final)}/10 (GOOD)")
        results.append(True)
    elif len(cols_final) >= 3:
        print(f"⚠️  Some columns used: {len(cols_final)}/10 (OK but could be better)")
        results.append(True)
    else:
        print(f"❌ Very few columns used: {len(cols_final)}/10 (BAD)")
        results.append(False)
    
    # Overall
    if all(results):
        print("\n✅ ACTIONS ARE WORKING CORRECTLY")
        print("   → The single-column problem is due to agent exploration")
        print("   → Use the updated train.py with --force_exploration")
    else:
        print("\n❌ ACTIONS MAY NOT BE WORKING")
        print("   → Check your config.py action mapping")
        print("   → Try updating tetris-gymnasium: pip install -U tetris-gymnasium")
        print("   → Check environment variant (v0, v1, etc.)")
    
    env.close()
    print("="*70)


if __name__ == "__main__":
    verify_horizontal_movement()
    
    print("\n💡 NEXT STEPS:")
    print("""
If actions are working:
  → Use: python train.py --force_exploration --reward_shaping aggressive
  → The fixes in train.py will solve the single-column problem

If actions are NOT working:
  → Update: pip install -U tetris-gymnasium
  → Check your config.py for correct action IDs
  → Try different environment variant in config.py
    """)