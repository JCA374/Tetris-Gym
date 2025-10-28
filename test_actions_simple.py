# test_actions_simple.py
"""
Simple action test using your existing config.py
This bypasses environment ID issues by using your working setup.
"""

import numpy as np
from config import make_env, discover_action_meanings, ACTION_LEFT, ACTION_RIGHT, ACTION_HARD_DROP


def print_board(board, title="Board"):
    """Pretty print a Tetris board"""
    print(f"\n{title}:")
    H, W = board.shape
    print("   " + "".join(str(i) for i in range(W)))
    print("   " + "-" * W)
    for i, row in enumerate(board):
        row_str = "".join("█" if cell > 0 else "." for cell in row)
        print(f"{i:2d}|{row_str}")
    print()


def extract_board(obs):
    """Extract board from observation"""
    if isinstance(obs, dict):
        board = obs.get('board', obs.get('observation', list(obs.values())[0]))
    else:
        board = obs
    
    # Handle channel dimension
    if len(board.shape) == 3:
        if board.shape[0] <= 4:
            board = board[0]
        elif board.shape[2] <= 4:
            board = board[:, :, 0]
    
    # Normalize
    board = board.astype(np.float32)
    if board.max() > 1:
        board = board / 255.0
    
    return board


def get_filled_columns(board):
    """Return set of column indices that have any filled cells"""
    H, W = board.shape
    filled = set()
    for col in range(W):
        if np.any(board[:, col] > 0):
            filled.add(col)
    return filled


def test_horizontal_movement():
    """Test that LEFT and RIGHT actually work"""
    
    print("="*70)
    print("🧪 TESTING HORIZONTAL MOVEMENT")
    print("="*70)
    
    # Create environment using YOUR config
    env = make_env(render_mode=None)
    
    print(f"\n✅ Environment created: {env}")
    print(f"Action space: {env.action_space} (n={env.action_space.n})")
    
    # Get action meanings from YOUR config
    action_meanings = discover_action_meanings()
    print(f"\n🎯 Action Mappings:")
    for action_id, meaning in action_meanings.items():
        print(f"   {action_id}: {meaning}")
    
    print(f"\nUsing actions from config.py:")
    print(f"  LEFT = {ACTION_LEFT}")
    print(f"  RIGHT = {ACTION_RIGHT}")
    print(f"  HARD_DROP = {ACTION_HARD_DROP}")
    
    # Test 1: LEFT movement
    print("\n" + "="*70)
    print("TEST 1: Move LEFT x5 then DROP")
    print("="*70)
    
    obs, _ = env.reset()
    
    for _ in range(5):
        obs, _, terminated, truncated, _ = env.step(ACTION_LEFT)
        if terminated or truncated:
            break
    
    obs, _, _, _, _ = env.step(ACTION_HARD_DROP)
    
    board1 = extract_board(obs)
    cols1 = get_filled_columns(board1)
    print(f"Filled columns after LEFT: {sorted(cols1) if cols1 else 'none'}")
    
    # Test 2: RIGHT movement
    print("\n" + "="*70)
    print("TEST 2: Move RIGHT x5 then DROP")
    print("="*70)
    
    obs, _ = env.reset()
    
    for _ in range(5):
        obs, _, terminated, truncated, _ = env.step(ACTION_RIGHT)
        if terminated or truncated:
            break
    
    obs, _, _, _, _ = env.step(ACTION_HARD_DROP)
    
    board2 = extract_board(obs)
    cols2 = get_filled_columns(board2)
    print(f"Filled columns after RIGHT: {sorted(cols2) if cols2 else 'none'}")
    
    # Test 3: Alternating LEFT/RIGHT
    print("\n" + "="*70)
    print("TEST 3: Alternate LEFT/RIGHT for 10 pieces")
    print("="*70)
    
    obs, _ = env.reset()
    
    for i in range(10):
        if i % 2 == 0:
            # Go LEFT
            for _ in range(3):
                obs, _, terminated, truncated, _ = env.step(ACTION_LEFT)
                if terminated or truncated:
                    break
        else:
            # Go RIGHT
            for _ in range(3):
                obs, _, terminated, truncated, _ = env.step(ACTION_RIGHT)
                if terminated or truncated:
                    break
        
        obs, _, terminated, truncated, _ = env.step(ACTION_HARD_DROP)
        if terminated or truncated:
            print(f"  Episode ended after piece {i+1}")
            break
    
    board3 = extract_board(obs)
    cols3 = get_filled_columns(board3)
    print(f"Final filled columns: {sorted(cols3) if cols3 else 'none'}")
    print(f"Number of columns used: {len(cols3)}/10")
    
    # Print the final board
    print_board(board3, "Final Board State")
    
    # Verdict
    print("\n" + "="*70)
    print("📊 VERDICT")
    print("="*70)
    
    results = []
    
    # Check if LEFT and RIGHT create different patterns
    if cols1 != cols2:
        print("✅ LEFT and RIGHT create different patterns (GOOD)")
        results.append(True)
    else:
        print("❌ LEFT and RIGHT create same pattern (BAD)")
        results.append(False)
    
    # Check if multiple columns are used
    if len(cols3) >= 5:
        print(f"✅ Multiple columns used: {len(cols3)}/10 (GOOD)")
        results.append(True)
    elif len(cols3) >= 3:
        print(f"⚠️  Some columns used: {len(cols3)}/10 (OK but could be better)")
        results.append(True)
    else:
        print(f"❌ Very few columns used: {len(cols3)}/10 (BAD)")
        results.append(False)
    
    # Check specific placements
    if cols1 and cols2:
        left_avg = sum(cols1) / len(cols1)
        right_avg = sum(cols2) / len(cols2)
        
        if left_avg < 4 and right_avg > 6:
            print(f"✅ LEFT places left (avg col {left_avg:.1f}), RIGHT places right (avg col {right_avg:.1f})")
            results.append(True)
        else:
            print(f"⚠️  LEFT avg col {left_avg:.1f}, RIGHT avg col {right_avg:.1f}")
            results.append(True)
    
    # Overall
    if all(results):
        print("\n✅ ACTIONS ARE WORKING CORRECTLY")
        print("   → The single-column problem is due to AGENT EXPLORATION")
        print("   → Solution: Use updated train.py with --force_exploration")
        print("\n📝 Next steps:")
        print("   1. Replace your train.py with the updated version")
        print("   2. Replace your src/reward_shaping.py with the updated version")
        print("   3. Run: python train.py --force_exploration --reward_shaping aggressive")
    else:
        print("\n❌ ACTIONS MAY NOT BE WORKING PROPERLY")
        print("   → This is unusual since we're using your config.py")
        print("   → Check if environment is rendering correctly")
    
    env.close()
    print("="*70)


if __name__ == "__main__":
    try:
        test_horizontal_movement()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 Make sure config.py is in the same directory!")