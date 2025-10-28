# test_environment_rendering.py
"""
Test script to diagnose Tetris environment rendering and action issues
"""

import sys
import os
from pathlib import Path

# Add parent directory to path so we can import config and src modules
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

import gymnasium as gym
import numpy as np
import tetris_gymnasium


def print_board(board, title="Board"):
    """Pretty print a Tetris board"""
    print(f"\n{title}:")
    print("   " + "".join(str(i) for i in range(board.shape[1])))
    print("   " + "-" * board.shape[1])
    for i, row in enumerate(board):
        row_str = "".join("█" if cell > 0 else "." for cell in row)
        print(f"{i:2d}|{row_str}")
    print()


def extract_board(obs):
    """Extract board from observation"""
    if isinstance(obs, dict):
        if 'board' in obs:
            board = obs['board']
        elif 'observation' in obs:
            board = obs['observation']
        else:
            for v in obs.values():
                if hasattr(v, 'shape') and len(v.shape) >= 2:
                    board = v
                    break
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


def get_column_heights(board):
    """Calculate column heights"""
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


def test_actions():
    """Test all actions and see what happens"""
    
    print("="*80)
    print("🔍 TESTING TETRIS ENVIRONMENT")
    print("="*80)
    
    # Create environment - try multiple possible IDs
    possible_ids = [
        "tetris_gymnasium/Tetris",
        "TetrisGymnasium/Tetris", 
        "Tetris-v0",
        "tetris_gymnasium:Tetris-v0"
    ]
    
    env = None
    for env_id in possible_ids:
        try:
            env = gym.make(env_id, render_mode=None)
            print(f"✅ Environment created with ID: {env_id}")
            break
        except:
            continue
    
    if env is None:
        # Try importing and registering manually
        try:
            from gymnasium.envs.registration import register
            import tetris_gymnasium
            
            register(
                id="TetrisTest-v0",
                entry_point="tetris_gymnasium.envs.tetris:Tetris",
            )
            env = gym.make("TetrisTest-v0", render_mode=None)
            print(f"✅ Environment created with manual registration")
        except Exception as e:
            print(f"❌ Could not create environment: {e}")
            print(f"\nPlease check your config.py to see how make_env() creates the environment")
            return
    
    print(f"\n✅ Environment: {env.spec.id}")
    print(f"Action space: {env.action_space} (n={env.action_space.n})")
    print(f"Observation space: {env.observation_space}")
    
    # Get action meanings
    action_meanings = {}
    if hasattr(env.unwrapped, 'get_action_meanings'):
        action_meanings = {i: m for i, m in enumerate(env.unwrapped.get_action_meanings())}
    else:
        # Default Tetris actions
        action_meanings = {
            0: "NOOP",
            1: "LEFT",
            2: "RIGHT",
            3: "DOWN",
            4: "ROTATE_CW",
            5: "ROTATE_CCW",
            6: "HARD_DROP",
            7: "SWAP"
        }
    
    print(f"\n🎯 Available actions ({len(action_meanings)}):")
    for action_id, meaning in action_meanings.items():
        print(f"   {action_id}: {meaning}")
    
    # Test random actions
    print(f"\n🎲 Testing random actions for 100 steps...")
    
    obs, info = env.reset()
    board = extract_board(obs)
    
    print_board(board, "Initial Board")
    print(f"Initial heights: {get_column_heights(board)}")
    
    action_counts = {i: 0 for i in range(env.action_space.n)}
    boards_seen = []
    
    for step in range(100):
        # Take random action
        action = env.action_space.sample()
        action_counts[action] += 1
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Extract board
        board = extract_board(obs)
        boards_seen.append(board.copy())
        
        # Print every 20 steps
        if (step + 1) % 20 == 0:
            heights = get_column_heights(board)
            print(f"\n--- Step {step + 1} ---")
            print(f"Last action: {action_meanings.get(action, f'UNKNOWN({action})')}")
            print(f"Column heights: {heights}")
            print(f"Non-zero columns: {sum(1 for h in heights if h > 0)}")
            print(f"Max height: {max(heights)}")
            print(f"Reward: {reward:.2f}")
        
        if done:
            print(f"\n⚠️  Episode ended at step {step + 1}")
            break
    
    # Final board
    board = extract_board(obs)
    print_board(board, "Final Board")
    
    heights = get_column_heights(board)
    print(f"Final heights: {heights}")
    print(f"Non-zero columns: {sum(1 for h in heights if h > 0)}/10")
    
    # Action distribution
    print(f"\n📊 Action Distribution:")
    total_actions = sum(action_counts.values())
    for action_id in sorted(action_counts.keys()):
        count = action_counts[action_id]
        pct = 100 * count / total_actions if total_actions > 0 else 0
        meaning = action_meanings.get(action_id, f"UNKNOWN({action_id})")
        print(f"   {meaning:12s}: {count:3d} ({pct:5.1f}%)")
    
    # Analysis
    print(f"\n🔍 ANALYSIS:")
    
    non_zero_cols = sum(1 for h in heights if h > 0)
    if non_zero_cols == 1:
        print(f"   ❌ CRITICAL: Only 1 column has pieces!")
        print(f"   → This suggests LEFT/RIGHT actions may not be working")
        print(f"   → Or the environment is not processing horizontal movement")
    elif non_zero_cols <= 3:
        print(f"   ⚠️  WARNING: Only {non_zero_cols} columns have pieces")
        print(f"   → Pieces are too clustered")
    else:
        print(f"   ✅ Good: {non_zero_cols} columns have pieces")
    
    # Check if board changed
    boards_unique = len(set(tuple(b.flatten()) for b in boards_seen))
    print(f"\n   Unique board states seen: {boards_unique}/{len(boards_seen)}")
    
    if boards_unique <= 5:
        print(f"   ❌ WARNING: Very few unique boards - actions may not be working!")
    
    env.close()
    print("\n" + "="*80)


def test_specific_action_sequence():
    """Test a specific sequence of actions to verify they work"""
    
    print("\n" + "="*80)
    print("🧪 TESTING SPECIFIC ACTION SEQUENCE")
    print("="*80)
    
    env = gym.make("tetris_gymnasium/Tetris", render_mode=None)
    
    # Get action meanings
    action_meanings = {}
    if hasattr(env.unwrapped, 'get_action_meanings'):
        action_meanings = {i: m for i, m in enumerate(env.unwrapped.get_action_meanings())}
    else:
        action_meanings = {
            0: "NOOP", 1: "LEFT", 2: "RIGHT", 3: "DOWN",
            4: "ROTATE_CW", 5: "ROTATE_CCW", 6: "HARD_DROP", 7: "SWAP"
        }
    
    obs, info = env.reset()
    board_initial = extract_board(obs)
    heights_initial = get_column_heights(board_initial)
    
    print(f"\nInitial heights: {heights_initial}")
    
    # Test sequence: LEFT, LEFT, LEFT, HARD_DROP
    sequence = [
        (1, "LEFT"),
        (1, "LEFT"),
        (1, "LEFT"),
        (6, "HARD_DROP")
    ]
    
    print(f"\nExecuting sequence: {' → '.join(name for _, name in sequence)}")
    
    for action, name in sequence:
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        if done:
            print(f"   Episode ended after {name}")
            break
    
    board_final = extract_board(obs)
    heights_final = get_column_heights(board_final)
    
    print(f"Final heights: {heights_final}")
    print_board(board_final, "Board after sequence")
    
    # Check if LEFT worked
    if heights_initial == heights_final:
        print(f"\n❌ WARNING: Heights unchanged - actions may not be working!")
    else:
        print(f"\n✅ Heights changed - actions are working")
    
    env.close()
    print("="*80)


if __name__ == "__main__":
    test_actions()
    test_specific_action_sequence()
    
    print("\n" + "="*80)
    print("📋 RECOMMENDATIONS:")
    print("="*80)
    print("""
If you see single-column stacking:
1. Use the updated train.py with --force_exploration flag
2. Use the updated reward_shaping.py (has anti-single-column penalties)
3. Increase epsilon_end to 0.1 (more exploration)
4. Train with: python train.py --force_exploration --epsilon_end 0.1 --reward_shaping aggressive

If actions aren't working at all:
1. Check your config.py action mapping
2. Verify tetris-gymnasium is latest version: pip install -U tetris-gymnasium
3. Try a different environment variant if available

To monitor training:
1. Watch for "Non-zero columns" in episode summaries
2. Should be 5-8 columns, not 1-2
3. Check action distribution - should have ~30% horizontal movement
    """)