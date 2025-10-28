#investigate_line_clearing.py
"""Deep investigation: Why aren't lines being cleared?"""

import sys
sys.path.insert(0, '/home/claude')

from config import make_env
import numpy as np

def check_if_row_can_be_filled():
    """Check if we can ever get a full row"""
    print("="*80)
    print("🔬 INVESTIGATING LINE CLEARING MECHANICS")
    print("="*80)
    
    env = make_env(render_mode=None, use_complete_vision=True)
    
    # Strategy: Try to fill bottom row by placing pieces carefully
    obs, _ = env.reset(seed=42)
    
    print("\n1️⃣ Checking initial board state:")
    board = obs[:, :, 0]  # Board channel
    print(f"   Board shape: {board.shape}")
    print(f"   Non-zero cells: {np.count_nonzero(board)}")
    print(f"   Bottom row: {board[-1, :]}")
    
    # Check if bottom row ever gets filled
    print("\n2️⃣ Attempting to fill bottom row...")
    
    rows_fullness = []
    max_fullness_seen = 0
    
    for episode in range(10):
        obs, _ = env.reset()
        done = False
        step = 0
        
        while not done and step < 200:
            # Use mostly hard drops to fill quickly
            if np.random.random() < 0.8:
                action = 5  # Hard drop
            else:
                action = np.random.choice([0, 1, 2])  # NOOP, LEFT, RIGHT
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1
            
            # Check board state
            board = obs[:, :, 0]
            
            # Check each row's fullness
            for row_idx in range(board.shape[0]):
                row = board[row_idx, :]
                fullness = np.count_nonzero(row)
                max_fullness_seen = max(max_fullness_seen, fullness)
                
                if fullness > 0:
                    rows_fullness.append(fullness)
                
                if fullness == 10:  # Full row!
                    print(f"   🎯 Full row detected at row {row_idx}, step {step}")
                    print(f"      Row contents: {row}")
                    print(f"      Lines cleared: {info.get('number_of_lines', 0)}")
            
            if info.get('number_of_lines', 0) > 0:
                print(f"   ✅ Lines cleared: {info.get('number_of_lines', 0)}")
                break
        
        if done:
            print(f"   Episode {episode}: {step} steps, game over")
    
    env.close()
    
    print(f"\n📊 Analysis:")
    print(f"   Max cells filled in any row: {max_fullness_seen}/10")
    
    if max_fullness_seen < 10:
        print(f"   ❌ PROBLEM: Never achieved a full row!")
        print(f"      The game ends before filling any row completely.")
        print(f"      This explains why 0 lines are cleared.")
    
    if len(rows_fullness) > 0:
        print(f"   Average row fullness (non-empty rows): {np.mean(rows_fullness):.1f}/10")


def test_piece_placement():
    """Test if pieces are actually being placed"""
    print("\n" + "="*80)
    print("🔬 TESTING PIECE PLACEMENT")
    print("="*80)
    
    env = make_env(render_mode=None, use_complete_vision=True)
    obs, _ = env.reset(seed=123)
    
    # Count pieces before and after hard drop
    board_before = obs[:, :, 0].copy()
    pieces_before = np.count_nonzero(board_before)
    
    print(f"\n   Before hard drop: {pieces_before} cells filled")
    
    # Hard drop should place the piece
    obs, reward, term, trunc, info = env.step(5)
    
    board_after = obs[:, :, 0].copy()
    pieces_after = np.count_nonzero(board_after)
    
    print(f"   After hard drop: {pieces_after} cells filled")
    print(f"   Difference: +{pieces_after - pieces_before} cells")
    print(f"   Reward: {reward}")
    
    if pieces_after == pieces_before:
        print(f"\n   ❌ PROBLEM: Piece wasn't placed!")
        print(f"      Hard drop should add piece to board.")
    elif pieces_after > pieces_before:
        print(f"\n   ✅ Piece was placed (+{pieces_after - pieces_before} cells)")
        
        # Now test multiple placements
        print(f"\n   Testing 5 consecutive placements:")
        for i in range(5):
            before = np.count_nonzero(obs[:, :, 0])
            obs, reward, term, trunc, info = env.step(5)
            after = np.count_nonzero(obs[:, :, 0])
            
            print(f"      Drop {i+1}: {before} -> {after} cells (+{after-before})")
            
            if term or trunc:
                print(f"      Game over after {i+1} drops")
                break
    
    env.close()


def check_reward_for_lines():
    """Check if environment gives reward for lines"""
    print("\n" + "="*80)
    print("🔬 CHECKING REWARD STRUCTURE")
    print("="*80)
    
    env = make_env(render_mode=None, use_complete_vision=True)
    
    print("\n   Running 100 steps and checking for any line-clearing rewards...")
    
    rewards_received = []
    line_counts = []
    
    for episode in range(5):
        obs, _ = env.reset()
        done = False
        step = 0
        
        while not done and step < 100:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1
            
            rewards_received.append(reward)
            lines = info.get('number_of_lines', 0)
            line_counts.append(lines)
            
            if lines > 0:
                print(f"   🎉 Episode {episode}, step {step}: {lines} lines, reward={reward}")
    
    env.close()
    
    print(f"\n📊 Reward analysis:")
    print(f"   Min reward: {min(rewards_received):.2f}")
    print(f"   Max reward: {max(rewards_received):.2f}")
    print(f"   Mean reward: {np.mean(rewards_received):.2f}")
    print(f"   Total lines seen: {sum(line_counts)}")
    
    unique_rewards = set(rewards_received)
    print(f"   Unique reward values seen: {sorted(unique_rewards)}")


def main():
    """Run all investigations"""
    check_if_row_can_be_filled()
    test_piece_placement()
    check_reward_for_lines()
    
    print("\n" + "="*80)
    print("🎯 CONCLUSION")
    print("="*80)
    print("""
Based on the tests above, the issue is likely:

1. If max row fullness < 10:
   → Pieces die before filling a complete row
   → Need to adjust game difficulty or piece placement strategy

2. If pieces aren't being placed:
   → Action mapping is wrong
   → Check tetris-gymnasium documentation

3. If lines are cleared but reward=0:
   → Reward structure issue
   → Check reward_mapping in environment config

Next steps:
- Review the test outputs above
- Check tetris-gymnasium version: pip show tetris-gymnasium
- Consider using a simpler Tetris environment (gym-tetris)
    """)


if __name__ == "__main__":
    main()


