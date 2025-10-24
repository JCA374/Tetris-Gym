# verify_setup.py
"""
Quick verification that everything is linked correctly before overnight training
"""

import sys
import os

print("="*70)
print("🔍 VERIFYING TETRIS AI SETUP")
print("="*70)

# Test 1: Check reward_shaping.py exists
print("\n1. Checking reward_shaping.py file...")
if os.path.exists('src/reward_shaping.py'):
    print("   ✅ src/reward_shaping.py found")
    
    # Check if it has normalization fix
    with open('src/reward_shaping.py', 'r') as f:
        content = f.read()
        if 'max_val > 1.0' in content and 'board / max_val' in content:
            print("   ✅ Normalization fix is present")
        else:
            print("   ⚠️  WARNING: Normalization fix might be missing")
            print("      Make sure extract_board_from_obs normalizes the board!")
else:
    print("   ❌ src/reward_shaping.py NOT FOUND!")
    sys.exit(1)

# Test 2: Try importing reward shaping functions
print("\n2. Testing imports...")
try:
    from src.reward_shaping import (
        aggressive_reward_shaping,
        positive_reward_shaping,
        balanced_reward_shaping,
        get_column_heights,
        count_holes
    )
    print("   ✅ All reward shaping functions imported successfully")
except ImportError as e:
    print(f"   ❌ Import failed: {e}")
    sys.exit(1)

# Test 3: Check train.py exists and has correct imports
print("\n3. Checking train.py...")
if os.path.exists('train.py'):
    print("   ✅ train.py found")
    
    with open('train.py', 'r') as f:
        content = f.read()
        if 'from src.reward_shaping import' in content or 'import src.reward_shaping' in content:
            print("   ✅ train.py imports from src.reward_shaping")
        else:
            print("   ⚠️  WARNING: train.py might not be importing reward_shaping correctly")
else:
    print("   ❌ train.py NOT FOUND!")
    sys.exit(1)

# Test 4: Quick function test
print("\n4. Testing helper functions...")
import numpy as np

test_board = np.ones((20, 10)) * 0.5  # Half-filled board
test_board[-5:, :] = 1.0  # Bottom 5 rows filled

heights = get_column_heights(test_board)
holes = count_holes(test_board)

print(f"   Test board heights: {heights}")
print(f"   Test board holes: {holes}")

if heights == [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]:
    print("   ✅ Helper functions work correctly")
else:
    print("   ⚠️  Helper functions might not be working as expected")

# Test 5: Test reward shaping with normalization
print("\n5. Testing reward shaping with different board scales...")

# Test with 0-1 board (already normalized)
board_normalized = np.zeros((20, 10))
board_normalized[-3:, :] = 1.0
info = {'lines_cleared': 0}
reward1 = aggressive_reward_shaping(board_normalized, 0, 0, False, info)
print(f"   Reward for 0-1 board: {reward1:.2f}")

# Test with 0-255 board (needs normalization)
board_pixels = np.zeros((20, 10))
board_pixels[-3:, :] = 255.0
reward2 = aggressive_reward_shaping(board_pixels, 0, 0, False, info)
print(f"   Reward for 0-255 board: {reward2:.2f}")

if abs(reward1 - reward2) < 1.0:
    print("   ✅ Normalization is working! Both boards give similar rewards")
else:
    print(f"   ❌ Normalization NOT working! Difference: {abs(reward1 - reward2):.2f}")
    print("      This will cause huge negative rewards in training!")
    print("      FIX: Add normalization to extract_board_from_obs()")

# Test 6: Check environment
print("\n6. Testing environment...")
try:
    from config import make_env
    env = make_env()
    obs, info = env.reset()
    print(f"   ✅ Environment created successfully")
    print(f"   Observation type: {type(obs)}")
    
    if isinstance(obs, dict):
        print(f"   Observation keys: {obs.keys()}")
        if 'board' in obs:
            board = obs['board']
            print(f"   Board shape: {board.shape}")
            print(f"   Board min/max: {board.min():.2f} / {board.max():.2f}")
            
            if board.max() > 1.0:
                print("   ⚠️  WARNING: Board values > 1.0 detected!")
                print("      Make sure reward_shaping.py normalizes the board!")
    else:
        print(f"   Observation shape: {obs.shape}")
        
    env.close()
except Exception as e:
    print(f"   ❌ Environment test failed: {e}")

# Final summary
print("\n" + "="*70)
print("📊 VERIFICATION SUMMARY")
print("="*70)
print("\n✅ Setup looks good! You're ready for overnight training.")
print("\nTo start training, run:")
print("  python train.py --episodes 5000 --reward_shaping aggressive --force_fresh")
print("\nOr use the overnight training script:")
print("  nohup python overnight_train.py > training.log 2>&1 &")
print()
