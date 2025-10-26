#!/usr/bin/env python3
# diagnose_training.py - Find out what's wrong

import numpy as np
import sys
import os

print("🔍 TETRIS TRAINING DIAGNOSTICS")
print("="*60)

# Test 1: Check if horizontal distribution exists
print("\n1. Checking if horizontal distribution function exists...")
try:
    from src.reward_shaping import calculate_horizontal_distribution
    print("   ✅ Function exists!")
except ImportError:
    print("   ❌ MISSING! This is why pieces stack vertically!")
    print("   FIX: Add the function to src/reward_shaping.py")
    sys.exit(1)

# Test 2: Check if it's being called
print("\n2. Checking if reward functions use horizontal distribution...")
try:
    with open('src/reward_shaping.py', 'r') as f:
        content = f.read()
    
    functions = ['balanced_reward_shaping', 'aggressive_reward_shaping', 'positive_reward_shaping']
    for func in functions:
        func_start = content.find(f"def {func}")
        if func_start == -1:
            continue
        func_end = content.find("\ndef ", func_start + 1)
        func_code = content[func_start:func_end] if func_end != -1 else content[func_start:]
        
        if 'calculate_horizontal_distribution' in func_code:
            print(f"   ✅ {func} calls it")
        else:
            print(f"   ❌ {func} does NOT call it!")
except Exception as e:
    print(f"   ⚠️  Could not check: {e}")

# Test 3: Test reward normalization
print("\n3. Testing reward magnitude...")
try:
    from src.reward_shaping import balanced_reward_shaping, extract_board_from_obs
    
    # Create test board - moderately filled
    test_board = np.zeros((20, 10))
    test_board[-5:, :] = 1  # Fill bottom 5 rows
    
    info = {'lines_cleared': 0}
    reward = balanced_reward_shaping(test_board, 0, 0, False, info)
    
    print(f"   Test reward: {reward:.2f}")
    
    if reward < -1000:
        print(f"   ❌ REWARD TOO NEGATIVE! (expected -50 to -100)")
        print(f"   This means board normalization is broken")
    elif reward > -200:
        print(f"   ✅ Reward in normal range")
    else:
        print(f"   ⚠️  Reward somewhat high but might work")
        
except Exception as e:
    print(f"   ❌ Test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Test board extraction and normalization
print("\n4. Testing board extraction and normalization...")
try:
    from src.reward_shaping import extract_board_from_obs
    
    # Test with dict observation (like from env)
    test_obs_dict = {'board': np.ones((20, 10)) * 255}
    extracted = extract_board_from_obs(test_obs_dict)
    
    print(f"   Input max value: 255")
    print(f"   Extracted max value: {extracted.max()}")
    print(f"   Extracted min value: {extracted.min()}")
    
    if extracted.max() > 1.0:
        print(f"   ❌ NORMALIZATION FAILED! Board not normalized to 0-1")
        print(f"   This causes rewards to be 100-255x too large!")
    else:
        print(f"   ✅ Normalization working")
        
except Exception as e:
    print(f"   ❌ Test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Quick environment test
print("\n5. Testing environment observations...")
try:
    from config import make_env
    
    env = make_env(render_mode="rgb_array")
    obs, info = env.reset()
    
    print(f"   Observation shape: {obs.shape}")
    print(f"   Observation dtype: {obs.dtype}")
    print(f"   Observation range: [{obs.min()}, {obs.max()}]")
    
    if obs.max() > 1.0:
        print(f"   ⚠️  Observation values > 1.0")
        print(f"   Make sure reward shaping normalizes these!")
    else:
        print(f"   ✅ Observation in 0-1 range")
    
    env.close()
    
except Exception as e:
    print(f"   ❌ Test failed: {e}")

# Test 6: Test with actual training step
print("\n6. Simulating training step...")
try:
    from config import make_env
    from src.reward_shaping import balanced_reward_shaping
    
    env = make_env(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Take random action
    action = env.action_space.sample()
    next_obs, env_reward, terminated, truncated, info = env.step(action)
    
    # Apply reward shaping
    shaped_reward = balanced_reward_shaping(obs, action, env_reward, terminated or truncated, info)
    
    print(f"   Environment reward: {env_reward:.2f}")
    print(f"   Shaped reward: {shaped_reward:.2f}")
    
    if shaped_reward < -1000:
        print(f"   ❌ SHAPED REWARD TOO NEGATIVE!")
    elif shaped_reward > -100:
        print(f"   ✅ Shaped reward in normal range")
    else:
        print(f"   ⚠️  Shaped reward a bit high")
    
    env.close()
    
except Exception as e:
    print(f"   ❌ Test failed: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "="*60)
print("📋 DIAGNOSTIC SUMMARY")
print("="*60)

print("\n🔧 RECOMMENDED FIXES:")
print("1. If horizontal_distribution is missing: Add it to reward_shaping.py")
print("2. If reward functions don't call it: Add the calls")
print("3. If normalization failed: Fix extract_board_from_obs()")
print("4. If rewards still < -1000: Force board normalization")

print("\n📊 Expected vs Current:")
print("   Reward range: -50 to +500 (currently: -2000)")
print("   Episode length: 200+ steps (currently: 9-14)")
print("   Lines cleared: 0.1-1.0 by ep 5000 (currently: 0)")

print("\n💡 After fixing, test with:")
print("   python train.py --episodes 10 --log_freq 2 --force_fresh")
print("   Should see rewards in -50 to +200 range")