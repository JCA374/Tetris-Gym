#!/usr/bin/env python3
"""
quick_vision_check.py

Ultra-quick check to see if your model can actually see the Tetris board
Run this first before any other diagnostics
"""

import sys
import os
import numpy as np

def test_basic_imports():
    """Test if all required packages are available"""
    print("🔧 Testing basic imports...")
    
    missing = []
    try:
        import gymnasium as gym
        print("   ✅ gymnasium")
    except ImportError:
        missing.append("gymnasium")
        print("   ❌ gymnasium")
    
    try:
        import tetris_gymnasium
        print("   ✅ tetris_gymnasium")
    except ImportError:
        missing.append("tetris-gymnasium")
        print("   ❌ tetris_gymnasium")
    
    try:
        import torch
        print("   ✅ torch")
    except ImportError:
        missing.append("torch")
        print("   ❌ torch")
    
    try:
        import matplotlib
        print("   ✅ matplotlib")
    except ImportError:
        missing.append("matplotlib")
        print("   ❌ matplotlib")
    
    try:
        import cv2
        print("   ✅ opencv (cv2)")
    except ImportError:
        missing.append("opencv-python")
        print("   ❌ opencv")
    
    if missing:
        print(f"\n❌ Missing packages: {missing}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    print("   ✅ All packages available")
    return True

def test_environment_creation():
    """Test basic environment creation"""
    print("\n🏗️  Testing environment creation...")
    
    try:
        # Test config import
        from config import make_env
        print("   ✅ config.py imported")
        
        # Test environment creation
        env = make_env(use_cnn=True, frame_stack=1)
        print("   ✅ Environment created")
        print(f"   📊 Observation space: {env.observation_space}")
        print(f"   📊 Action space: {env.action_space}")
        
        # Test reset
        obs, info = env.reset(seed=42)
        print(f"   ✅ Environment reset: {obs.shape} {obs.dtype}")
        print(f"   📊 Observation range: [{obs.min():.3f}, {obs.max():.3f}]")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"   ❌ Environment creation failed: {e}")
        return False

def test_agent_creation():
    """Test agent creation and compatibility"""
    print("\n🤖 Testing agent creation...")
    
    try:
        from config import make_env
        from src.agent import Agent
        
        # Create environment
        env = make_env(use_cnn=True, frame_stack=1)
        
        # Create agent
        agent = Agent(
            obs_space=env.observation_space,
            action_space=env.action_space,
            reward_shaping="simple"
        )
        print("   ✅ Agent created")
        print(f"   📊 Device: {agent.device}")
        
        # Test action selection
        obs, info = env.reset(seed=42)
        action = agent.select_action(obs)
        print(f"   ✅ Action selection: {action}")
        
        # Test experience storage
        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.remember(obs, action, reward, next_obs, terminated or truncated, info)
        print(f"   ✅ Experience storage: {len(agent.memory)} experiences")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"   ❌ Agent creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_board_vision():
    """Critical test: Can the agent actually see the board structure?"""
    print("\n👁️  Testing board vision (CRITICAL)...")
    
    try:
        from config import make_env
        import gymnasium as gym
        from gymnasium.envs.registration import register
        
        # Create raw environment to compare
        try:
            register(
                id="TetrisQuickTest-v0",
                entry_point="tetris_gymnasium.envs.tetris:Tetris",
            )
        except gym.error.Error:
            pass
        
        raw_env = gym.make("TetrisQuickTest-v0", render_mode="rgb_array")
        processed_env = make_env(use_cnn=True, frame_stack=1)
        
        # Reset both with same seed
        raw_obs, _ = raw_env.reset(seed=42)
        processed_obs, _ = processed_env.reset(seed=42)
        
        print(f"   📊 Raw observation type: {type(raw_obs)}")
        if isinstance(raw_obs, dict):
            print(f"   📊 Raw dict keys: {list(raw_obs.keys())}")
            if 'board' in raw_obs:
                board = raw_obs['board']
                print(f"   📊 Raw board shape: {board.shape}")
                print(f"   📊 Raw board range: [{board.min()}, {board.max()}]")
                
                # Check board dimensions
                if board.shape == (20, 10):
                    print("   ⚠️  WARNING: 20×10 board detected (may be wrong)")
                elif board.shape == (24, 18):
                    print("   ✅ 24×18 board detected (correct)")
                else:
                    print(f"   ❓ Unexpected board shape: {board.shape}")
        
        print(f"   📊 Processed observation: {processed_obs.shape} {processed_obs.dtype}")
        
        # Check if processing preserves information
        if len(processed_obs.shape) == 3:
            frame = processed_obs[:, :, -1]  # Last channel
            non_zero_pixels = np.sum(frame > 0.01)
            total_pixels = frame.size
            coverage = non_zero_pixels / total_pixels
            print(f"   📊 Board coverage in processed image: {coverage:.1%}")
            
            if coverage < 0.05:
                print("   🚨 CRITICAL: Board barely visible in processed image!")
                return False
            elif coverage > 0.8:
                print("   🚨 WARNING: Too much board coverage (may be distorted)")
            else:
                print("   ✅ Board coverage looks reasonable")
        
        # Test observation consistency across steps
        changes = []
        for step in range(3):
            action = processed_env.action_space.sample()
            new_obs, reward, terminated, truncated, info = processed_env.step(action)
            change = np.abs(new_obs - processed_obs).mean()
            changes.append(change)
            processed_obs = new_obs
            
            if terminated or truncated:
                processed_obs, _ = processed_env.reset()
        
        avg_change = np.mean(changes)
        print(f"   📊 Average observation change per step: {avg_change:.4f}")
        
        if avg_change < 0.001:
            print("   🚨 CRITICAL: Observations barely change with actions!")
            return False
        else:
            print("   ✅ Observations change meaningfully with actions")
        
        raw_env.close()
        processed_env.close()
        return True
        
    except Exception as e:
        print(f"   ❌ Board vision test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run quick vision check"""
    print("⚡ QUICK VISION CHECK FOR TETRIS AI")
    print("="*60)
    print("Running essential tests to see if your model can see the board...")
    print()
    
    # Track results
    results = []
    
    # Test 1: Basic imports
    results.append(test_basic_imports())
    
    # Test 2: Environment creation
    results.append(test_environment_creation())
    
    # Test 3: Agent creation
    results.append(test_agent_creation())
    
    # Test 4: Board vision (most critical)
    results.append(test_board_vision())
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("\n" + "="*60)
    print("📋 QUICK CHECK RESULTS")
    print("="*60)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✅ BASIC VISION SYSTEM WORKING!")
        print("\nIf training is still stuck, the issue is likely:")
        print("   • Hyperparameters (epsilon too low/high)")
        print("   • Reward shaping too weak")
        print("   • Need plateau breaking techniques")
        print("\n📋 Next steps:")
        print("   1. Check training logs: python analyze_current_training.py")
        print("   2. Run full diagnostics: python run_complete_diagnostics.py")
        print("   3. Try plateau breaker: python break_plateau_train.py")
        
    elif passed >= total - 1:
        print("⚠️  MOSTLY WORKING - Minor issues detected")
        print("\n📋 Quick fixes needed:")
        print("   • Fix the failing test above")
        print("   • Then rerun this check")
        
    else:
        print("🚨 CRITICAL ISSUES - Vision system not working!")
        print("\n📋 Emergency fixes:")
        if not results[0]:  # Imports failed
            print("   1. Install missing packages (see above)")
        if not results[1]:  # Environment failed
            print("   2. Fix config.py environment setup")
        if not results[2]:  # Agent failed
            print("   3. Fix src/agent.py compatibility")
        if not results[3]:  # Vision failed
            print("   4. Fix board preprocessing pipeline")
            
        print("\n🔧 Run full diagnostics after fixes:")
        print("   python run_complete_diagnostics.py")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)