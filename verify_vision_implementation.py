#!/usr/bin/env python3
"""
verify_vision_implementation.py
Quick test to verify your complete vision is working correctly
"""

import numpy as np
import sys


def test_current_implementation():
    """Test if complete vision is properly implemented"""
    print("🔍 VERIFYING COMPLETE VISION IMPLEMENTATION")
    print("="*60)

    # Test 1: Check config import
    try:
        from config import make_env
        print("✅ Config imported successfully")
    except ImportError as e:
        print(f"❌ Config import failed: {e}")
        return False

    # Test 2: Create environment and check observation space
    try:
        env = make_env(use_complete_vision=True, use_cnn=True)
        obs_shape = env.observation_space.shape
        print(f"✅ Environment created")
        print(f"   Observation shape: {obs_shape}")

        # Check if we have 4 channels
        if len(obs_shape) == 3 and obs_shape[-1] == 4:
            print("✅ 4-channel observation detected!")
        else:
            print(
                f"❌ Wrong observation shape! Expected (84, 84, 4), got {obs_shape}")
            return False

    except Exception as e:
        print(f"❌ Environment creation failed: {e}")
        return False

    # Test 3: Reset and analyze observation
    try:
        obs, info = env.reset(seed=42)
        print(f"\n📊 Observation Analysis:")

        for i in range(obs.shape[-1]):
            channel = obs[:, :, i]
            non_zero = np.sum(channel > 0.01)
            print(
                f"   Channel {i}: {non_zero} non-zero pixels (range: [{channel.min():.3f}, {channel.max():.3f}])")

        # Check if active piece channel has data
        active_channel = obs[:, :, 1]
        active_pixels = np.sum(active_channel > 0.01)

        if active_pixels > 0:
            print(f"\n✅ Active piece visible: {active_pixels} pixels")
        else:
            print(f"\n⚠️  WARNING: Active piece channel is empty!")

    except Exception as e:
        print(f"❌ Observation analysis failed: {e}")
        return False

    # Test 4: Check reward shaping
    print(f"\n🎯 Testing Reward Structure:")

    # Simulate a few steps
    total_reward = 0
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if info.get('lines_cleared', 0) > 0:
            print(f"   Step {step}: Lines cleared! Reward: {reward}")

        if terminated or truncated:
            break

    print(f"   Total reward from {step+1} steps: {total_reward}")

    env.close()

    # Test 5: Check if using complete vision in train.py
    print(f"\n📝 Configuration Summary:")
    print(f"   ✅ 4-channel observation space confirmed")
    print(f"   {'✅' if active_pixels > 0 else '❌'} Active piece channel has data")
    print(
        f"   Total non-zero pixels: {sum(np.sum(obs[:,:,i] > 0.01) for i in range(4))}")

    return True


def diagnose_training_issue():
    """Diagnose why training isn't working despite piece visibility"""
    print(f"\n🔧 DIAGNOSING TRAINING ISSUES")
    print("="*60)

    print("Possible issues:")
    print("\n1. REWARD SHAPING PROBLEM:")
    print("   Your rewards are 1000-2500 but NO lines cleared")
    print("   → The agent is getting reward from survival only")
    print("   → Line clear bonuses might be too small relative to survival")

    print("\n2. CONTINUING FROM OLD CHECKPOINT:")
    print("   You're at episode 6900 - this is continuation not fresh start")
    print("   → Old network weights trained on incomplete observations")
    print("   → Low epsilon (0.061) means little exploration")

    print("\n3. NETWORK ARCHITECTURE MISMATCH:")
    print("   If you changed from 1-channel to 4-channel observations")
    print("   → The old network might not handle new input correctly")

    print("\n🚨 RECOMMENDED FIXES:")
    print("\n1. START FRESH (Most Important):")
    print("   mkdir models_backup")
    print("   mv models/* models_backup/")
    print("   python train.py --episodes 200 --use_complete_vision")

    print("\n2. BOOST EPSILON TEMPORARILY:")
    print("   If continuing training, set epsilon back to 0.5-0.8")
    print("   The agent needs to explore with its new vision")

    print("\n3. CHECK REWARD SHAPING:")
    print("   Make sure line clear bonus is HUGE (100-500 per line)")
    print("   Reduce survival bonus to encourage risk-taking")

    print("\n4. USE THE EMERGENCY SCRIPT:")
    print("   python emergency_breakthrough_complete.py")
    print("   This has optimized settings for breaking through")


if __name__ == "__main__":
    print("Testing your complete vision implementation...")
    print()

    success = test_current_implementation()

    if success:
        print("\n✅ Complete vision appears to be implemented!")
        diagnose_training_issue()
    else:
        print("\n❌ Complete vision NOT properly implemented!")
        print("Make sure you're using the CompleteTetrisObservationWrapper")
