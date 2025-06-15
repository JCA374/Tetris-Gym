#!/usr/bin/env python3
"""
Quick fix script for Tetris Gymnasium setup issues
"""

import os
import shutil
import sys


def backup_files():
    """Backup current files"""
    print("Creating backup of current files...")

    backup_dir = "backup_before_fix"
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)

    files_to_backup = [
        "config.py",
        "src/agent.py",
        "test_setup.py"
    ]

    for file_path in files_to_backup:
        if os.path.exists(file_path):
            backup_path = os.path.join(backup_dir, os.path.basename(file_path))
            shutil.copy2(file_path, backup_path)
            print(f"  Backed up: {file_path} -> {backup_path}")


def show_fixes_needed():
    """Show what needs to be fixed"""
    print("\n" + "="*60)
    print("FIXES NEEDED FOR YOUR TETRIS GYMNASIUM SETUP")
    print("="*60)

    print("\n🔧 Issues found in your setup:")
    print("1. ❌ config.py missing make_env function")
    print("2. ❌ Tetris Gymnasium returns dict observations (not arrays)")
    print("3. ❌ Environment doesn't support board_height/width parameters")
    print("4. ❌ Agent needs updating for dict observations")
    print("5. ❌ Action space size is 8 (not 7) for Tetris")

    print("\n✅ Solutions provided:")
    print("1. ✅ New config.py with observation wrappers")
    print("2. ✅ Updated agent.py with better preprocessing")
    print("3. ✅ Fixed test_setup.py that handles dict observations")
    print("4. ✅ Tetris-specific test script")

    print("\n📋 To fix your setup:")
    print("1. Replace config.py with the updated version")
    print("2. Replace src/agent.py with the updated version")
    print("3. Run the new test scripts to verify")
    print("4. Start training!")


def check_current_status():
    """Check what's currently working"""
    print("\n🔍 Checking current status...")

    try:
        import gymnasium as gym
        import tetris_gymnasium

        # Test basic environment
        env = gym.make("tetris_gymnasium/Tetris", render_mode="rgb_array")
        obs, info = env.reset()

        print(f"✅ Tetris Gymnasium works")
        print(f"   Observation type: {type(obs)}")
        print(f"   Action space: {env.action_space}")

        if isinstance(obs, dict):
            print(f"   Dict keys: {list(obs.keys())}")
            total_elements = sum(np.prod(v.shape) for v in obs.values())
            print(f"   Total elements if flattened: {total_elements}")

        env.close()

    except Exception as e:
        print(f"❌ Basic environment test failed: {e}")
        return False

    # Check if our fixes are already applied
    try:
        from config import make_env
        print("✅ make_env function exists")

        env = make_env()
        obs, info = env.reset()
        print(f"✅ make_env works - obs shape: {obs.shape}")
        env.close()

        print("🎉 Your setup is already fixed!")
        return True

    except ImportError:
        print("❌ make_env function missing - need to update config.py")
        return False
    except Exception as e:
        print(f"❌ make_env has issues: {e}")
        return False


def show_file_contents():
    """Show what the updated files should contain"""
    print("\n📁 Updated file contents needed:")

    print("\n1. config.py should include:")
    print("   - TetrisObservationWrapper class")
    print("   - TetrisPreprocessWrapper class")
    print("   - FrameStackWrapper class")
    print("   - make_env() function")
    print("   - test_environment() function")

    print("\n2. src/agent.py should include:")
    print("   - model_type parameter in __init__")
    print("   - create_model() import and usage")
    print("   - Better _preprocess_state() method")

    print("\n3. test_setup.py should include:")
    print("   - Proper dict observation handling")
    print("   - Correct action space size (8)")
    print("   - Fixed environment configuration tests")


def main():
    """Main fix script"""
    print("Tetris Gymnasium Setup Fixer")
    print("="*60)

    # Check current status
    status_ok = check_current_status()

    if status_ok:
        print("\n✅ Your setup appears to be working!")
        print("Run 'python train.py --episodes 50' to start training.")
        return

    # Show what's wrong and how to fix it
    show_fixes_needed()

    # Offer to backup
    backup_choice = input(
        "\n📂 Create backup of current files? (y/n): ").lower()
    if backup_choice == 'y':
        backup_files()

    show_file_contents()

    print("\n" + "="*60)
    print("TO COMPLETE THE FIX:")
    print("="*60)
    print("1. Copy the new config.py content from the artifacts above")
    print("2. Copy the new src/agent.py content from the artifacts above")
    print("3. Run: python test_tetris_setup.py")
    print("4. If tests pass, run: python train.py --episodes 50")
    print("\nThe artifacts contain the complete, corrected file contents.")


if __name__ == "__main__":
    # Add numpy import for the check
    try:
        import numpy as np
    except ImportError:
        print("Installing numpy...")
        os.system("pip install numpy")
        import numpy as np

    main()
