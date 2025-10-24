#!/usr/bin/env python3
# tests/test_quick_check.py
"""
QUICK 30-SECOND DIAGNOSTIC
Run this first to immediately see if active piece is missing.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def quick_check():
    print("="*70)
    print("🔍 QUICK TETRIS OBSERVATION CHECK")
    print("="*70)
    print("\nChecking if your model can see the active piece...\n")
    
    try:
        from config import make_env
        
        # Create environment
        env = make_env(render_mode="rgb_array")
        obs, info = env.reset(seed=42)
        
        # Take a few steps
        for _ in range(3):
            obs, _, _, _, info = env.step(1)  # Move right
        
        print(f"Observation shape: {obs.shape}")
        print(f"Observation dtype: {obs.dtype}")
        print(f"Value range: [{obs.min():.3f}, {obs.max():.3f}]")
        
        # Determine format and check for piece
        if len(obs.shape) == 3:
            # Multi-channel
            if obs.shape[-1] < obs.shape[0]:  # HWC format
                h, w, c = obs.shape
                print(f"\n✅ Multi-channel observation: {c} channels (HWC format)")
                
                # Check each channel
                for i in range(c):
                    channel = obs[:, :, i]
                    active_pixels = np.sum(channel > 0.01)
                    percent = 100 * active_pixels / channel.size
                    
                    if i == 0:
                        name = "Board"
                    elif i == 1:
                        name = "Active Piece"
                    elif i == 2:
                        name = "Holder"
                    elif i == 3:
                        name = "Queue"
                    else:
                        name = f"Channel {i}"
                    
                    status = "✅" if active_pixels > 0 else "❌"
                    print(f"{status} Channel {i} ({name}): {active_pixels} pixels ({percent:.1f}%)")
                
                # Critical check: Channel 1 (active piece)
                if c > 1:
                    active_piece_pixels = np.sum(obs[:, :, 1] > 0.01)
                    if active_piece_pixels == 0:
                        print("\n" + "="*70)
                        print("🔴 PROBLEM FOUND!")
                        print("="*70)
                        print("Channel 1 (Active Piece) is EMPTY!")
                        print("Your model CANNOT see the piece it's supposed to place.")
                        print("\nThis explains why you only clear 0.02 lines per episode.")
                        print("\n💡 SOLUTION:")
                        print("1. Fix your observation wrapper to include active_tetromino_mask")
                        print("2. Ensure it's placed in channel 1 of the observation")
                        print("3. Retrain your model from scratch")
                        print("\n📚 Read tests/OBSERVATION_ANALYSIS.md for detailed fix")
                        return False
                    else:
                        print("\n" + "="*70)
                        print("✅ ACTIVE PIECE IS VISIBLE!")
                        print("="*70)
                        print(f"Channel 1 has {active_piece_pixels} active pixels.")
                        print("\nYour observation looks correct.")
                        print("If performance is still poor, the issue is likely:")
                        print("  - Hyperparameters (learning rate, epsilon decay)")
                        print("  - Reward shaping")
                        print("  - Training duration")
                        print("\nRun full diagnostic: python tests/test_observation_integrity.py")
                        return True
                else:
                    print("\n" + "="*70)
                    print("🔴 PROBLEM FOUND!")
                    print("="*70)
                    print("Only 1 channel detected!")
                    print("Your model only sees the BOARD, not the active piece.")
                    print("\nThis explains the 0.02 lines/episode plateau.")
                    print("\n💡 SOLUTION:")
                    print("Enable multi-channel observations in config.py")
                    return False
                    
            else:  # CHW format
                c, h, w = obs.shape
                print(f"\n✅ Multi-channel observation: {c} channels (CHW format)")
                
                for i in range(c):
                    channel = obs[i, :, :]
                    active_pixels = np.sum(channel > 0.01)
                    percent = 100 * active_pixels / channel.size
                    
                    if i == 0:
                        name = "Board"
                    elif i == 1:
                        name = "Active Piece"
                    elif i == 2:
                        name = "Holder"
                    elif i == 3:
                        name = "Queue"
                    else:
                        name = f"Channel {i}"
                    
                    status = "✅" if active_pixels > 0 else "❌"
                    print(f"{status} Channel {i} ({name}): {active_pixels} pixels ({percent:.1f}%)")
                
                if c > 1:
                    active_piece_pixels = np.sum(obs[1, :, :] > 0.01)
                    if active_piece_pixels == 0:
                        print("\n" + "="*70)
                        print("🔴 PROBLEM FOUND!")
                        print("="*70)
                        print("Channel 1 (Active Piece) is EMPTY!")
                        print("Your model CANNOT see the piece it's supposed to place.")
                        return False
                    else:
                        print("\n" + "="*70)
                        print("✅ ACTIVE PIECE IS VISIBLE!")
                        print("="*70)
                        return True
                else:
                    print("\n" + "="*70)
                    print("🔴 PROBLEM FOUND!")
                    print("="*70)
                    print("Only 1 channel detected!")
                    return False
                    
        elif len(obs.shape) == 2:
            # Single 2D array
            print(f"\n⚠️  Single 2D observation: {obs.shape}")
            print("\n" + "="*70)
            print("🔴 PROBLEM FOUND!")
            print("="*70)
            print("Your model only receives a 2D array (board only).")
            print("The active piece is NOT visible.")
            print("\nThis is why you're stuck at 0.02 lines/episode.")
            print("\n💡 SOLUTION:")
            print("1. Modify config.py to use multi-channel observations")
            print("2. Include: board, active_piece, holder, queue")
            print("3. Should result in shape (H, W, 4)")
            return False
            
        else:
            # Feature vector
            print(f"\n📊 Feature vector: {obs.shape[0]} features")
            print("\n⚠️  Cannot visually verify active piece in feature vector.")
            print("Recommendation: Switch to image-based observations for easier debugging.")
            return None
        
        env.close()
        
    except Exception as e:
        print(f"\n❌ Error during check: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = quick_check()
    
    print("\n" + "="*70)
    if result is False:
        print("❌ CRITICAL ISSUE DETECTED")
        print("\n📖 Next steps:")
        print("1. Read: tests/OBSERVATION_ANALYSIS.md")
        print("2. Run: python tests/test_visual_what_model_sees.py")
        print("3. Run: python tests/test_observation_integrity.py")
        print("4. Fix observation wrapper")
        print("5. Retrain model")
    elif result is True:
        print("✅ OBSERVATIONS LOOK GOOD")
        print("\n📖 If performance is still poor, investigate:")
        print("1. Hyperparameters (learning rate, epsilon)")
        print("2. Reward shaping")
        print("3. Model architecture")
        print("4. Training duration")
    else:
        print("❓ UNCLEAR RESULT")
        print("\n📖 Run full diagnostics:")
        print("python tests/test_observation_integrity.py")
    print("="*70)
