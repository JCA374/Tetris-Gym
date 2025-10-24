#!/usr/bin/env python3
# tests/test_visual_what_model_sees_fixed.py
"""
Simple visual test to show EXACTLY what the model sees vs what it SHOULD see.
This will make it immediately obvious if the active piece is missing.
FIXED: Uses current directory for output
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import make_env
    import gymnasium as gym
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def visualize_what_model_sees():
    """Show side-by-side comparison of what the model sees vs what it should see"""
    
    print("="*80)
    print("🔬 VISUAL TEST: What Does Your Model Actually See?")
    print("="*80)
    print("\nThis test will reveal if your model is 'blind' to the active piece.\n")
    
    # Create environment
    env = make_env(render_mode="rgb_array")
    obs, info = env.reset(seed=42)
    
    # Take a few steps to ensure there's an active piece
    for _ in range(3):
        obs, reward, terminated, truncated, info = env.step(1)  # Move right
        if terminated or truncated:
            obs, info = env.reset(seed=42)
    
    print(f"Observation shape: {obs.shape}")
    print(f"Observation dtype: {obs.dtype}")
    print(f"Value range: [{obs.min():.3f}, {obs.max():.3f}]")
    
    # Try to access raw game state from info
    print(f"\nInfo dictionary keys: {list(info.keys())}")
    
    # Determine output path - use current directory's tests folder
    output_dir = os.path.join(os.getcwd(), 'tests')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'what_model_sees.png')
    
    # Determine how to visualize based on observation shape
    if len(obs.shape) == 3:
        # Multi-channel observation
        channels = obs.shape[-1] if obs.shape[-1] < obs.shape[0] else obs.shape[0]
        print(f"\n📊 Your model receives {channels} channels")
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 10))
        
        # Determine channel axis (HWC vs CHW)
        if obs.shape[-1] < obs.shape[0]:  # HWC format
            h, w, c = obs.shape
            channel_axis = -1
        else:  # CHW format  
            c, h, w = obs.shape
            channel_axis = 0
        
        # Plot each channel
        for i in range(min(channels, 6)):
            ax = plt.subplot(2, 3, i+1)
            
            if channel_axis == -1:
                channel_data = obs[:, :, i]
            else:
                channel_data = obs[i, :, :]
            
            im = ax.imshow(channel_data, cmap='viridis', aspect='auto', interpolation='nearest')
            
            # Count active pixels
            active_pixels = np.sum(channel_data > 0.01)
            active_percent = 100 * active_pixels / channel_data.size
            
            # Determine what this channel likely represents
            if i == 0:
                title = f"Channel {i}: BOARD STATE"
                subtitle = f"(Placed pieces)"
            elif i == 1:
                title = f"Channel {i}: ACTIVE PIECE"
                if active_pixels == 0:
                    subtitle = f"EMPTY - PIECE NOT VISIBLE!"
                else:
                    subtitle = f"OK - {active_pixels} pixels"
            elif i == 2:
                title = f"Channel {i}: HOLDER"
                subtitle = f"({active_pixels} pixels)"
            elif i == 3:
                title = f"Channel {i}: QUEUE"
                subtitle = f"({active_pixels} pixels)"
            else:
                title = f"Channel {i}"
                subtitle = f"({active_pixels} pixels)"
            
            ax.set_title(f"{title}\n{subtitle}", fontsize=10, fontweight='bold')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # Add grid
            ax.set_xticks(np.arange(0, w, 1), minor=True)
            ax.set_yticks(np.arange(0, h, 1), minor=True)
            ax.grid(which='minor', color='gray', linestyle=':', linewidth=0.5, alpha=0.3)
        
        plt.suptitle('What Your Model Actually Sees (Current Observation)', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n💾 Visualization saved to: {output_path}")
        
        # Show diagnosis based on channels
        if channel_axis == -1:
            channel_1_pixels = np.sum(obs[:, :, 1] > 0.01) if channels > 1 else 0
        else:
            channel_1_pixels = np.sum(obs[1, :, :] > 0.01) if channels > 1 else 0
        
        print("\n" + "="*80)
        print("🔍 DIAGNOSIS")
        print("="*80)
        
        if channels >= 4 and channel_1_pixels > 0:
            print("\n✅ LOOKS EXCELLENT: 4-channel observation with active piece visible!")
            print("   - Model can see board, active piece, holder, and queue")
            print("\nSince observations are CORRECT, your 0.02 lines/ep issue is from:")
            print("   1. HYPERPARAMETERS:")
            print("      • Learning rate may be too high or too low")
            print("      • Epsilon decay may be wrong")
            print("      • Batch size may be suboptimal")
            print("\n   2. REWARD SHAPING:")
            print("      • Check if rewards properly incentivize line clearing")
            print("      • Verify penalties aren't too harsh")
            print("      • Ensure step rewards don't dominate")
            print("\n   3. TRAINING APPROACH:")
            print("      • May need MUCH longer training (20k-50k episodes)")
            print("      • Try different epsilon schedules")
            print("      • Experiment with different learning rates")
            print("\n   4. MODEL ARCHITECTURE:")
            print("      • Network may be too small (add layers)")
            print("      • Or too large (overfitting)")
            
        elif channels == 1:
            print("\n🔴 CRITICAL ISSUE: Only 1 channel!")
            print("   Model is missing active piece information")
            
        elif channels >= 2 and channel_1_pixels == 0:
            print("\n🔴 Channel 1 (active piece) is EMPTY!")
            print("   Wrapper exists but not extracting piece correctly")
    
    env.close()
    
    print("\n" + "="*80)


if __name__ == "__main__":
    visualize_what_model_sees()
