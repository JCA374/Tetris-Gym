#!/usr/bin/env python3
"""
Simple visual tool to check what the agent actually sees
"""

from gymnasium.envs.registration import register
import gymnasium as gym
from config import make_env
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


def visualize_agent_vision():
    """Create side-by-side visualization of raw vs processed observations"""

    # Register raw environment
    try:
        register(
            id="TetrisVisual-v0",
            entry_point="tetris_gymnasium.envs.tetris:Tetris",
        )
    except gym.error.Error:
        pass

    # Create both environments
    raw_env = gym.make("TetrisVisual-v0", render_mode="rgb_array")
    processed_env = make_env(frame_stack=4)

    # Reset with same seed
    raw_obs, _ = raw_env.reset(seed=42)
    processed_obs, _ = processed_env.reset(seed=42)

    print(f"Raw observation type: {type(raw_obs)}")
    print(f"Processed observation shape: {processed_obs.shape}")

    # Extract board from raw observation
    if isinstance(raw_obs, dict) and 'board' in raw_obs:
        raw_board = raw_obs['board']
        print(f"Raw board shape: {raw_board.shape}")
        print(f"Raw board unique values: {np.unique(raw_board)}")

        # Extract latest frame from processed observation
        if len(processed_obs.shape) == 3:
            latest_frame = processed_obs[:, :, -1]  # Last channel
        else:
            latest_frame = processed_obs

        print(f"Processed frame shape: {latest_frame.shape}")
        print(
            f"Processed frame range: [{latest_frame.min():.3f}, {latest_frame.max():.3f}]")

        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Raw board (actual Tetris grid)
        im1 = axes[0].imshow(raw_board, cmap='viridis', aspect='auto')
        axes[0].set_title(
            f'Raw Tetris Board\n{raw_board.shape}\nRange: [{raw_board.min()}, {raw_board.max()}]')
        axes[0].set_xlabel('Columns (width)')
        axes[0].set_ylabel('Rows (height)')

        # Add grid lines to show individual cells
        for i in range(raw_board.shape[0] + 1):
            axes[0].axhline(i - 0.5, color='white', linewidth=0.5, alpha=0.5)
        for j in range(raw_board.shape[1] + 1):
            axes[0].axvline(j - 0.5, color='white', linewidth=0.5, alpha=0.5)

        plt.colorbar(im1, ax=axes[0])

        # Processed frame (what the agent sees)
        im2 = axes[1].imshow(latest_frame, cmap='viridis', aspect='auto')
        axes[1].set_title(
            f'What Agent Sees\n{latest_frame.shape}\nRange: [{latest_frame.min():.3f}, {latest_frame.max():.3f}]')
        axes[1].set_xlabel('Pixels (width)')
        axes[1].set_ylabel('Pixels (height)')
        plt.colorbar(im2, ax=axes[1])

        # Overlay comparison (resize raw to match processed)
        import cv2
        raw_resized = cv2.resize(raw_board.astype(np.float32),
                                 (latest_frame.shape[1],
                                  latest_frame.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)

        # Normalize both for comparison
        if raw_resized.max() > 1:
            raw_norm = raw_resized / raw_resized.max()
        else:
            raw_norm = raw_resized

        if latest_frame.max() > 1:
            proc_norm = latest_frame / latest_frame.max()
        else:
            proc_norm = latest_frame

        # Create overlay
        overlay = np.zeros((*raw_norm.shape, 3))
        overlay[:, :, 0] = raw_norm      # Red channel = original
        overlay[:, :, 1] = proc_norm     # Green channel = processed
        # Blue channel stays 0

        # Where they match, you'll see yellow (red + green)
        # Where they differ, you'll see red (original only) or green (processed only)

        axes[2].imshow(overlay)
        axes[2].set_title(
            'Overlay Comparison\nYellow=Match, Red=Original, Green=Processed')
        axes[2].set_xlabel('Pixels')
        axes[2].set_ylabel('Pixels')

        plt.tight_layout()
        plt.savefig('agent_vision_check.png', dpi=150, bbox_inches='tight')
        plt.show()

        # Calculate similarity
        correlation = np.corrcoef(
            raw_norm.flatten(), proc_norm.flatten())[0, 1]
        print(f"\nSpatial correlation: {correlation:.4f}")

        if correlation > 0.9:
            print("✅ EXCELLENT: Agent sees board structure correctly")
        elif correlation > 0.7:
            print("⚠️  GOOD: Agent mostly sees board structure correctly")
        elif correlation > 0.5:
            print("❌ POOR: Agent sees distorted board structure")
        else:
            print("🚨 CRITICAL: Agent cannot see board structure at all")

        # Test with some steps
        print("\nTesting observation changes over time...")

        obs_changes = []
        for step in range(5):
            action = processed_env.action_space.sample()
            new_obs, reward, terminated, truncated, info = processed_env.step(
                action)

            # Calculate change magnitude
            change = np.abs(new_obs - processed_obs).mean()
            obs_changes.append(change)
            processed_obs = new_obs

            print(
                f"Step {step+1}: Action={action}, Change={change:.4f}, Reward={reward:.2f}")

            if terminated or truncated:
                processed_obs, _ = processed_env.reset()
                print("  (Reset due to game over)")

        avg_change = np.mean(obs_changes)
        print(f"\nAverage observation change: {avg_change:.4f}")

        if avg_change > 0.001:
            print("✅ Observations change meaningfully with actions")
        else:
            print("❌ Observations barely change - potential issue!")

    else:
        print("❌ Could not extract board from raw observation!")
        print(
            f"Raw observation keys: {list(raw_obs.keys()) if isinstance(raw_obs, dict) else 'Not a dict'}")

    # Cleanup
    raw_env.close()
    processed_env.close()


if __name__ == "__main__":
    print("🔍 Visual Board Verification")
    print("=" * 40)
    visualize_agent_vision()
    print("\n📊 Check 'agent_vision_check.png' for visual comparison")
