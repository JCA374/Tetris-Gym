#!/usr/bin/env python3
"""
Debug script to understand Tetris Gymnasium observations
"""

import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np


def debug_tetris_observations():
    """Debug what Tetris Gymnasium actually returns"""
    print("Debugging Tetris Gymnasium Observations")
    print("=" * 50)

    # Register environment
    try:
        register(
            id="TetrisDebug-v0",
            entry_point="tetris_gymnasium.envs.tetris:Tetris",
        )
        print("✅ Environment registered")
    except gym.error.Error:
        print("✅ Environment already registered")

    # Create raw environment
    env = gym.make("TetrisDebug-v0", render_mode="rgb_array")

    print(f"\nEnvironment info:")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")

    # Test reset
    obs, info = env.reset(seed=42)

    print(f"\nObservation after reset:")
    print(f"  Type: {type(obs)}")

    if isinstance(obs, dict):
        print(f"  Dict keys: {list(obs.keys())}")
        total_size = 0
        for key, value in obs.items():
            print(f"    {key}:")
            print(f"      Type: {type(value)}")
            print(
                f"      Shape: {value.shape if hasattr(value, 'shape') else 'N/A'}")
            print(
                f"      Dtype: {value.dtype if hasattr(value, 'dtype') else 'N/A'}")
            if hasattr(value, 'shape'):
                size = np.prod(value.shape)
                total_size += size
                print(f"      Size: {size}")
                print(f"      Range: [{value.min()}, {value.max()}]")
            else:
                total_size += 1
                print(f"      Value: {value}")

        print(f"  Total flattened size would be: {total_size}")

        # Test flattening manually
        print(f"\nTesting manual flattening:")
        flattened_parts = []
        for key in sorted(obs.keys()):
            value = obs[key]
            if hasattr(value, 'flatten'):
                flat = value.flatten()
                flattened_parts.append(flat)
                print(f"    {key}: flattened to {flat.shape}")
            else:
                flattened_parts.append(np.array([value]))
                print(f"    {key}: converted to array [{value}]")

        try:
            flattened = np.concatenate(flattened_parts)
            print(f"  ✅ Concatenation successful: {flattened.shape}")
            print(f"  Range: [{flattened.min()}, {flattened.max()}]")
        except Exception as e:
            print(f"  ❌ Concatenation failed: {e}")

    else:
        print(f"  Shape: {obs.shape}")
        print(f"  Dtype: {obs.dtype}")
        print(f"  Range: [{obs.min()}, {obs.max()}]")

    # Test a few steps
    print(f"\nTesting steps:")
    for i in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(
            f"  Step {i+1}: action={action}, reward={reward}, done={terminated or truncated}")

        if terminated or truncated:
            obs, info = env.reset()
            print(f"    Reset after termination")

    env.close()
    print("\n✅ Debug completed successfully!")


if __name__ == "__main__":
    debug_tetris_observations()
