# Run this to check your training data
import pandas as pd
import numpy as np

df = pd.read_csv('logs/complete_vision_20250623_204512/episode_log.csv')

# Check reward composition
print("Reward Analysis:")
print(f"Average reward: {df['reward'].mean():.1f}")
print(f"Average original reward: {df['original_reward'].mean():.1f}")
print(
    f"Reward shaping impact: {(df['reward'] - df['original_reward']).mean():.1f}")

# Check if ANY lines were cleared
total_lines = df['lines_cleared'].sum()
episodes_with_lines = (df['lines_cleared'] > 0).sum()
print(f"\nLine clearing:")
print(f"Total lines: {total_lines}")
print(f"Episodes with lines: {episodes_with_lines}/{len(df)}")

# Check recent performance
recent = df.tail(100)
print(f"\nLast 100 episodes:")
print(f"Avg steps: {recent['steps'].mean():.1f}")
print(f"Max steps: {recent['steps'].max()}")
