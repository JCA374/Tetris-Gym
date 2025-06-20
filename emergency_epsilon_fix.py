#!/usr/bin/env python3
"""
Emergency Epsilon Fix - Force exploration to break line clearing plateau
Run this for 200-300 episodes to force breakthrough
"""

import numpy as np
import random
from src.agent import Agent
from config import make_env
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


# Set seeds
random.seed(42)
np.random.seed(42)

print("🚨 EMERGENCY EPSILON TRAINING")
print("Designed to force line clearing discovery")

# Create environment
env = make_env(frame_stack=4)

# Create agent
agent = Agent(
    obs_space=env.observation_space,
    action_space=env.action_space,
    reward_shaping="simple"
)

# Load checkpoint
if agent.load_checkpoint(latest=True):
    start_episode = agent.episodes_done
    print(f"✅ Loaded checkpoint from episode {start_episode}")
else:
    start_episode = 0
    print("❌ No checkpoint found - starting fresh")

# 🔥 FORCE EPSILON BOOST
original_epsilon = agent.epsilon
agent.epsilon = 0.5              # MASSIVE exploration boost
agent.epsilon_end = 0.1          # Don't drop below 10%
agent.epsilon_decay = 0.999      # MUCH slower decay

print(f"Epsilon: {original_epsilon:.4f} → {agent.epsilon:.4f}")
print("🎯 Goal: Discover line clearing within 100-200 episodes")

# 🔥 BOOST REWARD SHAPING
line_clear_bonus = 50    # Massive bonus for ANY line clear
tetris_bonus = 200       # Huge bonus for 4-line Tetris
game_over_penalty = -100  # Harsh death penalty

print(
    f"Reward shaping: +{line_clear_bonus} per line, +{tetris_bonus} for Tetris")

# Training parameters
target_episodes = start_episode + 300  # Train 300 more episodes
lines_cleared_total = 0
breakthrough_threshold = 20  # Stop when we get 20+ total lines

print(f"Training episodes {start_episode+1} to {target_episodes}")
print("=" * 60)

for episode in range(start_episode, target_episodes):
    obs, info = env.reset()
    episode_reward = 0
    episode_steps = 0
    lines_this_episode = 0
    done = False

    while not done:
        # 🔥 FORCED EXPLORATION for first 150 episodes
        if (episode - start_episode) < 150 and random.random() < 0.4:
            # Force useful actions only (no NO-OP)
            # Right, Left, Down, Rotate, Rotate, Hard Drop
            action = random.choice([1, 2, 3, 4, 5, 6])
        else:
            action = agent.select_action(obs)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # 🔥 APPLY AGGRESSIVE REWARD SHAPING
        lines_cleared = info.get('lines_cleared', 0)
        if lines_cleared > 0:
            lines_this_episode += lines_cleared
            reward += lines_cleared * line_clear_bonus

            if lines_cleared == 4:  # Tetris!
                reward += tetris_bonus
                print(f"🎉 TETRIS! Episode {episode+1}, 4 lines cleared!")

        if done:
            reward += game_over_penalty  # Death penalty

        # Store and learn
        agent.remember(obs, action, reward, next_obs, done, info)
        if episode_steps % 2 == 0 and len(agent.memory) >= agent.batch_size:
            agent.learn()

        episode_reward += reward
        episode_steps += 1
        obs = next_obs

    lines_cleared_total += lines_this_episode
    agent.end_episode(episode_reward, episode_steps, lines_this_episode)

    # Progress reporting
    if (episode + 1) % 5 == 0:
        print(f"Episode {episode+1:4d} | "
              f"Reward: {episode_reward:7.1f} | "
              f"Lines: {lines_this_episode} (Total: {lines_cleared_total:2d}) | "
              f"Steps: {episode_steps:3d} | "
              f"Eps: {agent.epsilon:.3f}")

        # Check for breakthrough
        if lines_this_episode > 0:
            print(f"         ✅ LINE CLEAR DISCOVERED! Episode {episode+1}")

        if lines_cleared_total >= breakthrough_threshold:
            print(f"\n🎉 BREAKTHROUGH! {lines_cleared_total} lines cleared!")
            print(f"Agent has learned line clearing - saving model...")
            agent.save_checkpoint(episode + 1)
            break

    # Save periodically
    if (episode + 1) % 50 == 0:
        agent.save_checkpoint(episode + 1)

env.close()

print(f"\n" + "=" * 60)
print(f"EMERGENCY TRAINING COMPLETE")
print(f"=" * 60)
print(f"Episodes trained: {episode - start_episode + 1}")
print(f"Total lines cleared: {lines_cleared_total}")
print(f"Final epsilon: {agent.epsilon:.4f}")

if lines_cleared_total >= breakthrough_threshold:
    print(f"\n✅ SUCCESS! Agent learned line clearing")
    print(f"Continue with normal training:")
    print(
        f"python train.py --episodes {episode + 1500} --resume --reward_shaping simple")
else:
    print(f"\n⚠️  Partial progress. Lines cleared: {lines_cleared_total}")
    if lines_cleared_total > 0:
        print(f"Some progress made - continue with more aggressive settings")
        print(f"python emergency_epsilon_fix.py  # Run again")
    else:
        print(f"No progress - try plateau breaker:")
        print(f"python break_plateau_train.py --episodes 500 --epsilon-boost 0.8")
