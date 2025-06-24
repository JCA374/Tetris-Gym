#!/usr/bin/env python3
"""
emergency_positive_rewards.py
Fix negative reward spiral with guaranteed positive reinforcement
"""

import numpy as np
import sys
import os
from config import make_env
from src.agent import Agent


def positive_only_reward_shaping(obs, action, base_reward, done, info):
    """
    Guaranteed positive rewards to break negative spiral
    """
    # Start with base reward
    shaped_reward = base_reward

    # 1. STRONG survival bonus
    if not done:
        shaped_reward += 2.0  # +2 per step (strong incentive to stay alive)

    # 2. MASSIVE line clear bonuses
    lines = info.get('lines_cleared', 0)
    if lines > 0:
        line_bonuses = {
            1: 100,   # Single
            2: 300,   # Double
            3: 600,   # Triple
            4: 2000   # Tetris!
        }
        bonus = line_bonuses.get(lines, lines * 150)
        shaped_reward += bonus
        print(f"🎉 {lines} LINES CLEARED! +{bonus} bonus!")

    # 3. Height bonus (reward low placement, not punish high)
    if len(obs.shape) == 3 and obs.shape[2] >= 1:
        board_channel = obs[:, :, 0]
        filled_rows = np.any(board_channel > 0.01, axis=1)
        if np.any(filled_rows):
            first_filled = np.argmax(filled_rows)
            empty_rows = first_filled
            # Bonus for keeping board low
            if empty_rows > 10:
                shaped_reward += (empty_rows - 10) * 0.5

    # 4. Very small death penalty
    if done:
        shaped_reward -= 5  # Tiny penalty
        # But give credit for survival time
        steps = info.get('step_count', 0)
        if steps > 20:
            shaped_reward += steps * 0.2

    # 5. Guarantee minimum positive reward
    if shaped_reward < 0.1 and not done:
        shaped_reward = 0.1  # Always at least slightly positive

    return shaped_reward


def emergency_positive_training():
    """Emergency training with guaranteed positive rewards"""

    print("🚨 EMERGENCY POSITIVE REWARDS TRAINING")
    print("="*60)
    print("Breaking negative reward spiral...")
    print()

    # Create environment
    env = make_env(use_complete_vision=True, use_cnn=True)

    # Create FRESH agent (don't load negative training)
    agent = Agent(
        obs_space=env.observation_space,
        action_space=env.action_space,
        lr=5e-4,
        epsilon_start=0.5,  # Medium exploration
        epsilon_end=0.05,
        epsilon_decay=0.998,
        reward_shaping="none",
        max_episodes=3000
    )

    print("🆕 Starting fresh with positive-only rewards")
    print("Target: Consistent 100+ step episodes")
    print()

    # Training loop
    for episode in range(200):
        obs, info = env.reset()
        episode_reward = 0
        episode_steps = 0
        lines_cleared = 0

        done = False
        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Apply positive-only shaping
            shaped_reward = positive_only_reward_shaping(
                obs, action, reward, done, info)

            # Store experience
            agent.remember(obs, action, shaped_reward, next_obs, done, info)

            # Learn frequently
            if episode_steps % 2 == 0 and len(agent.memory) >= agent.batch_size:
                agent.learn()

            episode_reward += shaped_reward
            episode_steps += 1
            lines_cleared += info.get('lines_cleared', 0)

            obs = next_obs

        # Episode complete
        agent.end_episode(episode_reward, episode_steps, lines_cleared)

        # Progress report
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1:3d} | "
                  f"Reward: {episode_reward:6.1f} | "
                  f"Steps: {episode_steps:3d} | "
                  f"Lines: {lines_cleared} | "
                  f"ε: {agent.epsilon:.3f}")

    # Save positive-trained model
    agent.save_checkpoint(200)
    env.close()

    print("\n✅ Positive reward training complete!")
    print("Continue with: python train.py --episodes 3000 --resume")


if __name__ == "__main__":
    emergency_positive_training()
