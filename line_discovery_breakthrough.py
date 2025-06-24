#!/usr/bin/env python3
"""
line_discovery_breakthrough.py
Aggressive techniques to force line clearing discovery
"""

import numpy as np
import random
from config import make_env
from src.agent import Agent
import torch


def aggressive_line_reward_shaping(obs, action, base_reward, done, info):
    """
    Extremely aggressive reward shaping to discover lines
    """
    shaped_reward = base_reward

    # MINIMAL survival bonus (to not overshadow lines)
    if not done:
        shaped_reward += 0.1  # Very small

    # ASTRONOMICAL line clear rewards
    lines = info.get('lines_cleared', 0)
    if lines > 0:
        line_bonuses = {
            1: 500,    # Single - HUGE
            2: 1500,   # Double - MASSIVE
            3: 3000,   # Triple - ENORMOUS
            4: 10000   # Tetris - JACKPOT
        }
        bonus = line_bonuses.get(lines, lines * 1000)
        shaped_reward += bonus
        print(f"🎉🎉🎉 {lines} LINES! +{bonus} MEGA BONUS! 🎉🎉🎉")

    # Bonus for filling bottom rows (helps discover lines)
    if len(obs.shape) == 3 and obs.shape[2] >= 1:
        board_channel = obs[:, :, 0]
        # Check bottom 3 rows
        bottom_rows = board_channel[-3:, :]
        fill_ratio = np.sum(bottom_rows > 0.01) / bottom_rows.size

        if fill_ratio > 0.7:  # 70% filled
            shaped_reward += 10 * fill_ratio  # Bonus for almost complete rows

    # Tiny death penalty
    if done:
        shaped_reward -= 1

    return shaped_reward


def action_masking_for_lines(obs, agent, force_useful=True):
    """
    Mask actions to encourage line-clearing behavior
    """
    # Get Q-values for all actions
    with torch.no_grad():
        state_tensor = agent._preprocess_state(obs)
        q_values = agent.q_network(state_tensor)
        q_values_np = q_values.cpu().numpy().flatten()

    if force_useful and random.random() < 0.3:  # 30% of the time
        # Force useful actions that might lead to line clears
        useful_actions = [1, 2, 3, 6]  # right, left, down, hard_drop
        # Bias towards hard drop to complete lines faster
        if random.random() < 0.5:
            return 6  # hard_drop
        else:
            return random.choice(useful_actions)
    else:
        # Normal Q-value based selection
        return q_values_np.argmax()


def curriculum_line_discovery(env, agent):
    """
    Curriculum approach: start with partially filled board
    """
    obs, info = env.reset()

    # 50% chance to pre-fill bottom rows
    if random.random() < 0.5:
        # This is a hack - we're trying to encourage line clearing
        # by starting with almost-complete rows
        print("📚 Curriculum: Starting with partially filled rows")

        # Take random actions to fill bottom
        for _ in range(30):
            action = random.choice([1, 2, 3, 6])  # Useful actions only
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()
                break

    return obs, info


def line_discovery_training():
    """
    Aggressive training specifically to discover line clearing
    """
    print("🚀 LINE DISCOVERY BREAKTHROUGH TRAINING")
    print("="*60)
    print("Using extreme measures to force line discovery...")
    print()

    # Create environment
    env = make_env(use_complete_vision=True, use_cnn=True)

    # Load the agent that's already learned to survive
    agent = Agent(
        obs_space=env.observation_space,
        action_space=env.action_space,
        lr=1e-3,  # Higher learning rate for faster discovery
        epsilon_start=0.8,  # High exploration
        epsilon_end=0.1,
        epsilon_decay=0.995,
        reward_shaping="none",
        max_episodes=1000
    )

    # Try to load existing progress
    if agent.load_checkpoint(latest=True):
        print("✅ Loaded existing checkpoint")
        # Boost epsilon for more exploration
        agent.epsilon = max(0.5, agent.epsilon)
        print(f"   Epsilon boosted to: {agent.epsilon:.3f}")
    else:
        print("🆕 Starting fresh")

    lines_discovered = False
    total_lines = 0
    discovery_episode = None

    print("\n🎯 Goal: Clear at least 1 line!")
    print("Using:")
    print("  • Astronomical line rewards (500-10000)")
    print("  • Minimal survival bonus (0.1)")
    print("  • Action masking to useful moves")
    print("  • Curriculum with pre-filled boards")
    print()

    for episode in range(300):  # 300 episodes to discover lines
        # Use curriculum approach
        if not lines_discovered:
            obs, info = curriculum_line_discovery(env, agent)
        else:
            obs, info = env.reset()

        episode_reward = 0
        episode_steps = 0
        lines_this_episode = 0
        done = False

        while not done:
            # Use action masking for first 100 episodes
            if episode < 100 and not lines_discovered:
                action = action_masking_for_lines(
                    obs, agent, force_useful=True)
            else:
                action = agent.select_action(obs)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Apply aggressive shaping
            shaped_reward = aggressive_line_reward_shaping(
                obs, action, reward, done, info
            )

            # Store experience
            agent.remember(obs, action, shaped_reward, next_obs, done, info)

            # Learn frequently
            if len(agent.memory) >= agent.batch_size:
                agent.learn()

            episode_reward += shaped_reward
            episode_steps += 1

            # Check for lines
            lines = info.get('lines_cleared', 0)
            if lines > 0:
                lines_this_episode += lines
                total_lines += lines
                if not lines_discovered:
                    lines_discovered = True
                    discovery_episode = episode + 1
                    print(
                        f"\n🎊🎊🎊 BREAKTHROUGH! First line cleared at episode {discovery_episode}! 🎊🎊🎊\n")

            obs = next_obs

        # Episode complete
        agent.end_episode(episode_reward, episode_steps, lines_this_episode)

        # Progress report
        if (episode + 1) % 10 == 0 or lines_this_episode > 0:
            print(f"Episode {episode+1:3d} | "
                  f"Reward: {episode_reward:8.1f} | "
                  f"Steps: {episode_steps:3d} | "
                  f"Lines: {lines_this_episode} (Total: {total_lines}) | "
                  f"ε: {agent.epsilon:.3f}")

        # Stop if we've cleared 10+ lines total
        if total_lines >= 10:
            print(f"\n✅ SUCCESS! Cleared {total_lines} lines!")
            print("Line clearing behavior discovered!")
            break

    # Save the breakthrough model
    agent.save_checkpoint(episode + 1)
    env.close()

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Total lines cleared: {total_lines}")
    print(f"Discovery episode: {discovery_episode or 'Not discovered'}")

    if total_lines > 0:
        print("\n🎉 BREAKTHROUGH ACHIEVED!")
        print("Continue normal training with:")
        print("python train.py --episodes 5000 --resume")
    else:
        print("\n⚠️  Still no lines cleared.")
        print("Try the nuclear option:")
        print("python nuclear_line_forcing.py")


if __name__ == "__main__":
    line_discovery_training()
