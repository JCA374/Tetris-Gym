#!/usr/bin/env python3
"""
Enhanced training script specifically designed to break the 0-lines plateau
"""

import random
import numpy as np
from src.agent import Agent
from config import make_env
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


def break_plateau_training():
    """Training approach to force line discovery"""

    print("=== PLATEAU BREAKER TRAINING ===")
    print("Goal: Force agent to discover line clearing")
    print("="*50)

    # Strategy 1: Action Masking - Reduce action space temporarily
    # Only allow: down, left, right, hard drop (remove rotations initially)
    ACTION_MASK = [1, 2, 3, 6]  # Indices of allowed actions

    # Strategy 2: Dense Reward Episodes
    # Every Nth episode, give massive bonus for ANY downward progress
    DENSE_REWARD_FREQUENCY = 5

    # Create environment
    env = make_env(frame_stack=4)

    # Load existing agent
    agent = Agent(
        obs_space=env.observation_space,
        action_space=env.action_space,
        reward_shaping="simple"
    )

    # Try to load checkpoint
    if agent.load_checkpoint(latest=True):
        print("Loaded existing agent")
    else:
        print("Starting fresh agent")

    # Force higher exploration initially
    agent.epsilon = 0.5  # Override low epsilon
    agent.epsilon_decay = 0.999  # Slower decay

    print(f"Starting with epsilon: {agent.epsilon}")

    # Training loop with special strategies
    total_lines_cleared = 0
    episodes_since_line_clear = 0

    for episode in range(500):  # Shorter focused training
        obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        lines_cleared = 0

        # Strategy flags for this episode
        use_action_mask = (episode < 100)  # First 100 episodes
        use_dense_rewards = (episode % DENSE_REWARD_FREQUENCY == 0)
        use_forced_exploration = (episodes_since_line_clear > 50)

        while not done:
            # Select action with strategies
            if use_forced_exploration and random.random() < 0.3:
                # Force random action to break patterns
                action = random.choice(
                    ACTION_MASK) if use_action_mask else env.action_space.sample()
            else:
                action = agent.select_action(obs)
                # Apply action mask if enabled
                if use_action_mask and action not in ACTION_MASK:
                    action = random.choice(ACTION_MASK)

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Dense rewards for this episode
            if use_dense_rewards:
                # Reward ANY piece placement
                if 'piece_placed' in info or episode_steps % 10 == 0:
                    reward += 5  # Bonus for placing pieces

                # Big bonus for filling bottom rows
                if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'board'):
                    board = env.unwrapped.board
                    if board is not None:
                        bottom_row_filled = np.sum(
                            board[-1, :]) / board.shape[1]
                        if bottom_row_filled > 0.8:  # 80% filled
                            reward += 10

            # Check for line clears
            if info.get('lines_cleared', 0) > 0:
                lines_cleared += info['lines_cleared']
                total_lines_cleared += info['lines_cleared']
                episodes_since_line_clear = 0

                # MASSIVE celebration for first line clear
                if total_lines_cleared == 1:
                    print(
                        f"🎉 FIRST LINE CLEARED! Episode {episode}, Step {episode_steps}")
                    reward += 100  # Huge bonus
                else:
                    # Big bonus for any line
                    reward += 50 * info['lines_cleared']

            # Store experience
            agent.remember(obs, action, reward, next_obs, done, info)

            # Learn more frequently when using special strategies
            if episode_steps % 2 == 0 and len(agent.memory) >= agent.batch_size:
                agent.learn()

            episode_reward += reward
            episode_steps += 1
            obs = next_obs

        # Episode complete
        episodes_since_line_clear += 1
        agent.end_episode(episode_reward, episode_steps, lines_cleared)

        # Progress report
        if episode % 10 == 0:
            stats = agent.get_stats()
            print(f"Episode {episode:3d} | "
                  f"Reward: {episode_reward:6.1f} | "
                  f"Steps: {episode_steps:3d} | "
                  f"Lines: {lines_cleared} (Total: {total_lines_cleared}) | "
                  f"Eps: {agent.epsilon:.3f}")

            if use_action_mask:
                print("  [Using action mask]")
            if use_dense_rewards:
                print("  [Dense rewards active]")
            if use_forced_exploration:
                print("  [Forced exploration - stuck for 50+ episodes]")

        # Success check
        if total_lines_cleared >= 10:
            print(f"\n✅ SUCCESS! Agent has learned to clear lines!")
            print(f"Total lines cleared: {total_lines_cleared}")
            print(f"Saving breakthrough model...")
            agent.save_checkpoint(episode, model_dir="models/")
            break

        # Adjust strategies based on progress
        if episode == 100 and total_lines_cleared == 0:
            print("\n⚠️ Still no lines after 100 episodes. Increasing intervention...")
            agent.epsilon = 0.7  # Boost exploration
            DENSE_REWARD_FREQUENCY = 3  # More frequent bonuses

    # Final save
    env.close()
    print(f"\nTraining complete. Total lines cleared: {total_lines_cleared}")

    if total_lines_cleared == 0:
        print("\n❌ Failed to break plateau. Try:")
        print("1. Manual reward injection test")
        print("2. Environment with pre-filled rows")
        print("3. Imitation learning from scripted player")
    else:
        print("\n✅ Plateau broken! Continue normal training with:")
        print("python train.py --episodes 5000 --resume --reward_shaping simple")


if __name__ == "__main__":
    break_plateau_training()
