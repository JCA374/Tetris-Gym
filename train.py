#!/usr/bin/env python3
"""
train.py - Updated for Complete Vision
Key changes: Proper reward shaping modes including positive_reward_shaping
"""

from config import make_env, ENV_NAME, LR, GAMMA, BATCH_SIZE, MAX_EPISODES, MODEL_DIR, LOG_DIR
from src.agent import Agent
from src.utils import TrainingLogger, print_system_info, make_dir
import os
import sys
import argparse
import time
import json
import numpy as np
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


def parse_args():
    """Parse training arguments"""
    parser = argparse.ArgumentParser(
        description='Train Tetris AI with Complete Vision')

    parser.add_argument('--episodes', type=int, default=500,
                        help='Number of episodes to train (default: 500)')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate (default: 5e-4)')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help=f'Batch size (default: {BATCH_SIZE})')
    parser.add_argument('--gamma', type=float, default=GAMMA,
                        help=f'Discount factor (default: {GAMMA})')
    parser.add_argument('--model_type', type=str, default='dqn', choices=['dqn', 'dueling_dqn'],
                        help='Model architecture type')

    # Complete vision options
    parser.add_argument('--use_complete_vision', action='store_true', default=True,
                        help='Use complete 4-channel vision (REQUIRED for success)')
    parser.add_argument('--use_cnn', action='store_true', default=True,
                        help='Use CNN processing')

    # Epsilon settings for fresh training
    parser.add_argument('--epsilon_start', type=float, default=0.8,
                        help='Starting epsilon for exploration (default: 0.8)')
    parser.add_argument('--epsilon_end', type=float, default=0.05,
                        help='Final epsilon (default: 0.05)')
    parser.add_argument('--epsilon_decay', type=float, default=0.999,
                        help='Epsilon decay rate (default: 0.999)')

    # Training control
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from latest checkpoint')
    parser.add_argument('--save_freq', type=int, default=25,
                        help='Save model every N episodes')
    parser.add_argument('--log_freq', type=int, default=5,
                        help='Log progress every N episodes')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name for this experiment')

    # Reward shaping mode
    parser.add_argument('--reward_shaping', type=str, default='complete',
                        choices=['none', 'complete', 'positive'],
                        help='Type of reward shaping: none (raw), complete (complete vision shaping), positive (positive reinforcement)')
    return parser.parse_args()


# Replace any negative-heavy shaping with this:
def complete_vision_reward_shaping(obs, action, base_reward, done, info):
    shaped_reward = base_reward

    # ALWAYS positive for survival
    if not done:
        shaped_reward += 2.0  # Strong survival incentive

    # HUGE line bonuses
    lines = info.get('lines_cleared', 0)
    if lines > 0:
        shaped_reward += lines * 200  # Make lines incredibly valuable

    # Small death penalty
    if done:
        shaped_reward -= 5  # Minimal penalty

    return shaped_reward


# Look in your train.py - is it still using negative rewards?
# You need something like this:

def positive_reward_shaping(obs, action, base_reward, done, info):
    shaped_reward = base_reward

    # POSITIVE survival bonus (not negative!)
    if not done:
        shaped_reward += 1.0  # +1 per step

    # HUGE line bonuses
    lines = info.get('lines_cleared', 0)
    if lines > 0:
        shaped_reward += lines * 100

    # SMALL death penalty
    if done:
        shaped_reward -= 10  # Only -10, not -50 or -100

    return shaped_reward


def train(args):
    """Main training function with reward shaping modes"""
    start_time = time.time()
    print("🎯 TETRIS AI TRAINING WITH COMPLETE VISION")
    print("="*80)

    if not args.use_complete_vision:
        print("⚠️  WARNING: Complete vision disabled! This will likely fail!")
        print("   Add --use_complete_vision flag")

    env = make_env(
        use_complete_vision=args.use_complete_vision,
        use_cnn=args.use_cnn
    )
    print(f"✅ Environment created")
    print(f"   Observation space: {env.observation_space}")
    if len(env.observation_space.shape) == 3:
        channels = env.observation_space.shape[-1]
        if channels == 4:
            print(f"   ✅ 4-channel complete vision confirmed!")
        else:
            print(
                f"   ⚠️  Only {channels} channels - may be missing information!")

    agent = Agent(
        obs_space=env.observation_space,
        action_space=env.action_space,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        model_type=args.model_type,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        reward_shaping="none",
        max_episodes=args.episodes
    )

    start_episode = 0
    if args.resume:
        print(f"\n🔄 Loading checkpoint...")
        if agent.load_checkpoint(latest=True, model_dir=MODEL_DIR):
            start_episode = agent.episodes_done
            print(f"✅ Resumed from episode {start_episode}")
            if agent.epsilon < 0.3:
                print(
                    f"⚠️  Epsilon too low ({agent.epsilon:.3f}), boosting to 0.5")
                agent.epsilon = 0.5
        else:
            print("❌ No checkpoint found - starting fresh")

    experiment_name = args.experiment_name or f"complete_vision_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = TrainingLogger(LOG_DIR, experiment_name)

    # Determine shaping function
    if args.reward_shaping == 'complete':
        shaper_fn = complete_vision_reward_shaping
    elif args.reward_shaping == 'positive':
        shaper_fn = positive_reward_shaping
    else:
        shaper_fn = None

    lines_cleared_total = 0
    first_line_episode = None
    recent_rewards = []
    recent_lines = []

    print(f"\n🚀 Starting training")
    print(f"Episodes: {start_episode + 1} to {args.episodes}")
    print(f"Epsilon: {agent.epsilon:.3f}")
    print("-" * 80)

    for episode in range(start_episode, args.episodes):
        obs, info = env.reset()
        episode_reward = 0
        original_reward = 0
        episode_steps = 0
        lines_this_episode = 0
        done = False

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            raw_reward = reward
            if shaper_fn:
                shaped_reward = shaper_fn(obs, action, raw_reward, done, info)
            else:
                shaped_reward = raw_reward

            original_reward += raw_reward

            agent.remember(obs, action, shaped_reward,
                           next_obs, done, info, raw_reward)

            if episode_steps % 4 == 0 and len(agent.memory) >= agent.batch_size:
                agent.learn()

            episode_reward += shaped_reward
            episode_steps += 1
            lines = info.get('lines_cleared', 0)
            if lines > 0:
                lines_this_episode += lines
                lines_cleared_total += lines
                if first_line_episode is None:
                    first_line_episode = episode + 1
                    print(
                        f"\n🎉 FIRST LINE CLEARED! Episode {first_line_episode}")
            obs = next_obs

        agent.end_episode(episode_reward, episode_steps,
                          lines_this_episode, original_reward)

        recent_rewards.append(episode_reward)
        recent_lines.append(lines_this_episode)
        if len(recent_rewards) > 100:
            recent_rewards.pop(0)
            recent_lines.pop(0)

        logger.log_episode(
            episode=episode + 1,
            reward=episode_reward,
            steps=episode_steps,
            epsilon=agent.epsilon,
            lines_cleared=lines_this_episode,
            original_reward=original_reward,
            total_lines=lines_cleared_total
        )

        if (episode + 1) % args.log_freq == 0 or lines_this_episode > 0:
            avg_reward = np.mean(recent_rewards)
            avg_lines = np.mean(recent_lines)
            print(f"Episode {episode+1:4d} | "
                  f"Lines: {lines_this_episode} (Total: {lines_cleared_total:3d}) | "
                  f"Reward: {episode_reward:7.1f} (Avg: {avg_reward:6.1f}) | "
                  f"Steps: {episode_steps:3d} | "
                  f"Lines/Ep: {avg_lines:.2f} | "
                  f"ε: {agent.epsilon:.3f}")

        if (episode + 1) % args.save_freq == 0:
            agent.save_checkpoint(episode + 1, MODEL_DIR)
            logger.save_logs()
            logger.plot_progress()

    training_time = time.time() - start_time
    env.close()
    agent.save_checkpoint(args.episodes, MODEL_DIR)
    logger.save_logs()
    logger.plot_progress()

    print(f"\n" + "="*80)
    print(f"TRAINING COMPLETE")
    print("="*80)
    print(f"Total episodes: {args.episodes - start_episode}")
    avg_lines_all = lines_cleared_total / \
        (args.episodes - start_episode) if (args.episodes - start_episode) > 0 else 0
    print(f"Total lines cleared: {lines_cleared_total}")
    print(f"Average lines per episode: {avg_lines_all:.3f}")
    print(f"First line at episode: {first_line_episode or 'Never'}")
    print(f"Training time: {training_time/60:.1f} minutes")

    if lines_cleared_total == 0:
        print("\n⚠️  No lines cleared! Check:")
        print("  1. Are you using --use_complete_vision?")
        print("  2. Did you start fresh (not resume)?")
        print("  3. Is epsilon high enough for exploration?")
    elif avg_lines_all < 0.1:
        print("\n⚠️  Low performance. Try:")
        print("  1. Increase line clear rewards")
        print("  2. Start completely fresh")
        print("  3. Use emergency_breakthrough_complete.py")
    else:
        print("\n✅ Training successful!")


def main():
    """Main entry point"""
    args = parse_args()

    print("🎯 Tetris AI Training with Complete Vision")
    print("Key: The agent can now SEE the pieces it's placing!")
    print()

    train(args)


if __name__ == "__main__":
    main()
