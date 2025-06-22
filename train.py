#!/usr/bin/env python3
"""
train_complete_vision.py

Updated version of your existing train.py to use complete vision
Just replace your train.py with this content and run: python train.py
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
from tqdm import tqdm
import torch

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


def parse_args():
    """Parse training arguments"""
    parser = argparse.ArgumentParser(description='Train Tetris AI with Complete Vision')

    parser.add_argument('--episodes', type=int, default=500,
                        help='Number of episodes to train (default: 500)')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate (default: 5e-4, higher for richer info)')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help=f'Batch size (default: {BATCH_SIZE})')
    parser.add_argument('--gamma', type=float, default=GAMMA,
                        help=f'Discount factor (default: {GAMMA})')
    parser.add_argument('--model_type', type=str, default='dqn', choices=['dqn', 'dueling_dqn'],
                        help='Model architecture type')
    
    # Complete vision specific options
    parser.add_argument('--use_complete_vision', action='store_true', default=True,
                        help='Use complete 4-channel vision (recommended)')
    parser.add_argument('--use_cnn', action='store_true', default=True,
                        help='Use CNN processing for spatial relationships')
    
    # Training control
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from latest checkpoint')
    parser.add_argument('--save_freq', type=int, default=25,
                        help='Save model every N episodes')
    parser.add_argument('--log_freq', type=int, default=5,
                        help='Log progress every N episodes')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name for this experiment')

    return parser.parse_args()


def complete_vision_reward_shaping(obs, action, base_reward, done, info):
    """Enhanced reward shaping using complete 4-channel observation"""
    shaped_reward = base_reward
    
    # Extract information from 4-channel observation if available
    if len(obs.shape) == 3 and obs.shape[2] >= 2:
        board_channel = obs[:, :, 0]
        active_channel = obs[:, :, 1]
        
        # Active piece awareness bonuses
        active_piece_pixels = np.sum(active_channel > 0.01)
        if active_piece_pixels > 0:
            # Small bonus for visible active piece
            shaped_reward += 0.1
            
            # Height-based placement incentive
            active_rows = np.any(active_channel > 0.01, axis=1)
            if np.any(active_rows):
                lowest_piece_row = np.max(np.where(active_rows)[0])
                height_bonus = (24 - lowest_piece_row) * 0.02
                shaped_reward += height_bonus
    
    # MASSIVE line clear bonuses (agent can now achieve these!)
    lines_cleared = info.get('lines_cleared', 0)
    if lines_cleared > 0:
        # Exponential rewards for multiple lines
        line_bonus = lines_cleared * 50 * (lines_cleared ** 1.2)
        shaped_reward += line_bonus
        
        # Special Tetris celebration
        if lines_cleared == 4:
            shaped_reward += 200  # MASSIVE Tetris bonus
            print(f"🎉 TETRIS! 4 lines cleared, bonus: +200")
    
    # Balanced survival vs action incentive
    if not done:
        shaped_reward += 0.05
    else:
        shaped_reward -= 15  # Death penalty
    
    return shaped_reward


def train_complete_vision(args):
    """Main training function with complete vision"""
    start_time = time.time()

    print("🎯 TETRIS AI TRAINING WITH COMPLETE VISION")
    print("="*80)
    print("Using 4-channel observation: Board + Active Piece + Holder + Queue")
    print("Expected breakthrough: 20-100 episodes")
    print("="*80)

    # Create environment with complete vision
    env = make_env(
        use_cnn=args.use_cnn, 
        include_piece_info=args.use_complete_vision,
        frame_stack=1
    )
    
    print(f"✅ Environment created with complete vision")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Information channels: {env.observation_space.shape[2] if len(env.observation_space.shape) == 3 else 1}")

    # Initialize agent with enhanced parameters for complete vision
    agent = Agent(
        obs_space=env.observation_space,
        action_space=env.action_space,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        model_type=args.model_type,
        epsilon_start=0.8,    # Higher exploration for discovery
        epsilon_end=0.05,     # Maintain exploration longer  
        epsilon_decay=0.999,  # Much slower decay
        reward_shaping="none", # We'll apply our own
        max_episodes=args.episodes
    )

    # Resume if requested
    start_episode = 0
    if args.resume:
        print(f"\n🔄 Attempting to load existing checkpoint...")
        if agent.load_checkpoint(latest=True, model_dir=MODEL_DIR):
            start_episode = agent.episodes_done
            print(f"✅ Loaded checkpoint from episode {start_episode}")
            print(f"   Will adapt existing skills to complete vision")
            # Reset epsilon for exploration with new information
            agent.epsilon = 0.6
            print(f"   Reset epsilon to {agent.epsilon} for piece exploration")
        else:
            print("❌ No checkpoint found - starting fresh")

    # Initialize logger
    experiment_name = args.experiment_name or f"complete_vision_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = TrainingLogger(LOG_DIR, experiment_name)

    # Training metrics
    episode_rewards = []
    episode_lines = []
    lines_cleared_total = 0
    first_line_episode = None
    breakthrough_threshold = 20

    print(f"\n🚀 STARTING TRAINING")
    print(f"Episodes: {start_episode + 1} to {args.episodes}")
    print(f"Breakthrough target: {breakthrough_threshold} total lines")
    print(f"Current epsilon: {agent.epsilon:.4f}")
    print("-" * 80)

    # Training loop
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

            original_reward += reward

            # Apply complete vision reward shaping
            shaped_reward = complete_vision_reward_shaping(obs, action, reward, done, info)

            # Store experience
            agent.remember(obs, action, shaped_reward, next_obs, done, info, reward)

            # Learn frequently (more information to process)
            if episode_steps % 2 == 0:
                learning_metrics = agent.learn()

            # Track metrics
            episode_reward += shaped_reward
            episode_steps += 1
            lines_cleared = info.get('lines_cleared', 0)
            if lines_cleared > 0:
                lines_this_episode += lines_cleared
                if first_line_episode is None:
                    first_line_episode = episode + 1
                    print(f"\n🎉 FIRST LINE CLEARED! Episode {first_line_episode}")
                    print(f"   Complete vision breakthrough!")

            obs = next_obs

        # Episode end
        lines_cleared_total += lines_this_episode
        agent.end_episode(original_reward, episode_steps, lines_this_episode, original_reward)

        # Track data
        episode_rewards.append(episode_reward)
        episode_lines.append(lines_this_episode)

        # Log episode
        logger.log_episode(
            episode=episode + 1,
            reward=episode_reward,
            steps=episode_steps,
            epsilon=agent.epsilon,
            lines_cleared=lines_this_episode,
            original_reward=original_reward,
            total_lines=lines_cleared_total
        )

        # Progress reporting
        if (episode + 1) % args.log_freq == 0 or lines_this_episode > 0:
            recent_lines = sum(episode_lines[-10:]) if len(episode_lines) >= 10 else sum(episode_lines)
            recent_avg = recent_lines / min(10, len(episode_lines))
            
            print(f"Episode {episode+1:3d} | "
                  f"Lines: {lines_this_episode} (Total: {lines_cleared_total:2d}) | "
                  f"Reward: {episode_reward:7.1f} | "
                  f"Steps: {episode_steps:3d} | "
                  f"Avg(10): {recent_avg:.2f} | "
                  f"ε: {agent.epsilon:.4f}")

        # Check for breakthrough
        if lines_cleared_total >= breakthrough_threshold:
            print(f"\n🎉 BREAKTHROUGH ACHIEVED!")
            print(f"Total lines: {lines_cleared_total} in {episode + 1} episodes")
            print(f"Complete vision system SUCCESS!")
            break

        # Early victory detection  
        if episode >= 20:
            recent_avg = sum(episode_lines[-10:]) / min(10, len(episode_lines))
            if recent_avg >= 1.0:
                print(f"\n🏆 CONSISTENT LINE CLEARING! Avg: {recent_avg:.2f}")
                print(f"Complete vision breakthrough successful!")
                break

        # Save periodically
        if (episode + 1) % args.save_freq == 0:
            agent.save_checkpoint(episode + 1, MODEL_DIR)
            logger.save_logs()
            logger.plot_progress()

    # Final save
    agent.save_checkpoint(episode + 1, MODEL_DIR)
    logger.save_logs()
    logger.plot_progress()

    training_time = time.time() - start_time
    env.close()

    # Results summary
    episodes_trained = episode + 1 - start_episode
    avg_lines_per_episode = lines_cleared_total / episodes_trained if episodes_trained > 0 else 0
    
    print(f"\n" + "="*80)
    print(f"COMPLETE VISION TRAINING RESULTS")
    print(f"="*80)
    print(f"Episodes trained: {episodes_trained}")
    print(f"Total lines cleared: {lines_cleared_total}")
    print(f"Average lines per episode: {avg_lines_per_episode:.3f}")
    print(f"First line episode: {first_line_episode or 'None'}")
    print(f"Training time: {training_time/60:.1f} minutes")

    # Compare to 62k plateau
    original_avg = 0.03
    improvement = avg_lines_per_episode / original_avg if original_avg > 0 else float('inf')
    
    print(f"\nVS. ORIGINAL PLATEAU:")
    print(f"   Original (62k episodes): {original_avg} lines/episode")
    print(f"   Complete vision: {avg_lines_per_episode:.3f} lines/episode")
    print(f"   Improvement factor: {improvement:.1f}x")

    if lines_cleared_total >= breakthrough_threshold:
        print(f"\n✅ MISSION ACCOMPLISHED!")
        print(f"Complete vision broke the 62k episode plateau!")
        return True
    elif first_line_episode:
        print(f"\n⚠️  Breakthrough in progress - continue training!")
        return True
    else:
        print(f"\n🔧 May need more episodes or parameter adjustment")
        return False


def main():
    """Main entry point"""
    args = parse_args()
    
    print("🎯 Tetris AI Training with Complete Vision")
    print("Based on successful piece visibility test")
    print()
    
    success = train_complete_vision(args)
    
    if success:
        print("\n🎉 Training successful! Continue with advanced strategies.")
    else:
        print("\n🔧 Continue training or adjust parameters.")


if __name__ == "__main__":
    main()