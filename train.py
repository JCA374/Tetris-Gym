#!/usr/bin/env python3
"""
train.py - FIXED VERSION with reward shaping always enabled
Key fix: Reward shaping is now enabled by default and cannot be accidentally disabled
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

from src.reward_shaping import (
    aggressive_reward_shaping,
    positive_reward_shaping,
    balanced_reward_shaping
)

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

    # Epsilon settings for fresh training (FIXED for long training)
    parser.add_argument('--epsilon_start', type=float, default=1.0,
                    help='Starting epsilon for exploration (default: 1.0)')
    parser.add_argument('--epsilon_end', type=float, default=0.01,
                    help='Final epsilon (default: 0.01)')
    parser.add_argument('--epsilon_decay', type=float, default=0.9999,
                    help='Epsilon decay rate (default: 0.9999 for long training)')
    
    # Training control
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from latest checkpoint')
    parser.add_argument('--force_fresh', action='store_true',
                        help='Force fresh start even if checkpoint exists')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='Save model every N episodes')
    parser.add_argument('--log_freq', type=int, default=10,
                        help='Log progress every N episodes')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name for this experiment')

    # Reward shaping mode - NOW DEFAULTS TO 'positive'
    parser.add_argument('--reward_shaping', type=str, default='positive',
                        choices=['positive', 'aggressive', 'balanced'],
                        help='Type of reward shaping (default: positive)')
    
    return parser.parse_args()


def train(args):
    """Main training function with reward shaping ALWAYS enabled"""
    start_time = time.time()
    print("🎯 TETRIS AI TRAINING - FIXED VERSION")
    print("="*80)
    print("✅ Reward shaping: ENABLED by default")
    print(f"✅ Shaping mode: {args.reward_shaping}")
    print("="*80)

    if not args.use_complete_vision:
        print("⚠️  WARNING: Complete vision disabled! This will likely fail!")
        print("   Add --use_complete_vision flag")

    # Create environment
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
            print(f"   ⚠️  Only {channels} channels - may be missing information!")

    # Create agent
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
        reward_shaping="balanced", 
        max_episodes=args.episodes
    )

    # Handle resume/fresh start
    start_episode = 0
    if args.force_fresh:
        print("🆕 Forcing fresh start (ignoring any checkpoints)")
    elif args.resume:
        print(f"\n🔄 Attempting to load checkpoint...")
        if agent.load_checkpoint(latest=True, model_dir=MODEL_DIR):
            start_episode = agent.episodes_done
            print(f"✅ Resumed from episode {start_episode}")
            
            # Force higher epsilon for exploration with new shaping
            if agent.epsilon < 0.3:
                print(f"⚠️  Epsilon too low ({agent.epsilon:.3f}), boosting to 0.5")
                agent.epsilon = 0.5
        else:
            print("❌ No checkpoint found - starting fresh")
    else:
        print("🆕 Starting fresh training")

    # Setup experiment logging
    experiment_name = args.experiment_name or f"fixed_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = TrainingLogger(LOG_DIR, experiment_name)

    # Select reward shaping function (ALWAYS enabled!)
    shaping_functions = {
        'positive': positive_reward_shaping,
        'aggressive': aggressive_reward_shaping,
        'balanced': balanced_reward_shaping
    }
    shaper_fn = shaping_functions[args.reward_shaping]
    
    print(f"✅ Reward shaping function: {args.reward_shaping}")
    print(f"   Survival bonus: +2.0 per step")
    print(f"   Line bonuses: 500-10000 depending on type")
    print(f"   Death penalty: -10")

    # Training metrics
    lines_cleared_total = 0
    first_line_episode = None
    recent_rewards = []
    recent_lines = []
    recent_steps = []

    print(f"\n🚀 Starting training")
    print(f"Episodes: {start_episode + 1} to {args.episodes}")
    print(f"Epsilon: {agent.epsilon:.3f}")
    print("-" * 80)

    # MAIN TRAINING LOOP
    for episode in range(start_episode, args.episodes):
        obs, info = env.reset()
        episode_reward = 0
        original_reward = 0
        episode_steps = 0
        lines_this_episode = 0
        done = False

        while not done:
            # Select action
            action = agent.select_action(obs)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store raw reward
            raw_reward = reward
            original_reward += raw_reward
            
            # ✅ ALWAYS APPLY REWARD SHAPING (this is the fix!)
            shaped_reward = shaper_fn(obs, action, raw_reward, done, info)

            # Store experience with shaped reward
            agent.remember(obs, action, shaped_reward, next_obs, done, info, raw_reward)

            # Learn every 4 steps
            if episode_steps % 4 == 0 and len(agent.memory) >= agent.batch_size:
                agent.learn()

            # Update metrics
            episode_reward += shaped_reward
            episode_steps += 1
            
            # Track line clears
            lines = info.get('lines_cleared', 0)
            if lines > 0:
                lines_this_episode += lines
                lines_cleared_total += lines
                if first_line_episode is None:
                    first_line_episode = episode + 1
                    print(f"\n🎉🎉🎉 FIRST LINE CLEARED! Episode {first_line_episode} 🎉🎉🎉\n")
            
            obs = next_obs

        # End of episode - log stats
        agent.end_episode(episode_reward, episode_steps, lines_this_episode, original_reward)
        
        # ✅ FIXED: Add board state monitoring
        if episode % args.log_freq == 0:
            # Analyze final board state from last observation
            from src.reward_shaping import get_column_heights, count_holes, calculate_bumpiness
            
            board = None
            if isinstance(next_obs, dict) and 'board' in next_obs:
                board = next_obs['board']
            elif isinstance(next_obs, np.ndarray):
                if len(next_obs.shape) == 3:
                    board = next_obs[:, :, 0] if next_obs.shape[-1] > 1 else next_obs[:, :, 0]
                elif len(next_obs.shape) == 2:
                    board = next_obs
            
            if board is not None:
                heights = get_column_heights(board)
                
                # Calculate max row fullness
                max_row_fullness = 0
                for row in range(board.shape[0]):
                    filled = np.sum(board[row, :] > 0)
                    max_row_fullness = max(max_row_fullness, filled)
                
                # Calculate statistics
                max_height = max(heights) if heights else 0
                height_variance = np.var(heights)
                holes = count_holes(board)
                bumpiness = calculate_bumpiness(board)
                
                print(f"  📊 Board Stats:")
                print(f"     Max row fullness: {max_row_fullness}/10 cells")
                print(f"     Column heights: {heights}")
                print(f"     Max height: {max_height}, Variance: {height_variance:.2f}")
                print(f"     Holes: {holes}, Bumpiness: {bumpiness:.1f}")





        # Track recent performance
        recent_rewards.append(episode_reward)
        recent_lines.append(lines_this_episode)
        recent_steps.append(episode_steps)
        if len(recent_rewards) > 100:
            recent_rewards.pop(0)
            recent_lines.pop(0)
            recent_steps.pop(0)

        # Log episode data
        logger.log_episode(
            episode=episode + 1,
            reward=episode_reward,
            steps=episode_steps,
            epsilon=agent.epsilon,
            lines_cleared=lines_this_episode,
            original_reward=original_reward,
            total_lines=lines_cleared_total,
            shaped_reward_used=True  # Always true now!
        )

        # Print progress
        if (episode + 1) % args.log_freq == 0 or lines_this_episode > 0:
            avg_reward = np.mean(recent_rewards)
            avg_lines = np.mean(recent_lines)
            avg_steps = np.mean(recent_steps)
            
            print(f"Episode {episode+1:4d} | "
                  f"Lines: {lines_this_episode} (Total: {lines_cleared_total:3d}) | "
                  f"Reward: {episode_reward:7.1f} (Avg: {avg_reward:6.1f}) | "
                  f"Steps: {episode_steps:3d} (Avg: {avg_steps:4.1f}) | "
                  f"Lines/Ep: {avg_lines:.2f} | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"Shaping: YES")

        # Save checkpoint periodically
        if (episode + 1) % args.save_freq == 0:
            agent.save_checkpoint(episode + 1, MODEL_DIR)
            logger.save_logs()
            logger.plot_progress()
            print(f"💾 Checkpoint saved at episode {episode + 1}")

    # Training complete
    training_time = time.time() - start_time
    env.close()
    
    # Save final checkpoint
    agent.save_checkpoint(args.episodes, MODEL_DIR)
    logger.save_logs()
    logger.plot_progress()

    # Print final summary
    print(f"\n" + "="*80)
    print(f"TRAINING COMPLETE")
    print("="*80)
    episodes_trained = args.episodes - start_episode
    print(f"Total episodes: {episodes_trained}")
    print(f"Total lines cleared: {lines_cleared_total}")
    
    if episodes_trained > 0:
        avg_lines_all = lines_cleared_total / episodes_trained
        print(f"Average lines per episode: {avg_lines_all:.3f}")
    
    print(f"First line at episode: {first_line_episode or 'Never'}")
    print(f"Training time: {training_time/60:.1f} minutes")
    
    if len(recent_rewards) > 0:
        print(f"\nRecent performance (last {len(recent_rewards)} episodes):")
        print(f"  Average reward: {np.mean(recent_rewards):.1f}")
        print(f"  Average steps: {np.mean(recent_steps):.1f}")
        print(f"  Average lines/episode: {np.mean(recent_lines):.3f}")

    # Provide feedback
    if lines_cleared_total == 0:
        print("\n⚠️  No lines cleared! This shouldn't happen with reward shaping.")
        print("  Check:")
        print("  1. Is complete vision enabled? (should see '4-channel' above)")
        print("  2. Did epsilon stay high enough? (should be 0.5-0.8 initially)")
        print("  3. Run for more episodes (try 1000+)")
    elif lines_cleared_total < episodes_trained * 0.1:
        print("\n⚠️  Low line clearing rate. Try:")
        print("  1. Use --reward_shaping aggressive")
        print("  2. Train for more episodes (2000+)")
        print("  3. Start completely fresh with --force_fresh")
    else:
        print("\n✅ Training successful!")
        print(f"  Line clearing rate: {lines_cleared_total/episodes_trained:.2f} lines/episode")
        print("  Continue training for better performance!")

    print("\n" + "="*80)


def main():
    """Main entry point"""
    args = parse_args()

    print("🎯 Tetris AI Training - FIXED VERSION")
    print("Key fixes:")
    print("  ✅ Reward shaping always enabled")
    print("  ✅ Strong line clear bonuses (500-10000)")
    print("  ✅ Positive survival rewards (+2.0 per step)")
    print()

    train(args)


if __name__ == "__main__":
    main()
