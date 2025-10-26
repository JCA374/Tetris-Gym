# train.py
"""Training script for Tetris AI - FIXED with action discovery and line tracking"""

import argparse
import time
import numpy as np
import torch
from datetime import datetime
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import make_env, MODEL_DIR, LOG_DIR
from src.agent import Agent
from src.training_logger import TrainingLogger
from src.reward_shaping import (
    balanced_reward_shaping,
    aggressive_reward_shaping,
    positive_reward_shaping,
    extract_board_from_obs,
    get_column_heights,
    count_holes,
    calculate_bumpiness
)
from src.utils import make_dir


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Tetris AI with DQN')
    
    # Training parameters
    parser.add_argument('--episodes', type=int, default=10000,
                       help='Number of training episodes')
    parser.add_argument('--lr', type=float, default=0.0001,
                       help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--memory_size', type=int, default=100000,
                       help='Replay buffer size')
    parser.add_argument('--target_update', type=int, default=1000,
                       help='Target network update frequency')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='dqn',
                       choices=['dqn', 'dueling_dqn'],
                       help='Model architecture type')
    
    # Epsilon parameters
    parser.add_argument('--epsilon_start', type=float, default=1.0,
                       help='Starting epsilon for exploration')
    parser.add_argument('--epsilon_end', type=float, default=0.01,
                       help='Minimum epsilon')
    parser.add_argument('--epsilon_decay', type=float, default=0.9999,
                       help='Epsilon decay rate')
    parser.add_argument('--epsilon_method', type=str, default='exponential',
                       choices=['exponential', 'linear', 'adaptive'],
                       help='Epsilon decay method')
    
    # Environment parameters
    parser.add_argument('--render', action='store_true',
                       help='Render the environment during training')
    parser.add_argument('--use_complete_vision', action='store_true', default=True,
                       help='Use complete vision wrapper')
    parser.add_argument('--use_cnn', action='store_true', default=False,
                       help='Use CNN model (not implemented)')
    
    # Reward shaping
    parser.add_argument('--reward_shaping', type=str, default='balanced',
                       choices=['none', 'balanced', 'aggressive', 'positive'],
                       help='Reward shaping strategy')
    
    # Logging parameters
    parser.add_argument('--log_freq', type=int, default=100,
                       help='Logging frequency (episodes)')
    parser.add_argument('--save_freq', type=int, default=500,
                       help='Model save frequency (episodes)')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Name for this experiment')
    
    # Resume/Fresh start
    parser.add_argument('--resume', action='store_true',
                       help='Resume from latest checkpoint')
    parser.add_argument('--force_fresh', action='store_true',
                       help='Force fresh start, ignore checkpoints')
    
    return parser.parse_args()


def main():
    """Main training loop"""
    args = parse_args()
    
    # Setup directories
    make_dir(MODEL_DIR)
    make_dir(LOG_DIR)
    
    print("="*80)
    print("🎮 TETRIS AI TRAINING - FIXED VERSION")
    print("="*80)
    
    # Create environment and discover actions
    print("\n📦 Setting up environment...")
    env = make_env(
        render_mode="human" if args.render else None,
        use_complete_vision=args.use_complete_vision,
        use_cnn=args.use_cnn
    )
    print(f"✅ Environment created")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")
    
    # Action meanings are discovered in make_env()
    from config import ACTION_LEFT, ACTION_RIGHT, ACTION_HARD_DROP, ACTION_MEANINGS
    print(f"\n🎯 Action mappings:")
    if ACTION_MEANINGS:
        for i, meaning in enumerate(ACTION_MEANINGS):
            print(f"   {i}: {meaning}")
    else:
        print(f"   LEFT={ACTION_LEFT}, RIGHT={ACTION_RIGHT}, HARD_DROP={ACTION_HARD_DROP}")
    
    # Create agent
    print("\n🤖 Creating agent...")
    agent = Agent(
        obs_space=env.observation_space,
        action_space=env.action_space,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        memory_size=args.memory_size,
        target_update=args.target_update,
        model_type=args.model_type,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        epsilon_decay_method=args.epsilon_method,
        reward_shaping=args.reward_shaping,
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
        else:
            print("❌ No checkpoint found - starting fresh")
    else:
        print("🆕 Starting fresh training")
    
    # Setup experiment logging
    experiment_name = args.experiment_name or f"fixed_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = TrainingLogger(LOG_DIR, experiment_name)
    
    # Select reward shaping function
    shaping_functions = {
        'none': lambda o, a, r, d, i: r,
        'positive': positive_reward_shaping,
        'aggressive': aggressive_reward_shaping,
        'balanced': balanced_reward_shaping
    }
    shaper_fn = shaping_functions[args.reward_shaping]
    
    print(f"\n🎯 Reward shaping: {args.reward_shaping}")
    
    # Training metrics
    lines_cleared_total = 0
    first_line_episode = None
    recent_rewards = []
    recent_lines = []
    recent_steps = []
    
    # Track best performance
    best_lines_episode = 0
    best_reward = float('-inf')
    
    print(f"\n🚀 Starting training")
    print(f"Episodes: {start_episode + 1} to {args.episodes}")
    print(f"Epsilon: {agent.epsilon:.3f}")
    print("-" * 80)
    
    start_time = time.time()
    
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
            
            # Apply reward shaping
            shaped_reward = shaper_fn(obs, action, raw_reward, done, info)
            
            # Store experience with shaped reward
            agent.remember(obs, action, shaped_reward, next_obs, done, info, raw_reward)
            
            # Learn every 4 steps
            if episode_steps % 4 == 0 and len(agent.memory) >= agent.batch_size:
                agent.learn()
            
            # Update metrics
            episode_reward += shaped_reward
            episode_steps += 1
            
            # FIXED: Track line clears with multiple possible keys
            line_keys = ['lines_cleared', 'cleared_lines', 'lines', 'n_lines', 'number_of_lines']
            lines = 0
            for key in line_keys:
                if key in info:
                    lines = info.get(key, 0) or 0
                    if lines > 0 and key != 'lines_cleared':
                        print(f"   📝 Note: Lines tracked using key '{key}'")
                    break
            
            if lines > 0:
                lines_this_episode += lines
                lines_cleared_total += lines
                if first_line_episode is None:
                    first_line_episode = episode + 1
                    print(f"\n🎉🎉🎉 FIRST LINE CLEARED! Episode {first_line_episode} 🎉🎉🎉\n")
            
            obs = next_obs
        
        # End of episode - update agent
        agent.end_episode(episode_reward, episode_steps, lines_this_episode, original_reward)
        
        # Track recent performance
        recent_rewards.append(episode_reward)
        recent_lines.append(lines_this_episode)
        recent_steps.append(episode_steps)
        if len(recent_rewards) > 100:
            recent_rewards.pop(0)
            recent_lines.pop(0)
            recent_steps.pop(0)
        
        # Track best performance
        if lines_this_episode > best_lines_episode:
            best_lines_episode = lines_this_episode
            print(f"   🏆 New best: {best_lines_episode} lines!")
        if episode_reward > best_reward:
            best_reward = episode_reward
        
        # Log episode data
        logger.log_episode(
            episode=episode + 1,
            reward=episode_reward,
            steps=episode_steps,
            epsilon=agent.epsilon,
            lines_cleared=lines_this_episode,
            original_reward=original_reward,
            total_lines=lines_cleared_total,
            shaped_reward_used=(args.reward_shaping != 'none')
        )
        
        # Print progress
        if (episode + 1) % args.log_freq == 0 or lines_this_episode > 0:
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0
            avg_lines = np.mean(recent_lines) if recent_lines else 0
            avg_steps = np.mean(recent_steps) if recent_steps else 0
            
            print(f"Episode {episode+1:4d} | "
                  f"Lines: {lines_this_episode} (Total: {lines_cleared_total:3d}) | "
                  f"Reward: {episode_reward:7.1f} (Avg: {avg_reward:6.1f}) | "
                  f"Steps: {episode_steps:3d} (Avg: {avg_steps:4.1f}) | "
                  f"Lines/Ep: {avg_lines:.2f} | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"Best: {best_lines_episode}L")
            
            # Detailed board analysis
            if episode % (args.log_freq * 5) == 0:  # Every 5x log frequency
                board = extract_board_from_obs(next_obs)
                heights = get_column_heights(board)
                
                # Calculate max row fullness
                max_row_fullness = 0
                for r in range(board.shape[0]):
                    filled = int((board[r, :] > 0).sum())
                    if filled > max_row_fullness:
                        max_row_fullness = filled
                
                max_height = max(heights) if heights else 0
                height_variance = float(np.var(heights)) if len(heights) else 0.0
                holes = count_holes(board)
                bumpiness = calculate_bumpiness(board)
                
                print("  📊 Board Stats:")
                print(f"     Max row fullness: {max_row_fullness}/10 cells")
                print(f"     Column heights: {heights}")
                print(f"     Max height: {max_height}, Variance: {height_variance:.2f}")
                print(f"     Holes: {holes}, Bumpiness: {bumpiness:.1f}")
                
                # Check for problematic patterns
                if height_variance > 50 and max_height > 15:
                    if heights[0] < 3 and heights[-1] < 3 and max(heights[3:7]) > 15:
                        print("     ⚠️  WARNING: Center stacking detected! Sides are empty.")
                        print("     💡 This suggests action mapping or exploration issues.")
        
        # Save checkpoint periodically
        if (episode + 1) % args.save_freq == 0:
            checkpoint_path = agent.save_checkpoint(episode + 1, MODEL_DIR)
            logger.save_logs()
            logger.plot_progress()
            print(f"💾 Checkpoint saved: {checkpoint_path}")
            print(f"   Best performance: {best_lines_episode} lines, {best_reward:.1f} reward")
    
    # Training complete
    training_time = time.time() - start_time
    env.close()
    
    # Save final checkpoint
    final_path = agent.save_checkpoint(args.episodes, MODEL_DIR)
    logger.save_logs()
    logger.plot_progress()
    
    # Print final summary
    print(f"\n" + "="*80)
    print(f"🏁 TRAINING COMPLETE")
    print("="*80)
    episodes_trained = args.episodes - start_episode
    print(f"Total episodes: {episodes_trained}")
    print(f"Total lines cleared: {lines_cleared_total}")
    print(f"Best single episode: {best_lines_episode} lines")
    print(f"Best reward: {best_reward:.1f}")
    
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
        print("\n❌ NO LINES CLEARED!")
        print("Debugging suggestions:")
        print("1. Check action mappings printed above")
        print("2. Verify LEFT/RIGHT actions work: python test_actions.py")
        print("3. Check board analysis for center stacking pattern")
        print("4. Try manual play: python play_manual.py")
    elif lines_cleared_total < episodes_trained * 0.1:
        print("\n⚠️  Low line clearing rate. Try:")
        print("1. Train for more episodes (5000+)")
        print("2. Adjust exploration parameters")
        print("3. Use --reward_shaping aggressive")
    else:
        print("\n✅ Training successful!")
        print(f"Line clearing rate: {lines_cleared_total/episodes_trained:.2f} lines/episode")
        print("\n📊 Next steps:")
        print(f"1. Evaluate: python evaluate.py --model_path {final_path}")
        print(f"2. Continue: python train.py --resume --episodes {args.episodes + 5000}")
        print("3. Visualize: python visualize_training.py")


if __name__ == "__main__":
    main()