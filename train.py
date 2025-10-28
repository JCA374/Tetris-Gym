# train.py
"""
FIXED Tetris Training Script with Forced Action Diversity
"""

import gymnasium as gym
import numpy as np
import time
import argparse
import os
from collections import deque

from config import (
    make_env, 
    discover_action_meanings,
    ACTION_NOOP, ACTION_LEFT, ACTION_RIGHT, ACTION_DOWN,
    ACTION_ROTATE_CW, ACTION_ROTATE_CCW, ACTION_HARD_DROP, ACTION_SWAP
)
from src.agent import Agent
from src.reward_shaping import (
    balanced_reward_shaping,
    aggressive_reward_shaping,
    positive_reward_shaping
)
from src.training_logger import TrainingLogger
from src.utils import make_dir


# Directories
MODEL_DIR = "models"
LOG_DIR = "logs"


def parse_args():
    parser = argparse.ArgumentParser(description="Train Tetris DQN Agent - FIXED VERSION")
    
    # Training
    parser.add_argument('--episodes', type=int, default=10000, help='Number of episodes')
    parser.add_argument('--max_steps', type=int, default=1000, help='Max steps per episode')
    parser.add_argument('--save_freq', type=int, default=500, help='Save checkpoint every N episodes')
    parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint')
    
    # Agent hyperparameters
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--memory_size', type=int, default=50000, help='Replay buffer size')
    parser.add_argument('--target_update', type=int, default=500, help='Target network update frequency')
    
    # Exploration
    parser.add_argument('--epsilon_start', type=float, default=1.0, help='Starting epsilon')
    parser.add_argument('--epsilon_end', type=float, default=0.05, help='Final epsilon')  # INCREASED from 0.01
    parser.add_argument('--epsilon_decay', type=float, default=0.9995, help='Epsilon decay')
    parser.add_argument('--epsilon_method', type=str, default='exponential', 
                        choices=['exponential', 'linear'], help='Epsilon decay method')
    
    # Model
    parser.add_argument('--model_type', type=str, default='dqn', 
                        choices=['dqn', 'mlp'], help='Model architecture')
    
    # Reward shaping
    parser.add_argument('--reward_shaping', type=str, default='balanced',
                        choices=['none', 'balanced', 'aggressive', 'positive'],
                        help='Reward shaping strategy')
    
    # Experiment
    parser.add_argument('--experiment_name', type=str, default=None, help='Custom experiment name')
    
    # NEW: Force action diversity
    parser.add_argument('--force_exploration', action='store_true', 
                        help='Force horizontal movement exploration early in training')
    
    return parser.parse_args()


def force_diverse_actions(action, step, episode, force_exploration=True):
    """
    Force agent to use diverse actions during early training
    This prevents single-column stacking behavior
    """
    if not force_exploration:
        return action
    
    # Force horizontal movement for first 3000 episodes
    if episode < 3000:
        # Every 10 steps, force a random horizontal action
        if step % 10 < 3:  # 30% of the time
            return np.random.choice([ACTION_LEFT, ACTION_RIGHT])
        
        # Prevent too many rotations
        if action in [ACTION_ROTATE_CW, ACTION_ROTATE_CCW]:
            # 50% chance to replace rotation with horizontal movement
            if np.random.random() < 0.5:
                return np.random.choice([ACTION_LEFT, ACTION_RIGHT])
    
    return action


def main():
    args = parse_args()
    
    # Create directories
    make_dir(MODEL_DIR)
    make_dir(LOG_DIR)
    
    print("="*80)
    print("🎮 TETRIS DQN TRAINING - FIXED FOR EXPLORATION")
    print("="*80)
    
    # Environment
    print("\n📦 Creating environment...")
    env = make_env(render_mode="rgb_array")
    discover_action_meanings(env)
    
    print(f"✅ Environment created: {env.spec.id}")
    print(f"   Action space: {env.action_space} (n={env.action_space.n})")
    print(f"   Observation space: {env.observation_space}")
    
    # Print action mappings
    from config import get_action_meanings
    action_meanings = get_action_meanings()
    print(f"\n🎯 Action Mappings (n={len(action_meanings)}):")
    for action_id, meaning in action_meanings.items():
        print(f"   {action_id}: {meaning}")
    
    # Agent
    print(f"\n🤖 Initializing agent...")
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
    print(f"✅ Agent initialized for {args.episodes} episodes")
    print(f"   Epsilon method: {args.epsilon_method}")
    print(f"   Force exploration: {args.force_exploration}")
    
    # Resume from checkpoint
    start_episode = 0
    if args.resume:
        print(f"\n📁 Looking for checkpoint...")
        if agent.load_checkpoint(latest=True):
            start_episode = agent.episodes_done
            print(f"✅ Resumed from episode {start_episode}")
        else:
            print(f"⚠️  No checkpoint found, starting fresh")
    
    # Logger
    experiment_name = args.experiment_name or f"tetris_{int(time.time())}"
    logger = TrainingLogger(LOG_DIR, experiment_name)
    logger.log_config(vars(args))
    print(f"✅ Logger initialized: {experiment_name}")
    
    # Reward shaping
    shaper_map = {
        'none': lambda obs, a, r, d, i: r,
        'balanced': balanced_reward_shaping,
        'aggressive': aggressive_reward_shaping,
        'positive': positive_reward_shaping
    }
    shaper_fn = shaper_map[args.reward_shaping]
    print(f"Reward shaping: {args.reward_shaping}")
    
    # Training metrics
    lines_cleared_total = 0
    first_line_episode = None
    recent_rewards = deque(maxlen=100)
    recent_lines = deque(maxlen=100)
    recent_steps = deque(maxlen=100)
    
    best_lines_episode = 0
    best_reward = float('-inf')
    
    print(f"\n🚀 Starting training")
    print(f"Episodes: {start_episode + 1} to {args.episodes}")
    print(f"Epsilon: {agent.epsilon:.3f}")
    if args.force_exploration:
        print(f"⚡ FORCED EXPLORATION ENABLED for first 3000 episodes")
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
        
        # Track action usage this episode
        action_counts = {i: 0 for i in range(env.action_space.n)}
        
        while not done and episode_steps < args.max_steps:
            # Select action from agent
            raw_action = agent.select_action(obs)
            
            # Apply forced exploration
            action = force_diverse_actions(
                raw_action, 
                episode_steps, 
                episode,
                force_exploration=args.force_exploration
            )
            
            # Track action usage
            action_counts[action] += 1
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store raw reward
            raw_reward = reward
            original_reward += raw_reward
            
            # Apply reward shaping
            shaped_reward = shaper_fn(obs, action, raw_reward, done, info)
            
            # Store experience with ORIGINAL action (not forced)
            # This ensures the agent learns from its own mistakes
            agent.remember(obs, raw_action, shaped_reward, next_obs, done, info, raw_reward)
            
            # Learn every 4 steps
            if episode_steps % 4 == 0 and len(agent.memory) >= agent.batch_size:
                agent.learn()
            
            # Update metrics
            episode_reward += shaped_reward
            episode_steps += 1
            
            # Track line clears
            line_keys = ['lines_cleared', 'cleared_lines', 'lines', 'n_lines', 'number_of_lines']
            lines = 0
            for key in line_keys:
                if key in info:
                    lines = info.get(key, 0) or 0
                    break
            
            if lines > 0:
                lines_this_episode += lines
                lines_cleared_total += lines
                if first_line_episode is None:
                    first_line_episode = episode + 1
                    print(f"\n🎉🎉🎉 FIRST LINE CLEARED! Episode {first_line_episode} 🎉🎉🎉\n")
            
            obs = next_obs
        
        # End of episode
        agent.end_episode(episode_reward, episode_steps, lines_this_episode, original_reward)
        
        # Track recent performance
        recent_rewards.append(episode_reward)
        recent_lines.append(lines_this_episode)
        recent_steps.append(episode_steps)
        
        # Track best
        if lines_this_episode > best_lines_episode:
            best_lines_episode = lines_this_episode
            print(f"   🏆 New best: {best_lines_episode} lines!")
        if episode_reward > best_reward:
            best_reward = episode_reward
        
        # Log episode
        logger.log_episode(
            episode=episode + 1,
            reward=episode_reward,
            steps=episode_steps,
            epsilon=agent.epsilon,
            lines_cleared=lines_this_episode,
            total_lines=lines_cleared_total,
            original_reward=original_reward,
            shaped_reward_used=(args.reward_shaping != 'none')
        )
        
        # Print progress
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(recent_rewards)
            avg_steps = np.mean(recent_steps)
            avg_lines = np.mean(recent_lines)
            
            print(f"\nEpisode {episode + 1}/{args.episodes}")
            print(f"  Reward: {episode_reward:.1f} (avg: {avg_reward:.1f})")
            print(f"  Steps: {episode_steps} (avg: {avg_steps:.1f})")
            print(f"  Lines: {lines_this_episode} (avg: {avg_lines:.3f})")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            print(f"  Total lines cleared: {lines_cleared_total}")
            print(f"  Memory size: {len(agent.memory)}")
            
            # Print action distribution for this episode
            total_actions = sum(action_counts.values())
            if total_actions > 0:
                print(f"  Action distribution (episode {episode + 1}):")
                action_meanings = get_action_meanings()
                for action_id in sorted(action_counts.keys()):
                    count = action_counts[action_id]
                    pct = 100 * count / total_actions
                    meaning = action_meanings.get(action_id, f"UNKNOWN({action_id})")
                    if count > 0:  # Only show used actions
                        print(f"    {meaning:12s}: {count:3d} ({pct:5.1f}%)")
        
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
        print("1. Try --force_exploration flag")
        print("2. Increase epsilon_end to 0.1")
        print("3. Check environment with python tests/test_setup.py")
    elif lines_cleared_total < episodes_trained * 0.1:
        print("\n⚠️  Low line clearing rate. Try:")
        print("1. Train for more episodes (15000+)")
        print("2. Use --force_exploration")
        print("3. Use --reward_shaping aggressive")
    else:
        print("\n✅ Training successful!")
        print(f"Line clearing rate: {lines_cleared_total/episodes_trained:.2f} lines/episode")
    
    print(f"\n📊 Next steps:")
    print(f"1. Evaluate: python evaluate.py --model_path {final_path}")
    print(f"2. Continue: python train.py --resume --episodes {args.episodes + 5000}")
    print("3. Watch agent: python evaluate.py --episodes 5 --render")


if __name__ == "__main__":
    main()