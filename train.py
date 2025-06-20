#!/usr/bin/env python3
"""
Enhanced training script with reward shaping integration and ablation studies
FIXED: Variable scope errors and proper episode handling
"""

from config import make_env, ENV_NAME, LR, GAMMA, BATCH_SIZE, MAX_EPISODES, MODEL_DIR, LOG_DIR
from src.agent import Agent
from src.utils import TrainingLogger, print_system_info, benchmark_environment, make_dir
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
    """Enhanced argument parsing with reward shaping options"""
    parser = argparse.ArgumentParser(
        description='Train Tetris AI with Reward Shaping')

    # Standard training arguments
    parser.add_argument('--episodes', type=int, default=MAX_EPISODES,
                        help=f'Number of episodes to train (default: {MAX_EPISODES})')
    parser.add_argument('--lr', type=float, default=LR,
                        help=f'Learning rate (default: {LR})')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help=f'Batch size (default: {BATCH_SIZE})')
    parser.add_argument('--gamma', type=float, default=GAMMA,
                        help=f'Discount factor (default: {GAMMA})')
    parser.add_argument('--model_type', type=str, default='dqn', choices=['dqn', 'dueling_dqn'],
                        help='Model architecture type')

    # Reward shaping arguments
    parser.add_argument('--reward_shaping', type=str, default='none',
                        choices=['none', 'simple', 'full'],
                        help='Type of reward shaping to use')
    parser.add_argument('--height_weight', type=float, default=-1.0,
                        help='Weight for height penalty (simple shaping)')
    parser.add_argument('--hole_weight', type=float, default=-2.0,
                        help='Weight for hole penalty (simple shaping)')

    # Training control
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from latest checkpoint')
    parser.add_argument('--no_render', action='store_true',
                        help='Disable rendering during training')
    parser.add_argument('--save_freq', type=int, default=100,
                        help='Save model every N episodes')
    parser.add_argument('--log_freq', type=int, default=10,
                        help='Log progress every N episodes')
    parser.add_argument('--eval_freq', type=int, default=50,
                        help='Evaluate model every N episodes')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name for this experiment')

    # Ablation study options
    parser.add_argument('--ablation_study', action='store_true',
                        help='Run ablation study comparing different shaping methods')

    return parser.parse_args()


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def evaluate_agent(agent, env, n_episodes=5):
    """Evaluate agent performance"""
    agent.q_network.eval()  # Set to evaluation mode

    total_rewards = []
    total_steps = []

    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False

        while not done:
            action = agent.select_action(obs, eval_mode=True)
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            episode_steps += 1
            done = terminated or truncated

        total_rewards.append(episode_reward)
        total_steps.append(episode_steps)

    agent.q_network.train()  # Set back to training mode

    return {
        'mean_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'mean_steps': np.mean(total_steps),
        'std_steps': np.std(total_steps),
        'min_reward': np.min(total_rewards),
        'max_reward': np.max(total_rewards),
    }


def train_single_configuration(args):
    """Train with a single configuration and return results - FIXED VERSION"""
    start_time = time.time()

    # Create environment
    render_mode = None if args.no_render else "rgb_array"
    env = make_env(ENV_NAME, render_mode=render_mode)

    # Setup reward shaping configuration
    shaping_config = {}
    if args.reward_shaping == 'simple':
        shaping_config = {
            'height_weight': args.height_weight,
            'hole_weight': args.hole_weight
        }

    # Initialize agent with reward shaping and max_episodes parameter
    agent = Agent(
        obs_space=env.observation_space,
        action_space=env.action_space,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        model_type=args.model_type,
        reward_shaping=args.reward_shaping,
        shaping_config=shaping_config,
        max_episodes=args.episodes  # 🔥 PASS max_episodes TO AGENT
    )

    # Resume if requested
    start_episode = 0
    if args.resume:
        if agent.load_checkpoint(latest=True, model_dir=MODEL_DIR):
            start_episode = agent.episodes_done
            print(f"Resuming from episode {start_episode}")
        else:
            print("No checkpoint found, starting fresh")

    # 🔥 CRITICAL FIX: Handle case where agent has already completed target episodes
    if start_episode >= args.episodes:
        print(f"⚠️  Agent already completed {start_episode} episodes (target: {args.episodes})")
        print(f"Options:")
        print(f"1. Extend training: python train.py --episodes {start_episode + 5000} --resume")
        print(f"2. Start fresh: python train.py --episodes {args.episodes}")
        print(f"3. Evaluate current model: python evaluate.py --episodes 20 --render")
        
        # Return current stats
        return {
            'final_avg_reward': np.mean(agent.total_rewards[-50:]) if agent.total_rewards else 0,
            'max_reward': np.max(agent.total_rewards) if agent.total_rewards else 0,
            'avg_lines_cleared': 0,  # Would need to calculate from episode_metrics
            'training_time': 0,
            'shaping_analysis': {},
            'total_episodes': start_episode,
            'note': 'Training already completed'
        }

    # Initialize logger
    experiment_name = args.experiment_name or f"tetris_{args.reward_shaping}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = TrainingLogger(LOG_DIR, experiment_name)

    # Training metrics tracking
    episode_rewards = []
    episode_lines_cleared = []
    original_rewards = []  # For comparison

    # 🔥 FIXED: Proper episode range calculation
    total_episodes_to_train = args.episodes
    episodes_to_train = total_episodes_to_train - start_episode
    
    print(f"Starting training for {episodes_to_train} episodes...")
    print(f"Episode range: {start_episode + 1} to {total_episodes_to_train}")
    print(f"Reward shaping: {args.reward_shaping}")
    print(f"Current epsilon: {agent.epsilon:.4f}")

    # 🔥 FIXED: Initialize episode variable properly
    episode = start_episode  # Initialize BEFORE the loop

    # Training loop with proper episode handling
    for episode_offset in range(episodes_to_train):
        episode = start_episode + episode_offset  # Current episode number
        
        obs, info = env.reset()
        episode_reward = 0
        original_episode_reward = 0  # Track original reward
        episode_steps = 0
        lines_cleared_this_episode = 0

        done = False
        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Track original reward
            original_episode_reward += reward

            # Store experience with info for reward shaping
            agent.remember(obs, action, reward, next_obs, done, info, reward)

            # Learn
            learning_metrics = agent.learn()

            # Update tracking
            episode_reward += reward  # This will be shaped if shaping is enabled
            episode_steps += 1
            lines_cleared_this_episode += info.get('lines_cleared', 0)

            obs = next_obs

        # Episode end
        agent.end_episode(episode_reward, episode_steps,
                          lines_cleared_this_episode, original_episode_reward)

        # Track metrics
        episode_rewards.append(episode_reward)
        episode_lines_cleared.append(lines_cleared_this_episode)
        original_rewards.append(original_episode_reward)

        # Log episode
        logger.log_episode(
            episode=episode + 1,
            reward=episode_reward,
            steps=episode_steps,
            epsilon=agent.epsilon,
            lines_cleared=lines_cleared_this_episode,
            original_reward=original_episode_reward,
            shaping_improvement=episode_reward - original_episode_reward
        )

        # Periodic logging
        if (episode_offset + 1) % args.log_freq == 0:
            stats = agent.get_stats()
            
            # Calculate recent averages
            recent_rewards = episode_rewards[-10:] if len(episode_rewards) >= 10 else episode_rewards
            recent_lines = episode_lines_cleared[-10:] if len(episode_lines_cleared) >= 10 else episode_lines_cleared
            
            print(f"Episode {episode+1:5d} | "
                  f"Reward: {episode_reward:6.1f} | "
                  f"Orig: {original_episode_reward:6.1f} | "
                  f"Avg: {np.mean(recent_rewards):6.1f} | "
                  f"Lines: {lines_cleared_this_episode:2d} | "
                  f"Avg Lines: {np.mean(recent_lines):4.1f} | "
                  f"ε: {agent.epsilon:.4f}")

        # Periodic evaluation
        if (episode_offset + 1) % args.eval_freq == 0:
            print(f"\nEvaluating at episode {episode+1}...")
            eval_results = evaluate_agent(agent, env, n_episodes=5)
            print(
                f"Evaluation - Mean: {eval_results['mean_reward']:.1f} ± {eval_results['std_reward']:.1f}")

        # Periodic saves
        if (episode_offset + 1) % args.save_freq == 0:
            agent.save_checkpoint(episode + 1, MODEL_DIR)  # episode is now properly defined
            logger.save_logs()
            logger.plot_progress()

    # 🔥 FIXED: Final save with proper episode number
    final_episode = episode + 1  # episode is defined from the loop
    agent.save_checkpoint(final_episode, MODEL_DIR)
    logger.save_logs()
    logger.plot_progress()

    training_time = time.time() - start_time
    env.close()

    # Return results for ablation study
    return {
        'final_avg_reward': np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else np.mean(episode_rewards) if episode_rewards else 0,
        'max_reward': np.max(episode_rewards) if episode_rewards else 0,
        'avg_lines_cleared': np.mean(episode_lines_cleared) if episode_lines_cleared else 0,
        'training_time': training_time,
        'shaping_analysis': agent.get_shaping_analysis(),
        'total_episodes': final_episode
    }


def train(args):
    """Main training function with reward shaping"""
    print("Starting Tetris AI Training with Reward Shaping")
    print("=" * 60)
    print(f"Reward shaping method: {args.reward_shaping}")
    print(f"Target episodes: {args.episodes}")

    # Print system information
    print_system_info()

    # Create directories
    make_dir(MODEL_DIR)
    make_dir(LOG_DIR)

    # Run training
    result = train_single_configuration(args)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)
    print(f"Final average reward: {result['final_avg_reward']:.2f}")
    print(f"Maximum reward achieved: {result['max_reward']:.2f}")
    print(f"Average lines cleared per episode: {result['avg_lines_cleared']:.2f}")
    print(f"Total training time: {result['training_time']/3600:.2f} hours")
    print(f"Total episodes completed: {result['total_episodes']}")

    # Print shaping analysis if available
    shaping_analysis = result['shaping_analysis']
    if shaping_analysis and args.reward_shaping != 'none':
        print("\nReward Shaping Analysis:")
        improvement = shaping_analysis.get('reward_improvement', 0)
        print(f"  Average reward improvement: {improvement:+.2f}")
        correlation = shaping_analysis.get('correlation', 0)
        print(f"  Original-Shaped correlation: {correlation:.3f}")


def main():
    """Main entry point"""
    args = parse_args()
    
    # 🔥 FIX: Validate episode count for very long training
    if args.episodes > 100000:
        print(f"⚠️  Very long training requested: {args.episodes:,} episodes")
        response = input("This will take days/weeks. Continue? (y/N): ")
        if response.lower() != 'y':
            print("Training cancelled.")
            return
            
    train(args)


if __name__ == "__main__":
    main()