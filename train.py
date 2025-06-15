#!/usr/bin/env python3
"""
Enhanced training script with reward shaping integration and ablation studies
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

# Import Agent from the corrected integration code


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


def run_ablation_study(args):
    """Run comprehensive ablation study"""
    print("Running Reward Shaping Ablation Study")
    print("=" * 60)

    shaping_methods = [
        ('none', 'No reward shaping'),
        ('simple', 'Height + Hole penalties only'),
        ('full', 'Full multi-objective shaping')
    ]

    results = {}

    for method, description in shaping_methods:
        print(f"\nTesting: {description}")
        print("-" * 40)

        # Override shaping method
        args.reward_shaping = method
        args.episodes = 200  # Shorter for ablation
        args.experiment_name = f"ablation_{method}"

        # Run training
        result = train_single_configuration(args)
        results[method] = result

        print(f"Results for {method}:")
        print(f"  Final avg reward: {result['final_avg_reward']:.2f}")
        print(f"  Max reward: {result['max_reward']:.2f}")
        print(f"  Avg lines cleared: {result['avg_lines_cleared']:.2f}")
        print(f"  Training time: {result['training_time']:.1f}s")

    # Save ablation results
    ablation_file = os.path.join(
        LOG_DIR, f"ablation_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(ablation_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("ABLATION STUDY COMPLETE")
    print("=" * 60)
    print("Results summary:")
    for method, result in results.items():
        print(f"{method:10s}: Avg={result['final_avg_reward']:6.1f}, "
              f"Max={result['max_reward']:6.1f}, "
              f"Lines={result['avg_lines_cleared']:4.1f}")
    print(f"\nDetailed results saved to: {ablation_file}")


def train_single_configuration(args):
    """Train with a single configuration and return results"""
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

    # Initialize agent with reward shaping
    agent = Agent(
        obs_space=env.observation_space,
        action_space=env.action_space,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        model_type=args.model_type,
        reward_shaping=args.reward_shaping,
        shaping_config=shaping_config
    )

    # Resume if requested
    start_episode = 0
    if args.resume:
        if agent.load_checkpoint(latest=True, model_dir=MODEL_DIR):
            start_episode = agent.episodes_done

    # Initialize logger
    experiment_name = args.experiment_name or f"tetris_{args.reward_shaping}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = TrainingLogger(LOG_DIR, experiment_name)

    # Training metrics tracking
    episode_rewards = []
    episode_lines_cleared = []
    original_rewards = []  # For comparison

    # Training loop
    print(f"Starting training for {args.episodes} episodes...")
    print(f"Reward shaping: {args.reward_shaping}")

    for episode in range(start_episode, args.episodes):
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
        if (episode + 1) % args.log_freq == 0:
            stats = agent.get_stats()
            shaping_analysis = agent.get_shaping_analysis()

            print(f"Episode {episode+1:4d} | "
                  f"Reward: {episode_reward:6.1f} | "
                  f"Orig: {original_episode_reward:6.1f} | "
                  f"Avg: {stats['avg_reward']:6.1f} | "
                  f"Lines: {lines_cleared_this_episode:2d} | "
                  f"Epsilon: {agent.epsilon:.4f}")

            if shaping_analysis and args.reward_shaping != 'none':
                improvement = shaping_analysis.get('reward_improvement', 0)
                print(f"         | Shaping improvement: {improvement:+5.1f}")

        # Periodic evaluation
        if (episode + 1) % args.eval_freq == 0:
            print(f"\nEvaluating at episode {episode+1}...")
            eval_results = evaluate_agent(agent, env, n_episodes=5)
            print(
                f"Evaluation - Mean: {eval_results['mean_reward']:.1f} ± {eval_results['std_reward']:.1f}")

        # Periodic saves
        if (episode + 1) % args.save_freq == 0:
            agent.save_checkpoint(episode + 1, MODEL_DIR)
            logger.save_logs()
            logger.plot_progress()

    # Final save and cleanup
    agent.save_checkpoint(episode + 1, MODEL_DIR)
    logger.save_logs()
    logger.plot_progress()

    training_time = time.time() - start_time
    env.close()

    # Return results for ablation study
    return {
        'final_avg_reward': np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else np.mean(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'avg_lines_cleared': np.mean(episode_lines_cleared),
        'training_time': training_time,
        'shaping_analysis': agent.get_shaping_analysis(),
        'total_episodes': len(episode_rewards)
    }


def train(args):
    """Main training function with reward shaping"""
    print("Starting Tetris AI Training with Reward Shaping")
    print("=" * 60)
    print(f"Reward shaping method: {args.reward_shaping}")

    # Print system information
    print_system_info()

    # Create directories
    make_dir(MODEL_DIR)
    make_dir(LOG_DIR)

    if args.ablation_study:
        run_ablation_study(args)
    else:
        result = train_single_configuration(args)

        print("\n" + "=" * 60)
        print("TRAINING COMPLETED")
        print("=" * 60)
        print(f"Final average reward: {result['final_avg_reward']:.2f}")
        print(f"Maximum reward achieved: {result['max_reward']:.2f}")
        print(
            f"Average lines cleared per episode: {result['avg_lines_cleared']:.2f}")
        print(f"Total training time: {result['training_time']/3600:.2f} hours")

        # Print shaping analysis
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
    train(args)


if __name__ == "__main__":
    main()
