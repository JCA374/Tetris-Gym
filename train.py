#!/usr/bin/env python3
"""
Training script for Tetris AI using Tetris Gymnasium - FIXED JSON serialization
"""

from src.utils import TrainingLogger, print_system_info, benchmark_environment, make_dir
from src.agent import Agent
from config import make_env, ENV_NAME, LR, GAMMA, BATCH_SIZE, MAX_EPISODES, MODEL_DIR, LOG_DIR
import os
import sys
import argparse
import time
from datetime import datetime

import torch
import numpy as np
from tqdm import tqdm

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


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


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Tetris AI')
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

    return parser.parse_args()


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


def train(args):
    """Main training function"""
    print("Starting Tetris AI Training")
    print("=" * 60)

    # Print system information
    print_system_info()

    # Create directories
    make_dir(MODEL_DIR)
    make_dir(LOG_DIR)

    # Initialize environment
    print(f"Creating environment: {ENV_NAME}")
    render_mode = None if args.no_render else "rgb_array"
    env = make_env(ENV_NAME, render_mode=render_mode)

    print(f"Environment created successfully!")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Benchmark environment
    print("\nBenchmarking environment...")
    benchmark_results = benchmark_environment(env, n_steps=100)

    # Initialize agent
    print(f"\nInitializing agent with model type: {args.model_type}")
    agent = Agent(
        obs_space=env.observation_space,
        action_space=env.action_space,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        model_type=args.model_type
    )

    # Resume training if requested
    start_episode = 0
    if args.resume:
        print("Attempting to resume from checkpoint...")
        if agent.load_checkpoint(latest=True, model_dir=MODEL_DIR):
            start_episode = agent.episodes_done
            print(f"Resumed from episode {start_episode}")
        else:
            print("No checkpoint found, starting fresh")

    # Initialize logger
    experiment_name = args.experiment_name or f"tetris_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = TrainingLogger(LOG_DIR, experiment_name)

    # Save training configuration (with JSON serialization fix)
    config = {
        'args': vars(args),
        'model_type': args.model_type,
        'environment': ENV_NAME,
        # Fix JSON serialization
        'benchmark_results': convert_numpy_types(benchmark_results),
        'start_time': datetime.now().isoformat(),
    }

    import json
    config_path = os.path.join(logger.experiment_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\nStarting training for {args.episodes} episodes")
    print(f"Experiment: {experiment_name}")
    print(f"Logs: {logger.experiment_dir}")
    print("=" * 60)

    # Training loop
    best_reward = float('-inf')
    start_time = time.time()

    try:
        for episode in range(start_episode, args.episodes):
            # Reset environment
            obs, info = env.reset()
            episode_reward = 0
            episode_steps = 0
            episode_start = time.time()

            done = False
            while not done:
                # Select action
                action = agent.select_action(obs)

                # Take step
                next_obs, reward, terminated, truncated, info = env.step(
                    action)
                done = terminated or truncated

                # Store experience
                agent.remember(obs, action, reward, next_obs, done)

                # Learn
                agent.learn()

                # Update for next iteration
                obs = next_obs
                episode_reward += reward
                episode_steps += 1

            # Episode completed
            episode_time = time.time() - episode_start
            agent.add_episode_reward(episode_reward)

            # Log episode
            logger.log_episode(
                episode=episode + 1,
                reward=episode_reward,
                steps=episode_steps,
                epsilon=agent.epsilon,
                episode_time=episode_time,
                total_steps=agent.steps_done
            )

            # Periodic logging
            if (episode + 1) % args.log_freq == 0:
                elapsed_time = time.time() - start_time
                avg_time_per_episode = elapsed_time / \
                    (episode - start_episode + 1)
                eta = avg_time_per_episode * (args.episodes - episode - 1)

                stats = agent.get_stats()
                print(f"Episode {episode+1:4d} | "
                      f"Reward: {episode_reward:6.1f} | "
                      f"Avg: {stats['avg_reward']:6.1f} | "
                      f"Epsilon: {agent.epsilon:.4f} | "
                      f"Steps: {episode_steps:3d} | "
                      f"ETA: {eta/60:.1f}m")

            # Periodic evaluation
            if (episode + 1) % args.eval_freq == 0:
                print(f"\nEvaluating at episode {episode+1}...")
                eval_results = evaluate_agent(agent, env, n_episodes=5)
                print(
                    f"Evaluation - Mean: {eval_results['mean_reward']:.1f} ± {eval_results['std_reward']:.1f}")

                # Log evaluation results
                logger.log_episode(
                    episode=episode + 1,
                    reward=eval_results['mean_reward'],
                    steps=eval_results['mean_steps'],
                    epsilon=agent.epsilon,
                    evaluation=True,
                    **{f"eval_{k}": v for k, v in eval_results.items()}
                )

                # Save best model
                if eval_results['mean_reward'] > best_reward:
                    best_reward = eval_results['mean_reward']
                    best_model_path = os.path.join(MODEL_DIR, 'best_model.pth')
                    torch.save(agent.q_network.state_dict(), best_model_path)
                    print(f"New best model saved! Reward: {best_reward:.1f}")

            # Periodic saves
            if (episode + 1) % args.save_freq == 0:
                agent.save_checkpoint(episode + 1, MODEL_DIR)
                logger.save_logs()
                logger.plot_progress()

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise

    finally:
        # Final save and cleanup
        print("\nSaving final checkpoint...")
        agent.save_checkpoint(episode + 1, MODEL_DIR)
        logger.save_logs()
        logger.plot_progress()

        # Print final statistics
        total_time = time.time() - start_time
        final_stats = agent.get_stats()

        print("\n" + "=" * 60)
        print("TRAINING COMPLETED")
        print("=" * 60)
        print(f"Total episodes: {final_stats['episodes']}")
        print(f"Total steps: {final_stats['steps']}")
        print(f"Training time: {total_time/3600:.2f} hours")
        print(f"Final epsilon: {final_stats['epsilon']:.4f}")
        print(f"Best reward: {best_reward:.1f}")
        print(f"Average reward (last 100): {final_stats['avg_reward']:.1f}")
        print(f"Logs saved to: {logger.experiment_dir}")
        print("=" * 60)

        env.close()


def main():
    """Main entry point"""
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
