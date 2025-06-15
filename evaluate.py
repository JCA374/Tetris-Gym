#!/usr/bin/env python3
"""
Evaluation script for Tetris AI using Tetris Gymnasium - FIXED RENDERING
"""

from src.utils import make_dir
from src.agent import Agent
from config import make_env, ENV_NAME, MODEL_DIR
import os
import sys
import argparse
import time
import numpy as np
import torch
import gymnasium as gym
from gymnasium.envs.registration import register

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate Tetris AI')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of episodes to evaluate (default: 10)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model checkpoint (default: latest)')
    parser.add_argument('--render', action='store_true',
                        help='Render the game during evaluation')
    parser.add_argument('--slow', action='store_true',
                        help='Slow down rendering for human viewing')
    parser.add_argument('--save_video', action='store_true',
                        help='Save video of gameplay')
    parser.add_argument('--model_type', type=str, default='dqn', choices=['dqn', 'dueling_dqn'],
                        help='Model architecture type')
    parser.add_argument('--detailed', action='store_true',
                        help='Show detailed statistics for each episode')

    return parser.parse_args()


class SynchronizedTetrisEnvironments:
    """
    Wrapper to keep two Tetris environments synchronized:
    - One for AI observations (wrapped with preprocessing)
    - One for human rendering (raw environment)
    """

    def __init__(self, render_mode="rgb_array", enable_rendering=False):
        # Create main environment for AI (wrapped)
        self.main_env = make_env(ENV_NAME, render_mode="rgb_array")

        # Create render environment if needed (raw, unwrapped)
        self.render_env = None
        self.enable_rendering = enable_rendering

        if enable_rendering:
            try:
                register(
                    id="TetrisEvalRender-v0",
                    entry_point="tetris_gymnasium.envs.tetris:Tetris",
                )
            except gym.error.Error:
                pass  # Already registered

            self.render_env = gym.make(
                "TetrisEvalRender-v0", render_mode=render_mode)
            print(f"✅ Render environment created with mode: {render_mode}")

    def reset(self, seed=None):
        """Reset both environments with the same seed"""
        if seed is None:
            seed = np.random.randint(0, 1000000)

        # Reset main environment
        obs, info = self.main_env.reset(seed=seed)

        # Reset render environment with same seed
        if self.render_env:
            self.render_env.reset(seed=seed)

        return obs, info

    def step(self, action):
        """Step both environments with the same action"""
        # Step main environment for observations and rewards
        obs, reward, terminated, truncated, info = self.main_env.step(action)

        # Step render environment to keep it synchronized
        if self.render_env:
            try:
                self.render_env.step(action)
            except Exception as e:
                print(f"Warning: Render environment step failed: {e}")

        return obs, reward, terminated, truncated, info

    def render(self):
        """Render the game"""
        if self.render_env:
            try:
                return self.render_env.render()
            except Exception as e:
                print(f"Warning: Rendering failed: {e}")
                return None
        else:
            # Fallback to main environment
            try:
                return self.main_env.render()
            except Exception as e:
                print(f"Warning: Main environment rendering failed: {e}")
                return None

    def close(self):
        """Close both environments"""
        self.main_env.close()
        if self.render_env:
            self.render_env.close()

    @property
    def observation_space(self):
        return self.main_env.observation_space

    @property
    def action_space(self):
        return self.main_env.action_space


def evaluate_model(agent, env, args):
    """
    Run evaluation for a given agent and environment.
    """
    print(f"Evaluating model for {args.episodes} episodes.")
    if args.render:
        print("🎮 Rendering enabled - you should see the Tetris game window")
    print("=" * 60)

    # Switch to eval mode
    agent.q_network.eval()

    episode_rewards = []
    episode_steps = []
    episode_times = []
    game_info = []

    for ep in range(args.episodes):
        print(f"\n🎯 Starting Episode {ep+1}/{args.episodes}")

        obs, info = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        start_time = time.time()

        while not done:
            # Select action (greedy evaluation)
            action = agent.select_action(obs, eval_mode=True)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Render if requested
            if args.render:
                env.render()
                if args.slow:
                    time.sleep(0.1)  # Slow down for human viewing

            total_reward += reward
            steps += 1

            # Print progress for longer episodes
            if args.detailed and (steps % 100 == 0):
                print(f"  Step {steps}: Reward {total_reward:.2f}")

            # Safety break for very long episodes
            if steps > 10000:
                print(f"  Episode too long, breaking at step {steps}")
                break

        duration = time.time() - start_time

        # Record metrics
        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        episode_times.append(duration)
        game_info.append({
            "episode": ep + 1,
            "reward": float(total_reward),
            "steps": int(steps),
            "time": float(duration),
            "avg_reward_per_step": float(total_reward / steps) if steps > 0 else 0.0
        })

        # Print episode summary
        print(f"✅ Episode {ep+1:3d}: Reward: {total_reward:7.1f}, "
              f"Steps: {steps:4d}, Time: {duration:5.2f}s, "
              f"Speed: {steps/duration:6.1f} steps/sec")

    # Back to train mode
    agent.q_network.train()

    return episode_rewards, episode_steps, episode_times, game_info


def print_statistics(episode_rewards, episode_steps, episode_times):
    """Print detailed statistics"""
    print("\n" + "=" * 60)
    print("EVALUATION STATISTICS")
    print("=" * 60)

    # Reward statistics
    print("\nReward Statistics:")
    print(f"  Mean:     {np.mean(episode_rewards):8.2f}")
    print(f"  Std:      {np.std(episode_rewards):8.2f}")
    print(f"  Min:      {np.min(episode_rewards):8.2f}")
    print(f"  Max:      {np.max(episode_rewards):8.2f}")
    print(f"  Median:   {np.median(episode_rewards):8.2f}")

    # Steps statistics
    print("\nSteps Statistics:")
    print(f"  Mean:     {np.mean(episode_steps):8.1f}")
    print(f"  Std:      {np.std(episode_steps):8.1f}")
    print(f"  Min:      {np.min(episode_steps):8.0f}")
    print(f"  Max:      {np.max(episode_steps):8.0f}")
    print(f"  Median:   {np.median(episode_steps):8.1f}")

    # Time statistics
    print("\nTime Statistics:")
    print(f"  Mean:     {np.mean(episode_times):8.2f}s")
    print(f"  Total:    {np.sum(episode_times):8.1f}s")
    print(
        f"  Avg/step: {np.sum(episode_times)/np.sum(episode_steps)*1000:8.2f}ms")

    # Performance metrics
    print("\nPerformance Metrics:")
    avg_reward_per_step = np.mean(
        [r/s for r, s in zip(episode_rewards, episode_steps)])
    print(f"  Avg reward per step: {avg_reward_per_step:8.4f}")
    print(
        f"  Steps per second:    {np.sum(episode_steps)/np.sum(episode_times):8.1f}")

    # Success metrics
    positive_rewards = [r for r in episode_rewards if r > 0]
    if positive_rewards:
        print(
            f"  Episodes with positive reward: {len(positive_rewards)}/{len(episode_rewards)} ({100*len(positive_rewards)/len(episode_rewards):.1f}%)")

    print("=" * 60)


def save_results(episode_rewards, episode_steps, episode_times, game_info, args):
    """Save evaluation results"""
    results_dir = os.path.join(MODEL_DIR, "evaluation_results")
    make_dir(results_dir)

    import json
    from datetime import datetime

    # Sanitize NumPy types for JSON
    for ep in game_info:
        for k, v in ep.items():
            if isinstance(v, np.generic):
                ep[k] = v.item()

    # Prepare results data
    results = {
        'evaluation_time': datetime.now().isoformat(),
        'args': vars(args),
        'summary': {
            'episodes': len(episode_rewards),
            'mean_reward': float(np.mean(episode_rewards)),
            'std_reward': float(np.std(episode_rewards)),
            'mean_steps': float(np.mean(episode_steps)),
            'std_steps': float(np.std(episode_steps)),
            'mean_time': float(np.mean(episode_times)),
            'total_time': float(np.sum(episode_times)),
        },
        'episodes': game_info
    }

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = os.path.join(results_dir, f"evaluation_{timestamp}.json")

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Also save as CSV
    import csv
    csv_file = os.path.join(results_dir, f"evaluation_{timestamp}.csv")

    with open(csv_file, 'w', newline='') as f:
        if game_info:
            writer = csv.DictWriter(f, fieldnames=game_info[0].keys())
            writer.writeheader()
            writer.writerows(game_info)

    print(f"CSV data saved to: {csv_file}")


def main():
    """Main evaluation function"""
    args = parse_args()

    print("🎮 Tetris AI Evaluation")
    print("=" * 60)

    # Determine render mode
    if args.render:
        render_mode = "human"
        print("🖥️  Rendering mode: HUMAN (visual window)")
    else:
        render_mode = "rgb_array"
        print("🖥️  Rendering mode: RGB_ARRAY (no window)")

    # Create synchronized environments
    env = SynchronizedTetrisEnvironments(
        render_mode=render_mode,
        enable_rendering=args.render
    )

    print(f"Environment: {ENV_NAME}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Initialize agent
    agent = Agent(
        obs_space=env.observation_space,
        action_space=env.action_space,
        model_type=args.model_type
    )

    # Load model
    if args.model_path:
        model_path = args.model_path
    else:
        model_path = os.path.join(MODEL_DIR, 'latest_checkpoint.pth')

    print(f"\n🔄 Loading model from: {model_path}")

    if os.path.exists(model_path):
        if args.model_path and args.model_path.endswith('.pth') and 'checkpoint' not in args.model_path:
            # Custom path - load just the model weights
            agent.q_network.load_state_dict(torch.load(
                model_path, map_location=agent.device, weights_only=True))
            print("✅ Model weights loaded successfully")
        else:
            # Checkpoint path - load full checkpoint
            success = agent.load_checkpoint(path=model_path)
            if not success:
                print("❌ Failed to load checkpoint!")
                return
            print("✅ Checkpoint loaded successfully")
    else:
        print(f"❌ Model file not found: {model_path}")

        # Try to find best model
        best_model_path = os.path.join(MODEL_DIR, 'best_model.pth')
        if os.path.exists(best_model_path):
            print(f"🔄 Loading best model instead: {best_model_path}")
            agent.q_network.load_state_dict(torch.load(
                best_model_path, map_location=agent.device, weights_only=True))
            print("✅ Best model loaded successfully")
        else:
            print("❌ No trained model found!")
            print("\nTip: Train a model first:")
            print("  python train.py --episodes 100")
            return

    # Print agent info
    print(f"\n🤖 Agent loaded:")
    print(f"   Model type: {args.model_type}")
    print(f"   Device: {agent.device}")
    if hasattr(agent, 'epsilon'):
        print(f"   Epsilon: {agent.epsilon:.4f}")

    # Run evaluation
    try:
        if args.render:
            print(f"\n🎮 Starting evaluation with visual rendering...")
            print(f"   Look for the Tetris game window!")
            if args.slow:
                print(f"   Using slow mode for better visibility")

        episode_rewards, episode_steps, episode_times, game_info = evaluate_model(
            agent, env, args)

    except KeyboardInterrupt:
        print("\n⏹️  Evaluation interrupted by user")
        return
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    finally:
        # Always clean up
        env.close()

    # Print statistics
    print_statistics(episode_rewards, episode_steps, episode_times)

    # Save results
    save_results(episode_rewards, episode_steps,
                 episode_times, game_info, args)

    print("\n🎉 Evaluation completed successfully!")

    if args.render:
        print("\n💡 Rendering tips:")
        print("   • Use --slow for slower, more visible gameplay")
        print("   • Use --detailed for step-by-step progress")
        print("   • Remove --render for faster, text-only evaluation")


if __name__ == "__main__":
    main()
