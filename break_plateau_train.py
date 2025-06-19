#!/usr/bin/env python3
"""
Enhanced plateau breaker with configurable strategies, proper logging, and metrics collection
"""

import random
import numpy as np
import logging
import argparse
import json
import csv
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from tqdm import tqdm
import time

from src.agent import Agent
from config import make_env
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


@dataclass
class PlateauConfig:
    """Configuration for plateau breaking strategies"""
    # Training settings
    max_episodes: int = 500
    target_lines: int = 10

    # Action masking
    use_action_mask: bool = True
    action_mask_episodes: int = 100
    allowed_actions: List[int] = None  # Will default to [1,2,3,6]

    # Dense rewards
    dense_reward_frequency: int = 5
    piece_placement_bonus: float = 5.0
    bottom_row_bonus: float = 10.0
    first_line_bonus: float = 100.0
    line_clear_bonus: float = 50.0

    # Forced exploration
    exploration_boost_threshold: int = 50  # Episodes without lines before boost
    forced_exploration_rate: float = 0.3
    epsilon_boost: float = 0.5
    epsilon_decay_override: float = 0.999

    # Adaptive strategy
    adaptive_intervention: bool = True
    intervention_escalation_episodes: int = 100


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode"""
    episode: int
    reward: float
    steps: int
    lines_cleared: int
    epsilon: float
    action_mask_used: bool
    dense_rewards_used: bool
    forced_exploration_used: bool
    max_height: float = 0.0
    holes: int = 0


class PlateauBreaker:
    """Enhanced plateau breaking trainer with configurable strategies"""

    def __init__(self, config: PlateauConfig):
        self.config = config
        self.logger = self._setup_logging()

        # Initialize allowed actions if not specified
        if config.allowed_actions is None:
            # right, left, down, hard_drop
            config.allowed_actions = [1, 2, 3, 6]

        # Tracking variables
        self.total_lines_cleared = 0
        self.episodes_since_line_clear = 0
        self.episode_metrics: List[EpisodeMetrics] = []
        self.breakthrough_achieved = False

        # Strategy state
        self.current_dense_freq = config.dense_reward_frequency
        self.current_epsilon_boost = config.epsilon_boost

    def _setup_logging(self) -> logging.Logger:
        """Setup structured logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('plateau_breaker.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)

    def _should_use_action_mask(self, episode: int) -> bool:
        """Determine if action masking should be used"""
        return (self.config.use_action_mask and
                episode < self.config.action_mask_episodes)

    def _should_use_dense_rewards(self, episode: int) -> bool:
        """Determine if dense rewards should be used"""
        return episode % self.current_dense_freq == 0

    def _should_use_forced_exploration(self) -> bool:
        """Determine if forced exploration should be used"""
        return self.episodes_since_line_clear > self.config.exploration_boost_threshold

    def _apply_action_mask(self, action: int) -> int:
        """Apply action masking if the action is not allowed"""
        if action not in self.config.allowed_actions:
            return random.choice(self.config.allowed_actions)
        return action

    def _calculate_dense_rewards(self, info: Dict[str, Any], episode_steps: int, env) -> float:
        """Calculate additional dense rewards"""
        dense_reward = 0.0

        # Bonus for piece placement
        if 'piece_placed' in info or episode_steps % 10 == 0:
            dense_reward += self.config.piece_placement_bonus

        # Bonus for filling bottom rows
        try:
            if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'board'):
                board = env.unwrapped.board
                if board is not None and len(board.shape) == 2:
                    bottom_row_filled = np.sum(board[-1, :]) / board.shape[1]
                    if bottom_row_filled > 0.8:  # 80% filled
                        dense_reward += self.config.bottom_row_bonus
        except Exception as e:
            self.logger.debug(f"Could not calculate bottom row bonus: {e}")

        return dense_reward

    def _escalate_intervention(self, episode: int):
        """Escalate intervention strategies if no progress"""
        if (episode > 0 and episode % self.config.intervention_escalation_episodes == 0
                and self.total_lines_cleared == 0):

            self.logger.warning(
                f"No lines cleared after {episode} episodes. Escalating intervention...")

            # Increase dense reward frequency
            self.current_dense_freq = max(2, self.current_dense_freq - 1)

            # Increase exploration boost
            self.current_epsilon_boost = min(
                0.8, self.current_epsilon_boost + 0.1)

            self.logger.info(f"New dense frequency: {self.current_dense_freq}")
            self.logger.info(
                f"New epsilon boost: {self.current_epsilon_boost}")

    def train(self) -> Dict[str, Any]:
        """Main training loop with plateau breaking strategies"""
        self.logger.info("Starting Plateau Breaker Training")
        self.logger.info(f"Target: {self.config.target_lines} lines cleared")
        self.logger.info(f"Max episodes: {self.config.max_episodes}")

        # Set random seeds for reproducibility
        random.seed(42)
        np.random.seed(42)

        # Create environment and agent
        env = make_env(frame_stack=4)
        agent = Agent(
            obs_space=env.observation_space,
            action_space=env.action_space,
            reward_shaping="simple"
        )

        # Load existing checkpoint if available
        if agent.load_checkpoint(latest=True):
            self.logger.info("Loaded existing agent checkpoint")
        else:
            self.logger.info("Starting with fresh agent")

        # Override epsilon settings for exploration
        original_epsilon = agent.epsilon
        agent.epsilon = self.config.epsilon_boost
        agent.epsilon_decay = self.config.epsilon_decay_override

        self.logger.info(
            f"Epsilon boosted from {original_epsilon:.3f} to {agent.epsilon:.3f}")

        # Training loop with progress bar
        start_time = time.time()

        for episode in tqdm(range(self.config.max_episodes), desc="Training"):
            obs, info = env.reset()
            done = False
            episode_reward = 0
            episode_steps = 0
            lines_cleared_this_episode = 0

            # Determine strategies for this episode
            use_action_mask = self._should_use_action_mask(episode)
            use_dense_rewards = self._should_use_dense_rewards(episode)
            use_forced_exploration = self._should_use_forced_exploration()

            # Episode loop
            while not done:
                # Action selection with strategies
                if use_forced_exploration and random.random() < self.config.forced_exploration_rate:
                    # Force random exploration
                    action = (random.choice(self.config.allowed_actions)
                              if use_action_mask else env.action_space.sample())
                else:
                    action = agent.select_action(obs)
                    if use_action_mask:
                        action = self._apply_action_mask(action)

                # Environment step
                next_obs, reward, terminated, truncated, info = env.step(
                    action)
                done = terminated or truncated

                # Apply dense rewards if enabled
                if use_dense_rewards:
                    reward += self._calculate_dense_rewards(
                        info, episode_steps, env)

                # Check for line clears and apply bonuses
                lines_this_step = info.get('lines_cleared', 0)
                if lines_this_step > 0:
                    lines_cleared_this_episode += lines_this_step
                    self.total_lines_cleared += lines_this_step
                    self.episodes_since_line_clear = 0

                    # Apply line clear bonuses
                    if self.total_lines_cleared == lines_this_step:  # First ever line clear
                        reward += self.config.first_line_bonus
                        self.logger.info(
                            f"🎉 FIRST LINE CLEARED! Episode {episode}, Step {episode_steps}")
                    else:
                        reward += self.config.line_clear_bonus * lines_this_step

                # Store experience and learn
                agent.remember(obs, action, reward, next_obs, done, info)

                if episode_steps % 2 == 0 and len(agent.memory) >= agent.batch_size:
                    agent.learn()

                episode_reward += reward
                episode_steps += 1
                obs = next_obs

            # Episode completed
            self.episodes_since_line_clear += 1
            agent.end_episode(episode_reward, episode_steps,
                              lines_cleared_this_episode)

            # Record metrics
            metrics = EpisodeMetrics(
                episode=episode,
                reward=episode_reward,
                steps=episode_steps,
                lines_cleared=lines_cleared_this_episode,
                epsilon=agent.epsilon,
                action_mask_used=use_action_mask,
                dense_rewards_used=use_dense_rewards,
                forced_exploration_used=use_forced_exploration
            )
            self.episode_metrics.append(metrics)

            # Periodic reporting
            if episode % 10 == 0:
                self._log_progress(episode, metrics)

            # Check for breakthrough
            if self.total_lines_cleared >= self.config.target_lines:
                self.breakthrough_achieved = True
                self.logger.info(
                    f"✅ BREAKTHROUGH! Target of {self.config.target_lines} lines reached!")
                self.logger.info(f"Saving breakthrough model...")
                agent.save_checkpoint(episode, model_dir="models/")
                break

            # Escalate intervention if needed
            if self.config.adaptive_intervention:
                self._escalate_intervention(episode)

        # Training completed
        training_time = time.time() - start_time
        env.close()

        # Save final checkpoint
        agent.save_checkpoint(self.config.max_episodes, model_dir="models/")

        # Generate and save results
        results = self._generate_results(training_time)
        self._save_results(results)

        return results

    def _log_progress(self, episode: int, metrics: EpisodeMetrics):
        """Log training progress"""
        strategy_info = []
        if metrics.action_mask_used:
            strategy_info.append("ActionMask")
        if metrics.dense_rewards_used:
            strategy_info.append("DenseReward")
        if metrics.forced_exploration_used:
            strategy_info.append("ForcedExplore")

        strategy_str = f" [{','.join(strategy_info)}]" if strategy_info else ""

        self.logger.info(
            f"Episode {episode:3d} | "
            f"Reward: {metrics.reward:6.1f} | "
            f"Steps: {metrics.steps:3d} | "
            f"Lines: {metrics.lines_cleared} (Total: {self.total_lines_cleared}) | "
            f"Eps: {metrics.epsilon:.3f}"
            f"{strategy_str}"
        )

    def _generate_results(self, training_time: float) -> Dict[str, Any]:
        """Generate comprehensive results summary"""
        return {
            "config": asdict(self.config),
            "breakthrough_achieved": self.breakthrough_achieved,
            "total_lines_cleared": self.total_lines_cleared,
            "total_episodes": len(self.episode_metrics),
            "training_time_seconds": training_time,
            "final_epsilon": self.episode_metrics[-1].epsilon if self.episode_metrics else 0,
            "avg_reward_last_50": np.mean([m.reward for m in self.episode_metrics[-50:]]) if len(self.episode_metrics) >= 50 else 0,
            "avg_steps_per_episode": np.mean([m.steps for m in self.episode_metrics]) if self.episode_metrics else 0,
            "episodes_with_lines": sum(1 for m in self.episode_metrics if m.lines_cleared > 0),
            "strategy_usage": {
                "action_mask_episodes": sum(1 for m in self.episode_metrics if m.action_mask_used),
                "dense_reward_episodes": sum(1 for m in self.episode_metrics if m.dense_rewards_used),
                "forced_exploration_episodes": sum(1 for m in self.episode_metrics if m.forced_exploration_used)
            }
        }

    def _save_results(self, results: Dict[str, Any]):
        """Save results to files"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Save summary JSON
        with open(f"plateau_breaker_results_{timestamp}.json", "w") as f:
            json.dump(results, f, indent=2)

        # Save detailed episode metrics CSV
        if self.episode_metrics:
            with open(f"plateau_breaker_episodes_{timestamp}.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=asdict(
                    self.episode_metrics[0]).keys())
                writer.writeheader()
                for metrics in self.episode_metrics:
                    writer.writerow(asdict(metrics))

        self.logger.info(f"Results saved with timestamp {timestamp}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Enhanced Tetris Plateau Breaker")

    parser.add_argument("--episodes", type=int, default=500,
                        help="Maximum episodes to train")
    parser.add_argument("--target-lines", type=int, default=10,
                        help="Target number of lines to clear")
    parser.add_argument("--no-action-mask", action="store_true",
                        help="Disable action masking strategy")
    parser.add_argument("--dense-freq", type=int, default=5,
                        help="Dense reward frequency")
    parser.add_argument("--epsilon-boost", type=float, default=0.5,
                        help="Epsilon boost for exploration")
    parser.add_argument("--config-file", type=str,
                        help="JSON config file to load settings from")

    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()

    # Create config
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file, 'r') as f:
            config_dict = json.load(f)
        config = PlateauConfig(**config_dict)
    else:
        config = PlateauConfig(
            max_episodes=args.episodes,
            target_lines=args.target_lines,
            use_action_mask=not args.no_action_mask,
            dense_reward_frequency=args.dense_freq,
            epsilon_boost=args.epsilon_boost
        )

    # Run plateau breaker
    breaker = PlateauBreaker(config)
    results = breaker.train()

    # Print final summary
    print("\n" + "="*60)
    print("PLATEAU BREAKER RESULTS")
    print("="*60)
    print(f"Breakthrough achieved: {results['breakthrough_achieved']}")
    print(f"Total lines cleared: {results['total_lines_cleared']}")
    print(f"Episodes completed: {results['total_episodes']}")
    print(f"Training time: {results['training_time_seconds']:.1f} seconds")

    if results['breakthrough_achieved']:
        print("\n✅ SUCCESS! Continue training with:")
        print("python train.py --episodes 5000 --resume --reward_shaping simple")
    else:
        print("\n❌ Plateau not broken. Consider:")
        print("1. Adjusting hyperparameters")
        print("2. Using curriculum learning")
        print("3. Implementing imitation learning")


if __name__ == "__main__":
    main()
