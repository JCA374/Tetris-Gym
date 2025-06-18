# src/reward_shaping.py
"""
Production-ready Tetris reward shaping with careful weight balancing,
computational optimization, and comprehensive logging.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import time


@dataclass
class RewardMetrics:
    """Container for all reward shaping metrics"""
    # Core metrics
    max_height: float = 0.0
    avg_height: float = 0.0
    height_variance: float = 0.0
    holes: int = 0
    bumpiness: float = 0.0
    wells: float = 0.0

    # Strategic metrics
    lines_cleared: int = 0
    tetris_bonus: bool = False
    efficiency: float = 0.0
    accessibility: float = 0.0

    # Survival metrics
    step_reward: float = 0.0
    game_over_penalty: float = 0.0

    # Computational metrics
    computation_time: float = 0.0


class OptimizedBoardAnalyzer:
    """Efficient board analysis with caching and optimizations"""

    def __init__(self, board_height: int = 20, board_width: int = 10):
        self.board_height = board_height
        self.board_width = board_width

        # Pre-allocate arrays for efficiency
        self._heights = np.zeros(board_width, dtype=np.int32)
        self._holes = np.zeros(board_width, dtype=np.int32)

    def analyze_board(self, board: np.ndarray) -> Dict[str, float]:
        """Efficiently compute all board metrics"""
        start_time = time.perf_counter()

        # Reset arrays
        self._heights.fill(0)
        self._holes.fill(0)

        # Single pass to compute heights and holes
        for col in range(self.board_width):
            found_filled = False
            for row in range(self.board_height):
                if board[row, col] != 0:
                    if not found_filled:
                        self._heights[col] = self.board_height - row
                        found_filled = True
                elif found_filled:
                    self._holes[col] += 1

        # Normalize metrics
        max_height = float(np.max(self._heights)) / self.board_height
        avg_height = float(np.mean(self._heights)) / self.board_height
        height_variance = float(np.var(self._heights)) / \
            (self.board_height ** 2)
        total_holes = int(np.sum(self._holes))

        # Bumpiness (height differences between adjacent columns)
        bumpiness = 0.0
        for i in range(self.board_width - 1):
            bumpiness += abs(self._heights[i] - self._heights[i + 1])
        bumpiness = bumpiness / (self.board_width * self.board_height)

        # Wells (deep single-column gaps)
        wells = 0.0
        for i in range(self.board_width):
            left_height = self._heights[i - 1] if i > 0 else self._heights[i]
            right_height = self._heights[i +
                                         1] if i < self.board_width - 1 else self._heights[i]
            well_depth = min(left_height, right_height) - self._heights[i]
            wells += max(0, well_depth)
        wells = wells / (self.board_width * self.board_height)

        computation_time = time.perf_counter() - start_time

        return {
            'max_height': max_height,
            'avg_height': avg_height,
            'height_variance': height_variance,
            'holes': total_holes,
            'bumpiness': bumpiness,
            'wells': wells,
            'computation_time': computation_time
        }


class AdaptiveRewardWeights:
    """Dynamic weight adjustment based on training progress"""

    def __init__(self, initial_weights: Dict[str, float]):
        self.initial_weights = initial_weights.copy()
        self.current_weights = initial_weights.copy()
        self.episode_count = 0

        # Annealing schedules
        self.annealing_schedules = {
            'step_reward': {'start': 1.0, 'end': 0.1, 'episodes': 1000},
            'game_over_penalty': {'start': 1.0, 'end': 0.5, 'episodes': 500},
            # Increase hole penalty over time
            'holes': {'start': 0.5, 'end': 2.0, 'episodes': 800},
        }

    def update_weights(self, episode: int, performance_metrics: Dict[str, float]):
        """Update weights based on training progress and performance"""
        self.episode_count = episode

        for metric, schedule in self.annealing_schedules.items():
            if metric in self.current_weights:
                progress = min(1.0, episode / schedule['episodes'])
                # Linear interpolation
                new_value = schedule['start'] + progress * \
                    (schedule['end'] - schedule['start'])
                self.current_weights[metric] = self.initial_weights[metric] * new_value

        # Performance-based adjustments
        avg_lines = performance_metrics.get('avg_lines_per_episode', 0)
        if avg_lines < 1:  # Struggling to clear lines
            # Reduce hole penalty temporarily
            self.current_weights['holes'] *= 0.9
            # Increase height penalty
            self.current_weights['max_height'] *= 1.1
        elif avg_lines > 10:  # Performing well
            # Reward efficiency more
            self.current_weights['efficiency'] *= 1.05

    def get_weights(self) -> Dict[str, float]:
        return self.current_weights.copy()


class TetrisRewardShaper:
    """
    Production-ready reward shaping with careful weight balancing,
    computational optimization, and comprehensive logging.
    """

    def __init__(self,
                 board_height: int = 20,
                 board_width: int = 10,
                 enable_logging: bool = True,
                 log_frequency: int = 100):

        self.board_analyzer = OptimizedBoardAnalyzer(board_height, board_width)

        # Carefully tuned initial weights (normalized to prevent dominance)
        initial_weights = {
            # Primary objectives (higher weights)
            'lines_cleared': 20.0,
            'tetris_bonus': 15.0,
            'game_over_penalty': -50.0,

            # Board health (moderate weights, normalized)
            'max_height': -1.0,
            'holes': -2.0,
            'bumpiness': -0.5,
            'wells': -1.0,
            'height_variance': -0.3,

            # Strategic bonuses (low weights)
            'efficiency': 0.5,
            'accessibility': 0.2,
            'step_reward': 0.01,
        }

        self.adaptive_weights = AdaptiveRewardWeights(initial_weights)

        # State tracking
        self.previous_board = None
        self.previous_holes = 0
        self.episode_metrics = []
        self.total_lines_cleared = 0
        self.pieces_placed = 0

        # Performance tracking for ablation studies
        self.performance_history = deque(maxlen=100)
        self.metric_history = {metric: deque(
            maxlen=1000) for metric in initial_weights.keys()}

        # Logging setup
        self.enable_logging = enable_logging
        self.log_frequency = log_frequency
        if enable_logging:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)

    def calculate_reward(self,
                         obs: Any,
                         action: int,
                         base_reward: float,
                         done: bool,
                         info: Dict[str, Any]) -> Tuple[float, RewardMetrics]:
        """
        Calculate shaped reward with detailed metrics tracking
        
        Returns:
            Tuple of (shaped_reward, metrics_object)
        """
        metrics = RewardMetrics()
        shaped_reward = base_reward

        # Extract board state
        board = self._extract_board(obs)
        if board is None:
            return shaped_reward, metrics

        # Get current weights
        weights = self.adaptive_weights.get_weights()

        # Analyze current board state
        board_metrics = self.board_analyzer.analyze_board(board)

        # Update metrics object
        metrics.max_height = board_metrics['max_height']
        metrics.avg_height = board_metrics['avg_height']
        metrics.height_variance = board_metrics['height_variance']
        metrics.holes = board_metrics['holes']
        metrics.bumpiness = board_metrics['bumpiness']
        metrics.wells = board_metrics['wells']
        metrics.computation_time = board_metrics['computation_time']

        # Apply board health penalties
        shaped_reward += metrics.max_height * weights['max_height']
        shaped_reward += metrics.holes * weights['holes']
        shaped_reward += metrics.bumpiness * weights['bumpiness']
        shaped_reward += metrics.wells * weights['wells']
        shaped_reward += metrics.height_variance * weights['height_variance']

        # Line clearing bonuses
        lines_cleared = info.get('lines_cleared', 0)
        if lines_cleared > 0:
            self.total_lines_cleared += lines_cleared
            self.pieces_placed += 1

            metrics.lines_cleared = lines_cleared
            shaped_reward += lines_cleared * weights['lines_cleared']

            # Tetris bonus (4 lines)
            if lines_cleared == 4:
                metrics.tetris_bonus = True
                shaped_reward += weights['tetris_bonus']

            # Efficiency bonus
            if self.pieces_placed > 0:
                efficiency = self.total_lines_cleared / self.pieces_placed
                metrics.efficiency = efficiency
                shaped_reward += efficiency * weights['efficiency']

        # Survival/death rewards
        if done:
            metrics.game_over_penalty = weights['game_over_penalty']
            shaped_reward += weights['game_over_penalty']
        else:
            metrics.step_reward = weights['step_reward']
            shaped_reward += weights['step_reward']

        # Prevent reward hacking (agent standing still)
        if hasattr(self, '_last_action') and action == self._last_action:
            self._repeated_actions = getattr(self, '_repeated_actions', 0) + 1
            if self._repeated_actions > 10:  # Penalize excessive repetition
                shaped_reward -= 0.1
        else:
            self._repeated_actions = 0
        self._last_action = action

        # Update history for analysis
        self._update_metric_history(metrics, weights)

        # Log periodically
        if (self.enable_logging and
                len(self.episode_metrics) % self.log_frequency == 0):
            self._log_metrics()

        self.previous_board = board.copy()
        self.previous_holes = metrics.holes

        return shaped_reward, metrics

    def episode_end(self, episode_reward: float, episode_length: int, lines_cleared: int):
        """Called at end of episode for tracking and weight updates"""

        episode_data = {
            'reward': episode_reward,
            'length': episode_length,
            'lines_cleared': lines_cleared,
            'efficiency': lines_cleared / max(1, episode_length),
            'avg_lines_per_episode': lines_cleared
        }

        self.performance_history.append(episode_data)

        # Update adaptive weights
        if len(self.performance_history) >= 10:
            recent_performance = {
                'avg_lines_per_episode': np.mean([ep['lines_cleared']
                                                 for ep in list(self.performance_history)[-10:]])
            }
            episode_num = len(self.performance_history)
            self.adaptive_weights.update_weights(
                episode_num, recent_performance)

    def reset_episode(self):
        """Reset episode-specific tracking"""
        self.previous_board = None
        self.previous_holes = 0
        self.total_lines_cleared = 0
        self.pieces_placed = 0
        self._repeated_actions = 0

    def get_ablation_data(self) -> Dict[str, Any]:
        """Get data for ablation studies"""
        if not self.performance_history:
            return {}

        recent_episodes = list(self.performance_history)[-50:]

        return {
            'avg_reward': np.mean([ep['reward'] for ep in recent_episodes]),
            'avg_lines': np.mean([ep['lines_cleared'] for ep in recent_episodes]),
            'avg_efficiency': np.mean([ep['efficiency'] for ep in recent_episodes]),
            'current_weights': self.adaptive_weights.get_weights(),
            'metric_correlations': self._calculate_metric_correlations()
        }

    def _extract_board(self, obs: Any) -> Optional[np.ndarray]:
        """Extract board from various observation formats"""
        if isinstance(obs, dict):
            if 'board' in obs:
                return obs['board']
            elif 'observation' in obs:
                return obs['observation']
        elif isinstance(obs, np.ndarray):
            if len(obs.shape) == 2:  # Direct board
                return obs
            elif len(obs.shape) == 1:  # Flattened - try to reconstruct
                # This is environment-specific, may need adjustment
                if len(obs) >= 200:  # Likely contains board data
                    return obs[:200].reshape(20, 10)

        return None

    def _update_metric_history(self, metrics: RewardMetrics, weights: Dict[str, float]):
        """Update metric history for analysis"""
        for metric_name in self.metric_history.keys():
            value = getattr(metrics, metric_name, 0)
            if isinstance(value, bool):
                value = float(value)
            self.metric_history[metric_name].append(value)

    def _calculate_metric_correlations(self) -> Dict[str, float]:
        """Calculate correlations between metrics and performance"""
        if len(self.performance_history) < 20:
            return {}

        recent_rewards = [ep['reward']
                          for ep in list(self.performance_history)[-20:]]
        correlations = {}

        for metric_name, history in self.metric_history.items():
            if len(history) >= 20:
                recent_values = list(history)[-20:]
                if len(recent_values) == len(recent_rewards):
                    corr = np.corrcoef(recent_values, recent_rewards)[0, 1]
                    correlations[metric_name] = float(
                        corr) if not np.isnan(corr) else 0.0

        return correlations

    def _log_metrics(self):
        """Log current metrics and performance"""
        if not self.performance_history:
            return

        recent_performance = list(self.performance_history)[-10:]
        avg_reward = np.mean([ep['reward'] for ep in recent_performance])
        avg_lines = np.mean([ep['lines_cleared'] for ep in recent_performance])

        current_weights = self.adaptive_weights.get_weights()

        self.logger.info(
            f"Reward Shaping Metrics (Episodes {len(self.performance_history)-10}-{len(self.performance_history)}):")
        self.logger.info(f"  Avg Reward: {avg_reward:.2f}")
        self.logger.info(f"  Avg Lines Cleared: {avg_lines:.2f}")
        self.logger.info(f"  Current Weights: {current_weights}")

        # Log metric correlations
        correlations = self._calculate_metric_correlations()
        if correlations:
            self.logger.info(f"  Metric-Reward Correlations: {correlations}")


class SimplifiedRewardShaper:
    """Simplified version for initial testing and ablation studies"""

    def __init__(self, height_weight: float = -0.5, hole_weight: float = -2.0):
        self.height_weight = height_weight
        self.hole_weight = hole_weight
        self.board_analyzer = OptimizedBoardAnalyzer()

    def calculate_reward(self, obs, action, base_reward, done, info):
        """Simple height + hole penalty shaping"""
        board = self._extract_board(obs)
        if board is None:
            return base_reward

        board_metrics = self.board_analyzer.analyze_board(board)

        shaped_reward = base_reward
        shaped_reward += board_metrics['max_height'] * self.height_weight
        shaped_reward += board_metrics['holes'] * self.hole_weight

        if done:
            shaped_reward -= 20  # Game over penalty
        else:
            shaped_reward += 0.01  # Survival bonus

        return shaped_reward

    def _extract_board(self, obs):
        """Extract board from observation"""
        if isinstance(obs, dict) and 'board' in obs:
            return obs['board']
        return None

    def reset_episode(self):
        pass

    def episode_end(self, *args):
        pass


# Factory function for easy integration
def create_reward_shaper(shaper_type: str = "full", **kwargs):
    """Factory function to create reward shapers"""
    if shaper_type == "simple":
        return SimplifiedRewardShaper(**kwargs)
    elif shaper_type == "full":
        return TetrisRewardShaper(**kwargs)
    else:
        raise ValueError(f"Unknown shaper type: {shaper_type}")


# Example usage and testing
if __name__ == "__main__":
    # Test the reward shaper
    shaper = TetrisRewardShaper(enable_logging=True)

    # Simulate some episodes
    dummy_board = np.random.randint(0, 2, (20, 10))
    dummy_obs = {'board': dummy_board}

    for episode in range(5):
        shaper.reset_episode()

        for step in range(100):
            shaped_reward, metrics = shaper.calculate_reward(
                dummy_obs, 0, 1.0, False, {'lines_cleared': 0}
            )

        shaper.episode_end(100, 100, 5)

    # Print ablation data
    ablation_data = shaper.get_ablation_data()
    print("Ablation Data:", ablation_data)
