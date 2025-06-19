import gymnasium as gym
import numpy as np


class TetrisCurriculumWrapper(gym.Wrapper):
    """
    Curriculum wrapper that starts with easier scenarios to help agent discover line clearing
    """

    def __init__(self, env, stage=0):
        super().__init__(env)
        self.stage = stage
        self.episode_count = 0
        self.lines_cleared_total = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Stage 0: Start with partially filled bottom rows to make line clearing easier
        if self.stage == 0 and self.episode_count % 3 == 0:  # Every 3rd episode
            # Pre-fill bottom rows with gaps for easy line clears
            if hasattr(self.env, 'unwrapped') and hasattr(self.env.unwrapped, 'board'):
                board = self.env.unwrapped.board
                if board is not None and len(board.shape) == 2:
                    # Fill bottom 2 rows except for 1-2 columns (for dropping pieces)
                    rows_to_fill = 2
                    for row_idx in range(-rows_to_fill, 0):
                        for col_idx in range(board.shape[1]):
                            if col_idx not in [4, 5]:  # Leave middle columns empty
                                board[row_idx, col_idx] = 1

        self.episode_count += 1
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Track lines cleared
        lines = info.get('lines_cleared', 0)
        if lines > 0:
            self.lines_cleared_total += lines

            # Bonus rewards in early stages to reinforce line clearing
            if self.stage == 0:
                reward += lines * 20  # Extra bonus for any line clear

        return obs, reward, terminated, truncated, info

    def advance_stage(self):
        """Advance to next curriculum stage"""
        self.stage += 1
        print(f"Advanced to curriculum stage {self.stage}")


class ExplorationBoostWrapper(gym.Wrapper):
    """
    Temporarily boost exploration when stuck in plateau
    """

    def __init__(self, env, boost_episodes=100):
        super().__init__(env)
        self.boost_episodes = boost_episodes
        self.episode_count = 0
        self.original_epsilon = None

    def reset(self, **kwargs):
        self.episode_count += 1
        return self.env.reset(**kwargs)

    def should_boost(self, agent):
        """Check if we should boost exploration"""
        if self.episode_count < self.boost_episodes:
            if self.original_epsilon is None:
                self.original_epsilon = agent.epsilon
            # Set higher exploration rate
            agent.epsilon = max(0.3, agent.epsilon)
            return True
        else:
            # Restore original epsilon
            if self.original_epsilon is not None:
                agent.epsilon = self.original_epsilon
            return False


def create_curriculum_env(base_env_name="TetrisManual-v0", stage=0):
    """Create environment with curriculum wrapper"""
    from config import make_env

    # Create base environment
    env = make_env(base_env_name, frame_stack=4)

    # Add curriculum wrapper
    env = TetrisCurriculumWrapper(env, stage=stage)

    # Add exploration boost wrapper
    env = ExplorationBoostWrapper(env)

    return env
