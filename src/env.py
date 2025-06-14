import gym
import gym_tetris
from gym_tetris.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
import numpy as np
from gym import spaces
import cv2


class TetrisPreprocessor(gym.ObservationWrapper):
    """
    Preprocesses Tetris observations:
    - Converts to grayscale
    - Resizes to smaller resolution
    - Normalizes pixel values
    """

    def __init__(self, env, width=84, height=84):
        super().__init__(env)
        self.width = width
        self.height = height
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(height, width, 1), dtype=np.uint8
        )

    def observation(self, observation):
        # Convert RGB to grayscale
        gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        # Resize
        resized = cv2.resize(gray, (self.width, self.height),
                             interpolation=cv2.INTER_AREA)
        # Add channel dimension
        return np.expand_dims(resized, axis=-1)


class TetrisFrameStack(gym.Wrapper):
    """
    Stack the last N frames to give the agent temporal information
    """

    def __init__(self, env, n_frames=4):
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = []

        # Update observation space
        low = env.observation_space.low
        high = env.observation_space.high
        low = np.repeat(low, n_frames, axis=-1)
        high = np.repeat(high, n_frames, axis=-1)
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=env.observation_space.dtype)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if isinstance(obs, tuple):
            obs = obs[0]

        # Initialize frame stack with the first observation
        self.frames = [obs] * self.n_frames
        return self._get_observation()

    def step(self, action):
        result = self.env.step(action)

        if len(result) == 4:
            obs, reward, done, info = result
            truncated = False
        else:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated

        self.frames.append(obs)
        self.frames = self.frames[-self.n_frames:]  # Keep only last n_frames

        stacked_obs = self._get_observation()

        if len(result) == 4:
            return stacked_obs, reward, done, info
        else:
            return stacked_obs, reward, terminated, truncated, info

    def _get_observation(self):
        return np.concatenate(self.frames, axis=-1)


class TetrisRewardShaper(gym.RewardWrapper):
    """
    Shapes rewards to encourage better Tetris play
    """

    def __init__(self, env):
        super().__init__(env)
        self.prev_score = 0
        self.prev_lines = 0
        self.prev_level = 0

    def reset(self, **kwargs):
        self.prev_score = 0
        self.prev_lines = 0
        self.prev_level = 0
        return self.env.reset(**kwargs)

    def reward(self, reward):
        # Get current game statistics
        info = getattr(self.env, '_last_info', {})

        if 'statistics' in info:
            stats = info['statistics']
            current_score = stats.get('score', 0)
            current_lines = stats.get('lines', 0)
            current_level = stats.get('level', 0)

            # Reward for score increase
            score_reward = (current_score - self.prev_score) * 0.01

            # Large reward for clearing lines
            lines_reward = (current_lines - self.prev_lines) * 100

            # Reward for level progression
            level_reward = (current_level - self.prev_level) * 50

            # Small negative reward each step to encourage efficiency
            time_penalty = -0.1

            # Update previous values
            self.prev_score = current_score
            self.prev_lines = current_lines
            self.prev_level = current_level

            # Combine rewards
            shaped_reward = score_reward + lines_reward + level_reward + time_penalty

            return shaped_reward

        return reward

    def step(self, action):
        result = self.env.step(action)

        # Store info for reward shaping
        if len(result) >= 4:
            self.env._last_info = result[-1]

        return result


def make_env(name="TetrisA-v0",
             action_type="simple",
             preprocess=True,
             frame_stack=4,
             reward_shaping=True):
    """
    Create and configure a Tetris environment
    
    Args:
        name: Environment name (TetrisA-v0, TetrisA-v1, TetrisA-v2, TetrisA-v3)
        action_type: "simple" or "complex" action space
        preprocess: Whether to apply image preprocessing
        frame_stack: Number of frames to stack (0 to disable)
        reward_shaping: Whether to apply reward shaping
    """

    # Create base environment
    env = gym_tetris.make(name)

    # Choose action space
    if action_type == "simple":
        actions = SIMPLE_MOVEMENT
    elif action_type == "complex":
        actions = COMPLEX_MOVEMENT
    else:
        raise ValueError("action_type must be 'simple' or 'complex'")

    # Apply joypad wrapper
    env = JoypadSpace(env, actions)

    # Apply reward shaping
    if reward_shaping:
        env = TetrisRewardShaper(env)

    # Apply preprocessing
    if preprocess:
        env = TetrisPreprocessor(env)

    # Apply frame stacking
    if frame_stack > 0:
        env = TetrisFrameStack(env, n_frames=frame_stack)

    return env


def test_environment():
    """Test the environment creation and basic functionality"""
    print("Testing Tetris environment...")

    try:
        env = make_env()
        print(f"✅ Environment created successfully!")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")

        # Test reset
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        print(f"✅ Reset successful! Observation shape: {obs.shape}")

        # Test a few steps
        for i in range(3):
            action = env.action_space.sample()
            result = env.step(action)
            print(f"Step {i+1}: Action {action}, Result length: {len(result)}")

        env.close()
        print("✅ Environment test completed!")
        return True

    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_environment()
