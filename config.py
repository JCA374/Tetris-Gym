import gymnasium as gym
import numpy as np
from gymnasium import spaces
import cv2


class TetrisPreprocessor(gym.ObservationWrapper):
    """
    Preprocesses Atari Tetris observations:
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
        self.game_over_penalty = -100

    def reset(self, **kwargs):
        self.prev_score = 0
        return self.env.reset(**kwargs)

    def reward(self, reward):
        # Original Atari reward is usually the score difference
        # We can enhance this with additional shaping

        # Small time penalty to encourage faster play
        time_penalty = -0.01

        # Bonus for score increases
        score_bonus = reward * 0.1 if reward > 0 else 0

        # Check if game is over (lives decreased significantly)
        if hasattr(self.env.unwrapped, 'ale'):
            lives = self.env.unwrapped.ale.lives()
            if lives == 0:
                return reward + self.game_over_penalty

        return reward + time_penalty + score_bonus

    def step(self, action):
        result = self.env.step(action)

        # Store info for reward shaping
        if len(result) >= 4:
            self.env._last_info = result[-1]

        return result


def make_env(name="ALE/Tetris-v5",
             render_mode="rgb_array",
             preprocess=True,
             frame_stack=4,
             reward_shaping=True):
    """
    Create and configure an Atari Tetris environment
    
    Args:
        name: Environment name (ALE/Tetris-v5 is recommended)
        render_mode: "human", "rgb_array", or None
        preprocess: Whether to apply image preprocessing
        frame_stack: Number of frames to stack (0 to disable)
        reward_shaping: Whether to apply reward shaping
    """

    # Create base Atari environment
    env = gym.make(name, render_mode=render_mode)

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
    print("Testing Atari Tetris environment...")

    try:
        env = make_env(render_mode="rgb_array")
        print(f"✅ Environment created successfully!")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        print(f"Action meanings: {env.unwrapped.get_action_meanings()}")

        # Test reset
        result = env.reset()
        if isinstance(result, tuple):
            obs = result[0]
        else:
            obs = result
        print(f"✅ Reset successful! Observation shape: {obs.shape}")

        # Test a few steps
        for i in range(3):
            action = env.action_space.sample()
            result = env.step(action)
            action_name = env.unwrapped.get_action_meanings()[action]
            print(
                f"Step {i+1}: Action {action} ({action_name}), Result length: {len(result)}")

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
