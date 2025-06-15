# config.py - Fixed configuration for Tetris Gymnasium AI

import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np
import cv2
from gymnasium.spaces import Box

# Register the Tetris environment manually (since auto-registration doesn't work)
try:
    register(
        id="TetrisManual-v0",
        entry_point="tetris_gymnasium.envs.tetris:Tetris",
    )
    print("✅ Tetris environment registered successfully")
except gym.error.Error:
    # Already registered
    pass

# Environment settings
ENV_NAME = "TetrisManual-v0"
RENDER_MODE = None  # "human" for visualization, None for training
FRAME_STACK = 4  # Number of frames to stack
PREPROCESS = True  # Apply image preprocessing
REWARD_SHAPING = True  # Apply custom reward shaping

# Training hyperparameters
LR = 1e-4  # Learning rate
GAMMA = 0.99  # Discount factor
BATCH_SIZE = 32  # Batch size for training
REPLAY_BUFFER_SIZE = 100000  # Experience replay buffer size
MIN_REPLAY_SIZE = 10000  # Minimum buffer size before training starts
EPSILON_START = 1.0  # Initial exploration rate
EPSILON_END = 0.1  # Final exploration rate
EPSILON_DECAY = 100000  # Decay steps for epsilon
TARGET_UPDATE_FREQ = 1000  # Steps between target network updates

# Training settings
MAX_EPISODES = 1000  # Maximum number of episodes
MAX_STEPS_PER_EPISODE = 10000  # Maximum steps per episode
SAVE_FREQUENCY = 50  # Save model every N episodes
LOG_FREQUENCY = 10  # Log metrics every N episodes

# Model architecture
CONV_CHANNELS = [32, 64, 64]  # Convolutional layer channels
CONV_KERNEL_SIZES = [8, 4, 3]  # Kernel sizes for conv layers
CONV_STRIDES = [4, 2, 1]  # Strides for conv layers
HIDDEN_SIZE = 512  # Hidden layer size

# Directories
MODEL_DIR = "models/"
LOG_DIR = "logs/"
CHECKPOINT_DIR = "checkpoints/"

# Device settings
DEVICE = "cuda"  # "cuda" or "cpu"

# Random seed for reproducibility
SEED = 42


class TetrisObservationWrapper(gym.ObservationWrapper):
    """
    Wrapper to convert Tetris Gymnasium dict observations to arrays
    """

    def __init__(self, env, use_rgb_rendering=False):
        super().__init__(env)
        self.use_rgb_rendering = use_rgb_rendering

        if use_rgb_rendering:
            # Use RGB rendering instead of dict observations
            self.observation_space = Box(
                low=0, high=255, shape=(240, 320, 3), dtype=np.uint8
            )
        else:
            # Calculate total size of flattened dict observations
            # Get a sample to understand the structure
            sample_obs = env.observation_space.sample()
            print(f"Sample observation keys: {list(sample_obs.keys())}")

            total_size = 0
            self.obs_info = {}

            for key, value in sample_obs.items():
                if isinstance(value, np.ndarray):
                    size = np.prod(value.shape)
                    total_size += size
                    self.obs_info[key] = {'shape': value.shape, 'size': size}
                    print(f"  {key}: shape={value.shape}, size={size}")
                else:
                    # Handle scalar values
                    total_size += 1
                    self.obs_info[key] = {'shape': (), 'size': 1}
                    print(f"  {key}: scalar value")

            print(f"Total flattened size: {total_size}")

            self.observation_space = Box(
                low=0, high=1, shape=(total_size,), dtype=np.float32
            )

        print(f"TetrisObservationWrapper: {self.observation_space}")

    def observation(self, obs):
        if self.use_rgb_rendering:
            # Get RGB array from environment
            rgb_array = self.env.render()
            if rgb_array is not None:
                return rgb_array
            else:
                # Fallback to zeros if render fails
                return np.zeros(self.observation_space.shape, dtype=np.uint8)
        else:
            # Flatten all dict components
            flattened = []

            for key in sorted(obs.keys()):  # Sort for consistency
                value = obs[key]
                if isinstance(value, np.ndarray):
                    # Normalize arrays to [0, 1] range
                    if value.dtype in [np.uint8, np.int32, np.int64]:
                        value = value.astype(np.float32)
                        if value.max() > 1:
                            value = value / 255.0 if value.max() <= 255 else value / value.max()
                    flattened.append(value.flatten())
                else:
                    # Handle scalar values
                    scalar_val = float(value)
                    if scalar_val > 1:
                        scalar_val = scalar_val / 255.0 if scalar_val <= 255 else scalar_val / 1000.0
                    flattened.append(np.array([scalar_val]))

            result = np.concatenate(flattened).astype(np.float32)
            return result


class TetrisPreprocessWrapper(gym.ObservationWrapper):
    """
    Preprocessing wrapper for Tetris observations
    """

    def __init__(self, env, target_size=(84, 84), grayscale=True):
        super().__init__(env)
        self.target_size = target_size
        self.grayscale = grayscale

        if grayscale:
            shape = (*target_size, 1)
        else:
            shape = (*target_size, 3)

        self.observation_space = Box(
            low=0, high=1, shape=shape, dtype=np.float32
        )

    def observation(self, obs):
        # Handle different observation types
        if len(obs.shape) == 3:  # RGB image
            resized = cv2.resize(obs, self.target_size)
            if self.grayscale:
                resized = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
                resized = np.expand_dims(resized, axis=-1)
        elif len(obs.shape) == 1:  # Flattened features
            # For feature vectors, create a square 2D representation
            obs_len = len(obs)

            # Find the best square size
            side_length = int(np.ceil(np.sqrt(obs_len)))
            target_len = side_length * side_length

            # Pad to make it square (with zeros, not negative padding)
            if obs_len < target_len:
                padding_needed = target_len - obs_len
                obs_padded = np.pad(obs, (0, padding_needed),
                                    mode='constant', constant_values=0)
            else:
                obs_padded = obs[:target_len]  # Truncate if too long

            # Reshape to 2D
            obs_2d = obs_padded.reshape(side_length, side_length)

            # Ensure values are in proper range for cv2
            obs_2d = np.clip(obs_2d * 255, 0, 255).astype(np.uint8)

            # Resize to target size
            resized = cv2.resize(obs_2d, self.target_size)

            # Convert back to float and normalize
            resized = resized.astype(np.float32) / 255.0

            if not self.grayscale:
                resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
            else:
                resized = np.expand_dims(resized, axis=-1)
        else:
            # For other shapes, just return as is
            resized = obs

        return resized


class FrameStackWrapper(gym.ObservationWrapper):
    """
    Stack multiple frames together
    """

    def __init__(self, env, num_frames=4):
        super().__init__(env)
        self.num_frames = num_frames
        self.frames = []

        # Update observation space
        old_space = env.observation_space
        new_shape = (*old_space.shape[:-1], old_space.shape[-1] * num_frames)
        self.observation_space = Box(
            low=old_space.low.min(),
            high=old_space.high.max(),
            shape=new_shape,
            dtype=old_space.dtype
        )

    def observation(self, obs):
        self.frames.append(obs)
        if len(self.frames) > self.num_frames:
            self.frames.pop(0)

        # Pad with first frame if we don't have enough frames yet
        while len(self.frames) < self.num_frames:
            self.frames.insert(0, obs)

        # Stack frames along last dimension
        stacked = np.concatenate(self.frames, axis=-1)
        return stacked

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames = []
        return self.observation(obs), info


def make_env(env_name=None, render_mode=None, preprocess=True, frame_stack=4,
             use_rgb_rendering=False, **env_kwargs):
    """
    Create and wrap Tetris Gymnasium environment
    
    Args:
        env_name: Environment name (default: ENV_NAME)
        render_mode: Render mode for environment
        preprocess: Whether to apply preprocessing
        frame_stack: Number of frames to stack (0 to disable)
        use_rgb_rendering: Use RGB rendering instead of dict observations
        **env_kwargs: Additional environment arguments
    
    Returns:
        Wrapped environment
    """
    if env_name is None:
        env_name = ENV_NAME

    print(f"Creating environment: {env_name}")

    # Create base environment
    env = gym.make(
        env_name, render_mode=render_mode or "rgb_array", **env_kwargs)

    # Apply observation wrapper to handle dict observations
    env = TetrisObservationWrapper(env, use_rgb_rendering=use_rgb_rendering)

    # Apply preprocessing if requested
    if preprocess:
        env = TetrisPreprocessWrapper(
            env, target_size=(84, 84), grayscale=True)

    # Apply frame stacking if requested
    if frame_stack > 1:
        env = FrameStackWrapper(env, num_frames=frame_stack)

    print(f"Final observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    return env


def test_environment(episodes=1, steps_per_episode=100):
    """
    Test environment creation and basic functionality
    
    Args:
        episodes: Number of test episodes
        steps_per_episode: Steps per episode
    
    Returns:
        True if test successful, False otherwise
    """
    try:
        print("Testing Tetris Gymnasium environment...")

        # Test basic environment
        env = make_env(render_mode="rgb_array")

        for episode in range(episodes):
            obs, info = env.reset(seed=42)
            total_reward = 0

            print(f"Episode {episode + 1}:")
            print(f"  Initial observation shape: {obs.shape}")
            print(f"  Observation range: [{obs.min():.3f}, {obs.max():.3f}]")
            print(f"  Action space: {env.action_space}")

            for step in range(steps_per_episode):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward

                if terminated or truncated:
                    print(f"  Episode ended at step {step + 1}")
                    break

            print(f"  Total reward: {total_reward}")
            print(f"  Final observation shape: {obs.shape}")

        env.close()
        print("✅ Environment test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_different_configs():
    """Test different environment configurations"""
    print("\nTesting different configurations...")

    configs = [
        {"name": "Default", "kwargs": {}},
        {"name": "No preprocessing", "kwargs": {"preprocess": False}},
        {"name": "No frame stack", "kwargs": {"frame_stack": 1}},
        {"name": "RGB rendering", "kwargs": {"use_rgb_rendering": True}},
    ]

    for config in configs:
        try:
            print(f"\n  Testing: {config['name']}")
            env = make_env(**config['kwargs'])
            obs, info = env.reset()
            print(f"    Observation shape: {obs.shape}")
            print(f"    Observation dtype: {obs.dtype}")
            print(f"    Observation range: [{obs.min():.3f}, {obs.max():.3f}]")
            env.close()
            print(f"    ✅ {config['name']} works")
        except Exception as e:
            print(f"    ❌ {config['name']} failed: {e}")


if __name__ == "__main__":
    # Test the environment creation
    success = test_environment()

    if success:
        test_different_configs()
        print("\n🎉 All environment tests passed!")
    else:
        print("\n❌ Environment tests failed!")
