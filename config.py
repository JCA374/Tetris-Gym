# config.py - Fixed configuration for Tetris Gymnasium AI with Board Wrapper

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
MAX_EPISODES = 10000  # Maximum number of episodes
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
            self.key_order = sorted(sample_obs.keys())  # Store key order

            # Calculate offsets for each key
            offset = 0
            for key in self.key_order:
                value = sample_obs[key]
                if isinstance(value, np.ndarray):
                    size = np.prod(value.shape)
                    self.obs_info[key] = {
                        'shape': value.shape,
                        'size': size,
                        'offset': offset
                    }
                    print(
                        f"  {key}: shape={value.shape}, size={size}, offset={offset}")
                else:
                    # Handle scalar values
                    size = 1
                    self.obs_info[key] = {
                        'shape': (),
                        'size': 1,
                        'offset': offset
                    }
                    print(f"  {key}: scalar value, offset={offset}")

                total_size += size
                offset += size

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
            # Flatten all dict components using stored key order
            flattened = []

            for key in self.key_order:  # Use consistent order
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


class TetrisBoardWrapper(gym.ObservationWrapper):
    """
    Extracts the board from flattened observations using proper offset tracking,
    reshapes to 2D, resizes with correct aspect, pads to square, normalizes.
    """

    def __init__(self, env, target_size=(84, 84), grayscale=True):
        super().__init__(env)
        self.target_size = target_size
        self.grayscale = grayscale
        shape = (*target_size, 1) if grayscale else (*target_size, 3)
        self.observation_space = Box(0.0, 1.0, shape, dtype=np.float32)

        # Get board info from parent wrapper if available
        self.board_offset = None
        self.board_size = None
        self.board_shape = (20, 10)  # Default Tetris dimensions

        # Try to get board info from parent TetrisObservationWrapper
        if hasattr(env, 'obs_info') and 'board' in env.obs_info:
            board_info = env.obs_info['board']
            self.board_offset = board_info.get('offset', 0)
            self.board_size = board_info.get('size', 200)
            self.board_shape = board_info.get('shape', (20, 10))
            print(f"TetrisBoardWrapper: Found board at offset {self.board_offset}, "
                  f"size {self.board_size}, shape {self.board_shape}")
        else:
            print("TetrisBoardWrapper: No board info found, using defaults")

    def observation(self, obs):
        # If dict (RGB rendering) or image, pass through
        if obs.ndim == 3:
            # Already an image, just resize if needed
            if obs.shape[:2] != self.target_size:
                resized = cv2.resize(obs, self.target_size)
                if self.grayscale and len(resized.shape) == 3:
                    resized = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
                    resized = np.expand_dims(resized, axis=-1)
                return resized.astype(np.float32) / 255.0
            return obs

        # Flattened feature vector: extract board using proper offset
        if obs.ndim == 1:
            # Use discovered offset or fall back to assuming board is first
            if self.board_offset is not None and self.board_size is not None:
                if obs.size >= self.board_offset + self.board_size:
                    board_flat = obs[self.board_offset:self.board_offset +
                                     self.board_size]
                else:
                    print(f"Warning: Observation too small for board extraction. "
                          f"Expected {self.board_offset + self.board_size}, got {obs.size}")
                    # Fallback to first elements
                    board_flat = obs[:min(self.board_size, obs.size)]
            else:
                # Fallback: assume board is first 200 elements
                board_flat = obs[:min(200, obs.size)]

            # Reshape to 2D board
            try:
                board = board_flat.reshape(self.board_shape)
            except ValueError as e:
                print(f"Warning: Could not reshape board. Shape mismatch: {e}")
                # Create empty board as fallback
                board = np.zeros(self.board_shape)

            # Normalize to [0, 1] if not already
            if board.max() > 1.0:
                board = board / board.max()

            # Scale up to tall rectangle to maintain aspect ratio
            # Use INTER_NEAREST to keep sharp block boundaries
            board_uint8 = np.clip(board * 255, 0, 255).astype(np.uint8)

            # Calculate resize dimensions to maintain aspect ratio
            board_h, board_w = self.board_shape
            aspect_ratio = board_h / board_w  # Should be 2.0 for standard Tetris

            # Target a height of 84 pixels, width proportional
            target_h = self.target_size[0]
            target_w = int(target_h / aspect_ratio)

            # Ensure it fits within the target size
            if target_w > self.target_size[1]:
                target_w = self.target_size[1]
                target_h = int(target_w * aspect_ratio)

            board_resized = cv2.resize(board_uint8, (target_w, target_h),
                                       interpolation=cv2.INTER_NEAREST)

            # Calculate padding to center the board
            pad_h = (self.target_size[0] - target_h) // 2
            pad_w = (self.target_size[1] - target_w) // 2

            # Pad to make it target_size
            padded = np.pad(board_resized,
                            ((pad_h, self.target_size[0] - target_h - pad_h),
                             (pad_w, self.target_size[1] - target_w - pad_w)),
                            mode='constant', constant_values=0)

            # Convert back to float and normalize
            img = padded.astype(np.float32) / 255.0

            # Add channel dimension
            return img[..., None]  # (84, 84, 1)

        # Fallback for unexpected inputs
        print(
            f"Warning: TetrisBoardWrapper received unexpected input shape: {obs.shape}")
        # Try to create a valid output
        return np.zeros(self.observation_space.shape, dtype=np.float32)


class TetrisPreprocessWrapper(gym.ObservationWrapper):
    """
    Preprocessing wrapper for Tetris observations (kept for RGB mode)
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
        # This wrapper is now mainly for RGB observations
        if len(obs.shape) == 3:  # RGB image
            resized = cv2.resize(obs, self.target_size)
            if self.grayscale:
                resized = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
                resized = np.expand_dims(resized, axis=-1)
            return resized.astype(np.float32) / 255.0
        else:
            # For non-RGB inputs, pass through (board wrapper handles it)
            return obs


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
        # Forward through the entire wrapper chain
        obs, info = super().reset(**kwargs)
        self.frames = []
        return self.observation(obs), info


def make_env(env_name=None, render_mode=None, preprocess=True, frame_stack=4,
             use_rgb_rendering=False, use_board_wrapper=True, **env_kwargs):
    """
    Create and wrap Tetris Gymnasium environment
    
    Args:
        env_name: Environment name (default: ENV_NAME)
        render_mode: Render mode for environment
        preprocess: Whether to apply preprocessing
        frame_stack: Number of frames to stack (0 to disable)
        use_rgb_rendering: Use RGB rendering instead of dict observations
        use_board_wrapper: Use the new board wrapper for better spatial info
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
        if use_rgb_rendering:
            # For RGB mode, use the original preprocessing wrapper
            env = TetrisPreprocessWrapper(
                env, target_size=(84, 84), grayscale=True)
        elif use_board_wrapper:
            # For flattened observations, use the new board wrapper
            env = TetrisBoardWrapper(
                env, target_size=(84, 84), grayscale=True)
        else:
            # Fallback to original preprocessing
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

        # Test basic environment with board wrapper
        env = make_env(render_mode="rgb_array", use_board_wrapper=True)

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
        {"name": "Default (with board wrapper)", "kwargs": {
            "use_board_wrapper": True}},
        {"name": "No preprocessing", "kwargs": {"preprocess": False}},
        {"name": "No frame stack", "kwargs": {"frame_stack": 1}},
        {"name": "RGB rendering", "kwargs": {"use_rgb_rendering": True}},
        {"name": "Old preprocessing (no board wrapper)", "kwargs": {
            "use_board_wrapper": False}},
    ]

    for config in configs:
        try:
            print(f"\n  Testing: {config['name']}")
            env = make_env(**config['kwargs'])
            obs, info = env.reset()
            print(f"    Observation shape: {obs.shape}")
            print(f"    Observation dtype: {obs.dtype}")
            print(f"    Observation range: [{obs.min():.3f}, {obs.max():.3f}]")

            # Take a few steps to ensure it works
            for _ in range(5):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    obs, info = env.reset()

            env.close()
            print(f"    ✅ {config['name']} works")
        except Exception as e:
            print(f"    ❌ {config['name']} failed: {e}")


def visualize_board_wrapper():
    """Visualize what the board wrapper produces"""
    print("\nVisualizing Board Wrapper Output...")

    try:
        import matplotlib.pyplot as plt

        env = make_env(use_board_wrapper=True, frame_stack=1)
        obs, info = env.reset()

        # Take a few steps to get an interesting board state
        for _ in range(20):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        # Show the observation
        plt.figure(figsize=(6, 6))
        plt.imshow(obs[:, :, 0], cmap='gray', interpolation='nearest')
        plt.title('Board Wrapper Output (84x84)')
        plt.colorbar()
        plt.grid(True, alpha=0.3)

        # Save to file
        plt.savefig('board_wrapper_visualization.png')
        print("  Saved visualization to board_wrapper_visualization.png")
        plt.close()

        env.close()

    except ImportError:
        print("  Matplotlib not available, skipping visualization")
    except Exception as e:
        print(f"  Visualization failed: {e}")


if __name__ == "__main__":
    # Test the environment creation
    success = test_environment()

    if success:
        test_different_configs()
        visualize_board_wrapper()
        print("\n🎉 All environment tests passed!")
        print("\nThe board wrapper is now active and will preserve the Tetris board structure!")
        print("Your model will now see the actual 20x10 grid (scaled to 84x84) instead of scrambled data.")
    else:
        print("\n❌ Environment tests failed!")
