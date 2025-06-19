# config.py - Final fixed configuration for Tetris Gymnasium AI

import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np
import cv2
from gymnasium.spaces import Box

# Register the Tetris environment manually
try:
    register(
        id="TetrisManual-v0",
        entry_point="tetris_gymnasium.envs.tetris:Tetris",
    )
    print("Tetris environment registered successfully")
except gym.error.Error:
    pass

# Environment settings
ENV_NAME = "TetrisManual-v0"
RENDER_MODE = None
FRAME_STACK = 4
PREPROCESS = True
REWARD_SHAPING = True

# Training hyperparameters
LR = 1e-4
GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 100000
MIN_REPLAY_SIZE = 10000
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 100000
TARGET_UPDATE_FREQ = 1000

# Training settings
MAX_EPISODES = 10000
MAX_STEPS_PER_EPISODE = 10000
SAVE_FREQUENCY = 50
LOG_FREQUENCY = 10

# Directories
MODEL_DIR = "models/"
LOG_DIR = "logs/"
CHECKPOINT_DIR = "checkpoints/"
DEVICE = "cuda"
SEED = 42


class TetrisObservationWrapper(gym.ObservationWrapper):
    """Extract only the board from dict observations for consistency"""

    def __init__(self, env):
        super().__init__(env)

        # Get sample observation to understand structure
        sample_obs = env.observation_space.sample()
        print(f"Sample observation keys: {list(sample_obs.keys())}")

        # Always extract just the board for simplicity and consistency
        if 'board' in sample_obs:
            board_shape = sample_obs['board'].shape
            print(f"Board shape: {board_shape}")

            # Create observation space for just the board
            self.observation_space = Box(
                low=0, high=1, shape=board_shape, dtype=np.float32
            )
        else:
            raise ValueError("No 'board' key found in observation space")

        print(f"TetrisObservationWrapper output: {self.observation_space}")

    def observation(self, obs):
        """Extract only the board and normalize it"""
        if isinstance(obs, dict) and 'board' in obs:
            board = obs['board'].astype(np.float32)

            # Normalize to [0, 1] range
            if board.max() > 1.0:
                board = board / board.max()

            return board
        else:
            raise ValueError(
                f"Expected dict with 'board' key, got {type(obs)}")


class TetrisBoardWrapper(gym.ObservationWrapper):
    """Convert board to CNN-friendly format"""

    def __init__(self, env, target_size=(84, 84)):
        super().__init__(env)
        self.target_size = target_size

        # Output is always grayscale image
        self.observation_space = Box(
            low=0.0, high=1.0, shape=(*target_size, 1), dtype=np.float32
        )

        print(f"TetrisBoardWrapper: {self.observation_space}")

    def observation(self, obs):
        """Convert board array to image format"""
        if obs.ndim != 2:
            raise ValueError(f"Expected 2D board, got shape {obs.shape}")

        # Convert to uint8 for OpenCV
        board_uint8 = np.clip(obs * 255, 0, 255).astype(np.uint8)

        # Get board dimensions
        board_h, board_w = obs.shape
        aspect_ratio = board_h / board_w

        # Calculate target dimensions maintaining aspect ratio
        target_h = self.target_size[0]
        target_w = int(target_h / aspect_ratio)

        # Ensure it fits
        if target_w > self.target_size[1]:
            target_w = self.target_size[1]
            target_h = int(target_w * aspect_ratio)

        # Resize with nearest neighbor to keep sharp edges
        resized = cv2.resize(board_uint8, (target_w, target_h),
                             interpolation=cv2.INTER_NEAREST)

        # Pad to target size
        pad_h = (self.target_size[0] - target_h) // 2
        pad_w = (self.target_size[1] - target_w) // 2

        padded = np.pad(resized,
                        ((pad_h, self.target_size[0] - target_h - pad_h),
                         (pad_w, self.target_size[1] - target_w - pad_w)),
                        mode='constant', constant_values=0)

        # Convert back to float and add channel dimension
        result = padded.astype(np.float32) / 255.0
        result = np.expand_dims(result, axis=-1)  # (84, 84, 1)

        return result


class FrameStackWrapper(gym.ObservationWrapper):
    """Fixed frame stacking with proper reset behavior"""

    def __init__(self, env, num_frames=4):
        super().__init__(env)
        self.num_frames = num_frames
        self.frames = None

        # Calculate output observation space
        old_space = env.observation_space
        if len(old_space.shape) == 3:  # (H, W, C)
            new_shape = (old_space.shape[0], old_space.shape[1],
                         old_space.shape[2] * num_frames)
        else:
            raise ValueError(
                f"Expected 3D observation space, got {old_space.shape}")

        self.observation_space = Box(
            low=old_space.low.min(),
            high=old_space.high.max(),
            shape=new_shape,
            dtype=old_space.dtype
        )

        print(f"FrameStackWrapper: {old_space.shape} -> {new_shape}")

    def observation(self, obs):
        """Stack frames with simple, reliable logic"""
        # Ensure observation has correct shape
        if len(obs.shape) != 3:
            raise ValueError(f"Expected 3D observation, got {obs.shape}")

        # Initialize frame buffer if needed
        if self.frames is None:
            # Create buffer filled with the current observation
            self.frames = [obs.copy() for _ in range(self.num_frames)]
        else:
            # Shift frames and add new one
            self.frames = self.frames[1:] + [obs.copy()]

        # Stack along channel dimension
        stacked = np.concatenate(self.frames, axis=-1)

        # Verify output shape (optional - remove for production)
        expected_shape = self.observation_space.shape
        if stacked.shape != expected_shape:
            raise ValueError(f"Frame stack output shape mismatch: "
                             f"got {stacked.shape}, expected {expected_shape}")

        return stacked

    def reset(self, **kwargs):
        """Reset frame buffer - trust the wrapper chain"""
        # Clear frame buffer so first observation will re-initialize it
        self.frames = None

        # super().reset() will call self.observation() exactly once
        return super().reset(**kwargs)


def make_env(env_name=None, render_mode=None, preprocess=True, frame_stack=4, **env_kwargs):
    """
    Create Tetris environment with consistent observation processing
    """
    if env_name is None:
        env_name = ENV_NAME

    print(f"Creating environment: {env_name}")

    # Create base environment
    env = gym.make(
        env_name, render_mode=render_mode or "rgb_array", **env_kwargs)

    # Step 1: Extract board only (consistent output)
    env = TetrisObservationWrapper(env)

    # Step 2: Convert to image format if preprocessing enabled
    if preprocess:
        env = TetrisBoardWrapper(env, target_size=(84, 84))

    # Step 3: Stack frames if requested
    if frame_stack > 1:
        env = FrameStackWrapper(env, num_frames=frame_stack)

    print(f"Final observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    return env


def test_environment(episodes=1, steps_per_episode=100):
    """Test environment creation and basic functionality"""
    try:
        print("Testing Tetris Gymnasium environment...")

        env = make_env(frame_stack=4)

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
        print("Environment test completed successfully!")
        return True

    except Exception as e:
        print(f"Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Test the environment creation
    success = test_environment()
    if success:
        print("\nEnvironment tests passed!")
        print("Ready for training!")
    else:
        print("\nEnvironment tests failed!")
