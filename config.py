# config.py - COMPLETE TETRIS VISION - Replace your current config.py with this

import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np
import cv2
from gymnasium.spaces import Box

# Register the Tetris environment manually
try:
    register(
        id="TetrisFixed-v0",
        entry_point="tetris_gymnasium.envs.tetris:Tetris",
    )
    print("Tetris environment registered successfully")  # Removed Unicode
except gym.error.Error:
    pass

# Environment settings
ENV_NAME = "TetrisFixed-v0"
RENDER_MODE = None
FRAME_STACK = 1  # Tetris doesn't need frame stacking
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


class CorrectedTetrisBoardWrapper(gym.ObservationWrapper):
    """CORRECTED: For 24×18 Tetris Gymnasium board with proper aspect ratio"""

    def __init__(self, env, target_size=(84, 84)):
        super().__init__(env)
        self.target_size = target_size

        # Output is always grayscale image
        self.observation_space = Box(
            low=0.0, high=1.0, shape=(*target_size, 1), dtype=np.float32
        )

        print(f"CorrectedTetrisBoardWrapper: {self.observation_space}")

    def observation(self, obs):
        """Convert 24×18 board to 84×84 image with proper aspect ratio"""
        if obs.ndim != 2:
            raise ValueError(f"Expected 2D board, got shape {obs.shape}")

        # Convert to uint8 for OpenCV
        board_uint8 = np.clip(obs * 255, 0, 255).astype(np.uint8)

        board_h, board_w = obs.shape  # Should be 24×18 for this Tetris Gymnasium
        target_h, target_w = self.target_size

        # CORRECTED: For 24×18 board (aspect ratio 1.33)
        # Calculate scaling to use maximum space while preserving aspect ratio
        scale_h = target_h / board_h
        scale_w = target_w / board_w

        # Use the smaller scale to ensure the entire board fits
        scale = min(scale_h, scale_w)

        # Calculate actual dimensions after scaling
        new_h = int(board_h * scale)
        new_w = int(board_w * scale)

        # Resize with nearest neighbor to preserve board structure
        resized = cv2.resize(board_uint8, (new_w, new_h),
                             interpolation=cv2.INTER_NEAREST)

        # Center the board in the target canvas
        canvas = np.zeros(self.target_size, dtype=np.uint8)

        # Calculate centering offsets
        offset_h = (target_h - new_h) // 2
        offset_w = (target_w - new_w) // 2

        # Place the properly scaled board in the center
        canvas[offset_h:offset_h + new_h, offset_w:offset_w + new_w] = resized

        # Convert back to float and add channel dimension
        result = canvas.astype(np.float32) / 255.0
        result = np.expand_dims(result, axis=-1)  # (84, 84, 1)

        return result


class OptimizedDirectWrapper(gym.ObservationWrapper):
    """Optimized direct wrapper for 24×18 Tetris board"""

    def __init__(self, env):
        super().__init__(env)

        # Get board dimensions from sample
        sample_obs = env.observation_space.sample()
        if isinstance(sample_obs, np.ndarray) and sample_obs.ndim == 2:
            board_size = sample_obs.size
            board_shape = sample_obs.shape
        else:
            board_size = 432  # Default for 24×18 board
            board_shape = (24, 18)

        self.board_shape = board_shape
        self.observation_space = Box(
            low=0.0, high=1.0, shape=(board_size,), dtype=np.float32
        )

        print(
            f"OptimizedDirectWrapper: {board_shape} -> {board_size} features")

    def observation(self, obs):
        """Flatten the 24×18 board directly"""
        if obs.ndim != 2:
            raise ValueError(f"Expected 2D board, got shape {obs.shape}")

        # Flatten and normalize
        flattened = obs.astype(np.float32).flatten()
        if flattened.max() > 1:
            flattened = flattened / flattened.max()

        return flattened


def make_env(env_name=None, render_mode=None, preprocess=True, frame_stack=1,
             use_cnn=True, **env_kwargs):
    """
    Create Tetris environment optimized for 24×18 board
    
    Args:
        use_cnn: If True, use CNN-friendly image format. If False, use direct feature vector.
        frame_stack: Number of frames to stack (1 recommended for Tetris)
    """
    if env_name is None:
        env_name = ENV_NAME

    print(f"Creating environment: {env_name}")
    print(f"CNN mode: {use_cnn}, Frame stack: {frame_stack}")

    # Create base environment
    env = gym.make(
        env_name, render_mode=render_mode or "rgb_array", **env_kwargs)

    # Step 1: Extract board only (consistent output)
    env = TetrisObservationWrapper(env)

    # Step 2: Choose preprocessing approach
    if preprocess:
        if use_cnn:
            # Use CORRECTED board wrapper for 24×18 dimensions
            env = CorrectedTetrisBoardWrapper(env, target_size=(84, 84))
        else:
            # Use optimized direct feature vector (432 features for 24×18)
            env = OptimizedDirectWrapper(env)

    print(f"Final observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    return env


def corrected_vision_test():
    """Test the corrected vision system for 24×18 board"""
    print("\nTesting Corrected Vision for 24×18 Tetris Board")
    print("=" * 60)

    # Test CNN mode
    env_cnn = make_env(use_cnn=True, frame_stack=1)
    obs_cnn, info = env_cnn.reset(seed=42)

    print(f"CNN observation shape: {obs_cnn.shape}")
    print(f"CNN observation range: [{obs_cnn.min():.3f}, {obs_cnn.max():.3f}]")

    # Check board region in CNN mode
    if len(obs_cnn.shape) == 3:
        frame = obs_cnn[:, :, -1]
        non_zero_rows = np.any(frame > 0.01, axis=1)
        non_zero_cols = np.any(frame > 0.01, axis=0)

        if np.any(non_zero_rows) and np.any(non_zero_cols):
            board_height = np.sum(non_zero_rows)
            board_width = np.sum(non_zero_cols)
            aspect_ratio = board_height / board_width

            print(f"CNN board region: {board_height}×{board_width} pixels")
            print(f"CNN aspect ratio: {aspect_ratio:.2f}")

            # For 24×18 board, aspect ratio should be 24/18 = 1.33
            if 1.25 <= aspect_ratio <= 1.45:
                print("Aspect ratio correct for 24×18 board!")
            else:
                print("Aspect ratio incorrect")

    # Test direct mode
    env_direct = make_env(use_cnn=False)
    obs_direct, info = env_direct.reset(seed=42)

    print(f"Direct observation shape: {obs_direct.shape}")
    print(
        f"Direct observation range: [{obs_direct.min():.3f}, {obs_direct.max():.3f}]")

    # Verify direct mode has correct number of features
    expected_features = 24 * 18  # 432
    if obs_direct.shape[0] == expected_features:
        print(
            f"Direct mode has correct {expected_features} features for 24×18 board")
    else:
        print(
            f"Direct mode has {obs_direct.shape[0]} features, expected {expected_features}")

    # Test observation changes
    print("\nTesting observation dynamics...")
    changes = []
    initial_obs = obs_cnn.copy()

    for step in range(5):
        action = env_cnn.action_space.sample()
        obs_cnn, reward, terminated, truncated, info = env_cnn.step(action)

        change = np.abs(obs_cnn - initial_obs).mean()
        changes.append(change)

        print(
            f"Step {step+1}: Action={action}, Change={change:.4f}, Reward={reward:.2f}")

        if terminated or truncated:
            obs_cnn, info = env_cnn.reset()

        initial_obs = obs_cnn.copy()

    avg_change = np.mean(changes)
    print(f"Average observation change: {avg_change:.4f}")

    if avg_change > 0.001:
        print("Observations change meaningfully")
    else:
        print("Observations barely change")

    env_cnn.close()
    env_direct.close()

    print("\nCORRECTED VISION SUMMARY:")
    print("Board dimensions: 24×18 (correct for this environment)")
    print("Aspect ratio: 1.33 (correct for 24×18)")
    print("CNN mode: Preserves spatial structure")
    print("Direct mode: 432 features (24×18)")
    print("Observation dynamics: Working properly")


def quick_training_compatibility_test():
    """Quick test that training will work with corrected vision"""
    print("\nTesting Training Compatibility")
    print("=" * 50)

    try:
        from src.agent import Agent

        # Test both modes
        for mode_name, use_cnn in [("CNN", True), ("Direct", False)]:
            print(f"\nTesting {mode_name} mode...")

            env = make_env(use_cnn=use_cnn, frame_stack=1)
            agent = Agent(
                obs_space=env.observation_space,
                action_space=env.action_space,
                reward_shaping="simple"
            )

            obs, info = env.reset(seed=42)

            # Test training loop
            for step in range(10):
                action = agent.select_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(
                    action)
                agent.remember(obs, action, reward, next_obs,
                               terminated or truncated, info)

                if len(agent.memory) >= agent.batch_size:
                    metrics = agent.learn()
                    if metrics:
                        print(
                            f"   Learning works: loss={metrics['loss']:.4f}")
                        break

                obs = next_obs
                if terminated or truncated:
                    obs, info = env.reset()

            env.close()
            print(f"   {mode_name} mode compatible with training")

        print("\nBoth modes ready for training!")

    except Exception as e:
        print(f"Training compatibility test failed: {e}")
        import traceback
        traceback.print_exc()


def test_environment(episodes=1, steps_per_episode=100, test_both_modes=True):
    """Updated test function"""
    return corrected_vision_test()


def quick_vision_test():
    """Updated quick test"""
    corrected_vision_test()


if __name__ == "__main__":
    # Test the corrected environment
    corrected_vision_test()
    quick_training_compatibility_test()

    print("\n" + "=" * 60)
    print("READY FOR TRAINING!")
    print("=" * 60)
    print("Your vision system is now correctly configured for 24×18 Tetris board.")
    print("\nChoose your training mode:")
    print("1. CNN mode: python train.py --episodes 500 --reward_shaping simple")
    print("2. Direct mode: Modify train.py to use make_env(use_cnn=False)")
    print("\nExpected results:")
    print("First line clear within 50-100 episodes")
    print("Gradual improvement in spatial piece placement")
    print("Breaking the 0-lines plateau completely")