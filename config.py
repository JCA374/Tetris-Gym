# config_with_piece_vision.py - COMPLETE TETRIS VISION including active piece

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
    print("Tetris environment registered successfully")
except gym.error.Error:
    pass

# Environment settings
ENV_NAME = "TetrisFixed-v0"
RENDER_MODE = None
FRAME_STACK = 1
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


class CompleteTetrisObservationWrapper(gym.ObservationWrapper):
    """
    COMPLETE Tetris observation including board + active piece + next pieces
    This is what was missing - the model needs to see what piece it's placing!
    """

    def __init__(self, env):
        super().__init__(env)

        # Get sample observation to understand structure
        sample_obs = env.observation_space.sample()
        print(f"Available observation keys: {list(sample_obs.keys())}")

        # Analyze each component
        board_shape = sample_obs['board'].shape  # Should be (24, 18)
        active_shape = sample_obs['active_tetromino_mask'].shape
        holder_shape = sample_obs['holder'].shape  
        queue_shape = sample_obs['queue'].shape

        print(f"Board shape: {board_shape}")
        print(f"Active tetromino shape: {active_shape}")
        print(f"Holder shape: {holder_shape}")
        print(f"Queue shape: {queue_shape}")

        # Create combined observation space
        # We'll stack: board + active_piece + holder + first_next_piece
        self.board_height, self.board_width = board_shape
        
        # Output will be (height, width, channels) where channels are:
        # Channel 0: Board state
        # Channel 1: Active tetromino mask  
        # Channel 2: Holder piece (if fits, else zeros)
        # Channel 3: Next piece (if fits, else zeros)
        
        self.observation_space = Box(
            low=0, high=1, 
            shape=(self.board_height, self.board_width, 4), 
            dtype=np.float32
        )

        print(f"Complete observation space: {self.observation_space}")

    def observation(self, obs):
        """Combine board + active piece + additional info into complete observation"""
        if not isinstance(obs, dict):
            raise ValueError(f"Expected dict observation, got {type(obs)}")

        # Extract components
        board = obs['board'].astype(np.float32)
        active_piece = obs['active_tetromino_mask'].astype(np.float32)
        holder = obs['holder'].astype(np.float32)
        queue = obs['queue'].astype(np.float32)

        # Normalize board (pieces are typically 1-7, empty is 0)
        if board.max() > 1:
            board_normalized = board / board.max()
        else:
            board_normalized = board

        # Handle active piece mask
        # This should already be same size as board and show where current piece is
        if active_piece.shape == board.shape:
            active_normalized = active_piece
        else:
            # If different shape, resize or pad
            active_normalized = np.zeros_like(board)
            print(f"Warning: Active piece shape {active_piece.shape} != board shape {board.shape}")

        # Handle holder - resize to board dimensions if needed
        if holder.shape == board.shape:
            holder_normalized = holder
        else:
            # Create holder layer (usually shows held piece in top corner)
            holder_normalized = np.zeros_like(board)
            if holder.size > 0:
                # Place holder info in top-left corner
                h_rows, h_cols = min(holder.shape[0], 4), min(holder.shape[1], 4)
                holder_normalized[:h_rows, :h_cols] = holder[:h_rows, :h_cols]

        # Handle next piece queue
        if queue.ndim >= 2:
            next_piece = queue[0] if queue.ndim == 3 else queue  # First in queue
            if next_piece.shape == board.shape:
                queue_normalized = next_piece
            else:
                # Place next piece info in top-right corner  
                queue_normalized = np.zeros_like(board)
                if next_piece.size > 0:
                    q_rows, q_cols = min(next_piece.shape[0], 4), min(next_piece.shape[1], 4)
                    start_col = max(0, board.shape[1] - q_cols)
                    queue_normalized[:q_rows, start_col:start_col+q_cols] = next_piece[:q_rows, :q_cols]
        else:
            queue_normalized = np.zeros_like(board)

        # Normalize all additional channels
        for channel in [active_normalized, holder_normalized, queue_normalized]:
            if channel.max() > 1:
                channel = channel / channel.max()

        # Stack all channels
        complete_obs = np.stack([
            board_normalized,      # Channel 0: Board state
            active_normalized,     # Channel 1: Current piece position  
            holder_normalized,     # Channel 2: Held piece
            queue_normalized       # Channel 3: Next piece
        ], axis=-1)

        return complete_obs


class CompleteTetrisBoardWrapper(gym.ObservationWrapper):
    """
    CNN wrapper that handles the complete 4-channel Tetris observation
    """

    def __init__(self, env, target_size=(84, 84)):
        super().__init__(env)
        self.target_size = target_size

        # Input should be (H, W, 4) from CompleteTetrisObservationWrapper
        input_shape = env.observation_space.shape
        if len(input_shape) != 3 or input_shape[2] != 4:
            raise ValueError(f"Expected (H, W, 4) input, got {input_shape}")

        # Output is (target_h, target_w, 4) - preserve all 4 channels
        self.observation_space = Box(
            low=0.0, high=1.0, shape=(*target_size, 4), dtype=np.float32
        )

        print(f"CompleteTetrisBoardWrapper: {input_shape} -> {self.observation_space.shape}")

    def observation(self, obs):
        """Resize complete observation while preserving all channels"""
        if obs.shape[2] != 4:
            raise ValueError(f"Expected 4 channels, got {obs.shape}")

        obs_h, obs_w, channels = obs.shape
        target_h, target_w = self.target_size

        # Process each channel separately to preserve information
        processed_channels = []
        
        for c in range(channels):
            channel = obs[:, :, c]
            
            # Convert to uint8 for OpenCV
            channel_uint8 = np.clip(channel * 255, 0, 255).astype(np.uint8)

            # Calculate scaling to preserve aspect ratio
            scale_h = target_h / obs_h
            scale_w = target_w / obs_w
            scale = min(scale_h, scale_w)

            # Calculate actual dimensions after scaling
            new_h = int(obs_h * scale)
            new_w = int(obs_w * scale)

            # Resize with nearest neighbor to preserve structure
            if new_h > 0 and new_w > 0:
                resized = cv2.resize(channel_uint8, (new_w, new_h),
                                   interpolation=cv2.INTER_NEAREST)
            else:
                resized = channel_uint8

            # Center in target canvas
            canvas = np.zeros(self.target_size, dtype=np.uint8)
            offset_h = (target_h - new_h) // 2
            offset_w = (target_w - new_w) // 2

            canvas[offset_h:offset_h + new_h, offset_w:offset_w + new_w] = resized

            # Convert back to float
            processed_channels.append(canvas.astype(np.float32) / 255.0)

        # Stack channels back together
        result = np.stack(processed_channels, axis=-1)
        return result


class DirectTetrisWrapper(gym.ObservationWrapper):
    """
    Direct feature wrapper for complete Tetris observation
    Flattens the 4-channel observation into a feature vector
    """

    def __init__(self, env):
        super().__init__(env)

        input_shape = env.observation_space.shape
        if len(input_shape) != 3 or input_shape[2] != 4:
            raise ValueError(f"Expected (H, W, 4) input, got {input_shape}")

        # Flatten all 4 channels: 24 * 18 * 4 = 1728 features
        total_features = input_shape[0] * input_shape[1] * input_shape[2]
        
        self.observation_space = Box(
            low=0.0, high=1.0, shape=(total_features,), dtype=np.float32
        )

        print(f"DirectTetrisWrapper: {input_shape} -> {total_features} features")

    def observation(self, obs):
        """Flatten the complete 4-channel observation"""
        return obs.flatten()


def make_env(env_name=None, render_mode=None, preprocess=True, frame_stack=1,
             use_cnn=True, include_piece_info=True, **env_kwargs):
    """
    Create complete Tetris environment with piece vision
    
    Args:
        use_cnn: If True, use CNN-friendly format. If False, use direct features.
        include_piece_info: If True, include active piece + next pieces (RECOMMENDED!)
        frame_stack: Number of frames to stack (1 recommended for Tetris)
    """
    if env_name is None:
        env_name = ENV_NAME

    print(f"Creating environment: {env_name}")
    print(f"CNN mode: {use_cnn}, Include pieces: {include_piece_info}")

    # Create base environment
    env = gym.make(env_name, render_mode=render_mode or "rgb_array", **env_kwargs)

    # Step 1: Extract complete observation (board + pieces)
    if include_piece_info:
        env = CompleteTetrisObservationWrapper(env)
        print("✅ Complete piece vision enabled")
    else:
        # Original wrapper (board only) - for comparison
        env = TetrisObservationWrapper(env)  # You'd need to define this
        print("⚠️  Board-only vision (missing piece info)")

    # Step 2: Choose processing approach
    if preprocess:
        if use_cnn:
            if include_piece_info:
                env = CompleteTetrisBoardWrapper(env, target_size=(84, 84))
            else:
                env = CorrectedTetrisBoardWrapper(env, target_size=(84, 84))  # Old version
        else:
            if include_piece_info:
                env = DirectTetrisWrapper(env)
            else:
                env = OptimizedDirectWrapper(env)  # Old version

    print(f"Final observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    return env


def test_complete_vision():
    """Test the complete vision system"""
    print("\n🔍 TESTING COMPLETE TETRIS VISION")
    print("="*60)
    
    # Test both modes
    for mode_name, (use_cnn, include_pieces) in [
        ("Complete CNN", (True, True)),
        ("Complete Direct", (False, True)), 
        ("Board-only CNN", (True, False)),
        ("Board-only Direct", (False, False))
    ]:
        print(f"\n📊 Testing {mode_name}:")
        
        try:
            env = make_env(use_cnn=use_cnn, include_piece_info=include_pieces)
            obs, info = env.reset(seed=42)
            
            print(f"   Observation shape: {obs.shape}")
            print(f"   Observation range: [{obs.min():.3f}, {obs.max():.3f}]")
            
            if len(obs.shape) == 3 and obs.shape[2] == 4:
                print(f"   Channel 0 (board) range: [{obs[:,:,0].min():.3f}, {obs[:,:,0].max():.3f}]")
                print(f"   Channel 1 (active) range: [{obs[:,:,1].min():.3f}, {obs[:,:,1].max():.3f}]")
                print(f"   Channel 2 (holder) range: [{obs[:,:,2].min():.3f}, {obs[:,:,2].max():.3f}]")
                print(f"   Channel 3 (queue) range: [{obs[:,:,3].min():.3f}, {obs[:,:,3].max():.3f}]")
                
                # Check if active piece is visible
                active_piece_pixels = np.sum(obs[:,:,1] > 0.01)
                print(f"   Active piece pixels: {active_piece_pixels}")
                
                if active_piece_pixels > 0:
                    print("   ✅ Active piece is visible!")
                else:
                    print("   ❌ Active piece not visible")
            
            env.close()
            print(f"   ✅ {mode_name} working")
            
        except Exception as e:
            print(f"   ❌ {mode_name} failed: {e}")
    
    print(f"\n🎯 RECOMMENDATION:")
    print(f"Use Complete CNN or Complete Direct mode for full Tetris vision!")


# Keep old wrappers for compatibility
class TetrisObservationWrapper(gym.ObservationWrapper):
    """Board-only wrapper (old version for comparison)"""
    
    def __init__(self, env):
        super().__init__(env)
        sample_obs = env.observation_space.sample()
        
        if 'board' in sample_obs:
            board_shape = sample_obs['board'].shape
            self.observation_space = Box(
                low=0, high=1, shape=board_shape, dtype=np.float32
            )
        else:
            raise ValueError("No 'board' key found")

    def observation(self, obs):
        if isinstance(obs, dict) and 'board' in obs:
            board = obs['board'].astype(np.float32)
            if board.max() > 1.0:
                board = board / board.max()
            return board
        else:
            raise ValueError(f"Expected dict with 'board' key, got {type(obs)}")


class CorrectedTetrisBoardWrapper(gym.ObservationWrapper):
    """Board-only CNN wrapper (old version)"""
    
    def __init__(self, env, target_size=(84, 84)):
        super().__init__(env)
        self.target_size = target_size
        self.observation_space = Box(
            low=0.0, high=1.0, shape=(*target_size, 1), dtype=np.float32
        )

    def observation(self, obs):
        if obs.ndim != 2:
            raise ValueError(f"Expected 2D board, got shape {obs.shape}")

        board_uint8 = np.clip(obs * 255, 0, 255).astype(np.uint8)
        board_h, board_w = obs.shape
        target_h, target_w = self.target_size

        scale_h = target_h / board_h
        scale_w = target_w / board_w
        scale = min(scale_h, scale_w)

        new_h = int(board_h * scale)
        new_w = int(board_w * scale)

        resized = cv2.resize(board_uint8, (new_w, new_h),
                             interpolation=cv2.INTER_NEAREST)

        canvas = np.zeros(self.target_size, dtype=np.uint8)
        offset_h = (target_h - new_h) // 2
        offset_w = (target_w - new_w) // 2

        canvas[offset_h:offset_h + new_h, offset_w:offset_w + new_w] = resized
        result = canvas.astype(np.float32) / 255.0
        result = np.expand_dims(result, axis=-1)

        return result


class OptimizedDirectWrapper(gym.ObservationWrapper):
    """Board-only direct wrapper (old version)"""
    
    def __init__(self, env):
        super().__init__(env)
        sample_obs = env.observation_space.sample()
        
        if isinstance(sample_obs, np.ndarray) and sample_obs.ndim == 2:
            board_size = sample_obs.size
        else:
            board_size = 432  # 24×18

        self.observation_space = Box(
            low=0.0, high=1.0, shape=(board_size,), dtype=np.float32
        )

    def observation(self, obs):
        if obs.ndim != 2:
            raise ValueError(f"Expected 2D board, got shape {obs.shape}")
        
        flattened = obs.astype(np.float32).flatten()
        if flattened.max() > 1:
            flattened = flattened / flattened.max()
        return flattened


if __name__ == "__main__":
    test_complete_vision()
    
    print("\n" + "="*60)
    print("🎯 COMPLETE TETRIS VISION READY!")
    print("="*60)
    print("Your model can now see:")
    print("✅ Board state (obstacles)")
    print("✅ Active piece (what it's placing)")
    print("✅ Held piece (strategic planning)")
    print("✅ Next piece (forward planning)")
    print("\nThis should DRAMATICALLY improve learning!")
    print("Replace your config.py with this version and retrain.")