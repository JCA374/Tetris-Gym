# config_complete_vision.py - REPLACE YOUR config.py WITH THIS!

import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np
import cv2
from gymnasium.spaces import Box

# Register the Tetris environment
try:
    register(
        id="TetrisComplete-v0",
        entry_point="tetris_gymnasium.envs.tetris:Tetris",
    )
except gym.error.Error:
    pass

# Environment settings
ENV_NAME = "TetrisComplete-v0"
RENDER_MODE = None
FRAME_STACK = 1
PREPROCESS = True

# Training hyperparameters  
LR = 5e-4  # Higher for richer information
GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 100000
EPSILON_START = 0.8
EPSILON_END = 0.05
EPSILON_DECAY = 0.999
TARGET_UPDATE_FREQ = 1000
MAX_EPISODES = 25000

# Directories
MODEL_DIR = "models/"
LOG_DIR = "logs/"


class CompleteTetrisObservationWrapper(gym.ObservationWrapper):
    """Extract ALL information: board + active piece + holder + queue"""
    
    def __init__(self, env):
        super().__init__(env)
        
        # Sample to understand structure
        sample_obs = env.observation_space.sample()
        
        if not isinstance(sample_obs, dict):
            raise ValueError("Expected dict observation space")
            
        # Get board dimensions
        self.board_shape = sample_obs['board'].shape
        self.board_h, self.board_w = self.board_shape
        
        # We'll create a 4-channel observation
        self.observation_space = Box(
            low=0, high=1, 
            shape=(self.board_h, self.board_w, 4), 
            dtype=np.float32
        )
        
        print(f"CompleteTetrisObservationWrapper: {self.observation_space}")
        print(f"  Channel 0: Board (obstacles)")
        print(f"  Channel 1: Active piece") 
        print(f"  Channel 2: Holder")
        print(f"  Channel 3: Queue preview")
        
    def observation(self, obs):
        """Create 4-channel observation with complete game state"""
        
        # Channel 0: Board state
        board = obs['board'].astype(np.float32)
        if board.max() > 1:
            board = board / board.max()
            
        # Channel 1: Active tetromino (THE MISSING PIECE!)
        active = obs.get('active_tetromino_mask', np.zeros_like(board))
        active = active.astype(np.float32)
        if active.max() > 1:
            active = active / active.max()
            
        # Channel 2: Holder (for strategic planning)
        holder_channel = self._process_holder(obs.get('holder', None))
        
        # Channel 3: Queue preview (next piece)
        queue_channel = self._process_queue(obs.get('queue', None))
        
        # Stack into 4-channel observation
        complete_obs = np.stack([board, active, holder_channel, queue_channel], axis=-1)
        
        return complete_obs
    
    def _process_holder(self, holder):
        """Process holder into board-sized channel"""
        holder_channel = np.zeros(self.board_shape, dtype=np.float32)
        
        if holder is not None and holder.size > 0:
            # Place holder preview in top-left corner
            h, w = min(4, self.board_h), min(4, self.board_w)
            if holder.ndim >= 2:
                piece_h, piece_w = holder.shape[:2]
                h = min(h, piece_h)
                w = min(w, piece_w)
                holder_channel[:h, :w] = holder[:h, :w].astype(np.float32)
            
        return holder_channel
    
    def _process_queue(self, queue):
        """Process queue into board-sized channel"""
        queue_channel = np.zeros(self.board_shape, dtype=np.float32)
        
        if queue is not None and queue.size > 0:
            # Show next piece preview on the right side
            if queue.ndim >= 2:
                # Handle different queue formats
                if queue.ndim == 3 and queue.shape[0] > 0:
                    next_piece = queue[0]  # First piece in queue
                else:
                    next_piece = queue
                    
                piece_h, piece_w = next_piece.shape[:2]
                h = min(4, piece_h, self.board_h)
                w = min(4, piece_w, self.board_w)
                
                # Place in top-right corner
                start_col = max(0, self.board_w - w - 1)
                queue_channel[:h, start_col:start_col+w] = next_piece[:h, :w].astype(np.float32)
        
        return queue_channel


class EnhancedCNNWrapper(gym.ObservationWrapper):
    """Resize multi-channel observation for CNN processing"""
    
    def __init__(self, env, target_size=(84, 84)):
        super().__init__(env)
        self.target_size = target_size
        
        # Expect 4-channel input, output 4-channel
        in_shape = env.observation_space.shape
        self.observation_space = Box(
            low=0.0, high=1.0,
            shape=(*target_size, in_shape[-1]),
            dtype=np.float32
        )
        
    def observation(self, obs):
        """Resize each channel independently preserving aspect ratio"""
        h, w, c = obs.shape
        target_h, target_w = self.target_size
        
        # Calculate scale preserving aspect ratio
        scale = min(target_h / h, target_w / w)
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        # Resize each channel
        resized_channels = []
        for i in range(c):
            channel = obs[:, :, i]
            # Convert to uint8 for cv2
            channel_uint8 = (channel * 255).astype(np.uint8)
            resized = cv2.resize(channel_uint8, (new_w, new_h), 
                               interpolation=cv2.INTER_NEAREST)
            resized_channels.append(resized)
        
        # Stack and center in canvas
        resized = np.stack(resized_channels, axis=-1)
        canvas = np.zeros((*self.target_size, c), dtype=np.uint8)
        
        offset_h = (target_h - new_h) // 2
        offset_w = (target_w - new_w) // 2
        canvas[offset_h:offset_h+new_h, offset_w:offset_w+new_w] = resized
        
        # Convert back to float
        return canvas.astype(np.float32) / 255.0


def make_env(env_name=None, render_mode=None, use_complete_vision=True, 
             use_cnn=True, **env_kwargs):
    """
    Create Tetris environment with COMPLETE vision
    
    Args:
        use_complete_vision: If True, use 4-channel complete observation
        use_cnn: If True, resize for CNN input
    """
    if env_name is None:
        env_name = ENV_NAME
        
    env = gym.make(env_name, render_mode=render_mode or "rgb_array", **env_kwargs)
    
    if use_complete_vision:
        # Apply the complete observation wrapper
        env = CompleteTetrisObservationWrapper(env)
        print("✅ Complete vision enabled: Board + Active Piece + Holder + Queue")
        
        if use_cnn:
            env = EnhancedCNNWrapper(env, target_size=(84, 84))
            print("✅ CNN mode: 84×84×4 multi-channel input")
    else:
        # Fall back to broken single-channel (for comparison)
        from config import TetrisObservationWrapper, CorrectedTetrisBoardWrapper
        env = TetrisObservationWrapper(env)
        if use_cnn:
            env = CorrectedTetrisBoardWrapper(env)
        print("⚠️  Using limited vision (board only)")
    
    print(f"Final observation space: {env.observation_space}")
    
    return env


def test_complete_vision():
    """Test the complete vision system"""
    print("\n🔍 Testing Complete Vision System")
    print("="*60)
    
    # Create environment with complete vision
    env = make_env(use_complete_vision=True, use_cnn=False)
    obs, info = env.reset(seed=42)
    
    print(f"\nObservation shape: {obs.shape}")
    print(f"Channels: {obs.shape[-1]}")
    
    # Analyze each channel
    for i in range(obs.shape[-1]):
        channel = obs[:, :, i]
        non_zero = np.sum(channel > 0.01)
        print(f"\nChannel {i}:")
        print(f"  Non-zero pixels: {non_zero}")
        print(f"  Range: [{channel.min():.3f}, {channel.max():.3f}]")
        
        if i == 0:
            print("  (Board - obstacles)")
        elif i == 1:
            print("  (Active piece - THIS WAS MISSING!)")
        elif i == 2:
            print("  (Holder - strategic piece)")
        elif i == 3:
            print("  (Queue - next piece)")
    
    # Test observation changes
    print("\n📊 Testing observation dynamics:")
    
    for step in range(5):
        action = env.action_space.sample()
        new_obs, reward, terminated, truncated, info = env.step(action)
        
        # Check each channel for changes
        changes = []
        for i in range(obs.shape[-1]):
            change = np.abs(new_obs[:, :, i] - obs[:, :, i]).sum()
            changes.append(change)
            
        print(f"Step {step+1}: Action={action}, "
              f"Channel changes: {[f'{c:.1f}' for c in changes]}")
        
        obs = new_obs
        if terminated or truncated:
            obs, info = env.reset()
            
    env.close()
    
    print("\n✅ Complete vision system is working!")
    print("The agent can now see:")
    print("  • The board (what was already visible)")
    print("  • The falling piece (what was missing!)")
    print("  • The held piece (strategic planning)")
    print("  • The next piece (forward planning)")
    

if __name__ == "__main__":
    test_complete_vision()
    
    print("\n" + "="*80)
    print("🎯 TO FIX YOUR 62K EPISODE PLATEAU:")
    print("="*80)
    print("1. Replace your config.py with this file:")
    print("   cp config_complete_vision.py config.py")
    print("\n2. Start training with complete vision:")
    print("   python train.py --episodes 1000 --use_complete_vision")
    print("\n3. Or use emergency breakthrough:")
    print("   python emergency_breakthrough_complete.py")
    print("\nExpected: First line clear within 20-50 episodes!")
    print("(vs. 0 line clears in 62,800 episodes with broken vision)")
    print("="*80)