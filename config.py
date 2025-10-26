# config.py
"""Configuration for Tetris RL Training - FIXED with 3D observations"""

import numpy as np
import gymnasium as gym

# Import tetris_gymnasium to register environments
import tetris_gymnasium.envs

# ✅ FIXED: Add all variables that train.py expects
ENV_NAME = 'tetris_gymnasium/Tetris'
LR = 0.0001
MAX_EPISODES = 10000
MODEL_DIR = "models/"
LOG_DIR = "logs/"

# Training Configuration
EPISODES = 10000
BATCH_SIZE = 64
GAMMA = 0.99
LEARNING_RATE = 0.0001
MEMORY_SIZE = 100000
MIN_MEMORY_SIZE = 1000
TARGET_UPDATE_FREQUENCY = 1000

# Epsilon (exploration) schedule
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.9999

# Model architecture
HIDDEN_UNITS = [512, 256, 128]

# Environment configuration
ENV_CONFIG = {
    'height': 20,
    'width': 10,
}

# Reward shaping modes
REWARD_SHAPING_MODE = 'balanced'

# Logging
LOG_INTERVAL = 10
SAVE_INTERVAL = 500
MILESTONE_INTERVAL = 1000


def make_env(render_mode="rgb_array", use_complete_vision=True, use_cnn=False):
    """
    Create Tetris environment with complete vision
    
    Args:
        render_mode: Rendering mode ('rgb_array', 'human', None)
        use_complete_vision: If True, wrap to convert dict to array
        use_cnn: Not used, kept for compatibility
    
    Returns:
        Gymnasium environment
    """
    # Create base environment
    env = gym.make(
        ENV_NAME,
        render_mode=render_mode,
        **ENV_CONFIG
    )
    
    print(f"✅ Environment created: {env.spec.id}")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")
    print(f"   Complete vision: {use_complete_vision}")
    
    # Wrap environment to convert dict observations to arrays
    if use_complete_vision:
        env = CompleteVisionWrapper(env)
    
    return env


class CompleteVisionWrapper(gym.ObservationWrapper):
    """
    Wrapper to convert dict observations to 3D array for CNN
    
    Extracts board and adds channel dimension for CNN processing
    Output shape: (20, 10, 1) - height x width x channels
    """
    
    def __init__(self, env):
        super().__init__(env)
        
        # Define new observation space - 3D with 1 channel
        # ✅ FIXED: Changed from (20, 10) to (20, 10, 1) for CNN compatibility
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(20, 10, 1),  # ← Added channel dimension!
            dtype=np.uint8
        )
        
        print("🎯 CompleteVisionWrapper initialized:")
        print(f"   Input space: {env.observation_space}")
        print(f"   Output space: {self.observation_space}")
    
    def observation(self, obs_dict):
        """
        Extract board from observation dict and add channel dimension
        
        Args:
            obs_dict: Dictionary with 'board' and other keys
            
        Returns:
            3D numpy array (20, 10, 1) - board with channel dimension
        """
        if isinstance(obs_dict, dict) and 'board' in obs_dict:
            board = obs_dict['board']
            
            # Extract playable area (20x10)
            if board.shape == (24, 18):
                # Standard tetris-gymnasium format with walls
                # Playable area is in the center
                playable = board[2:22, 4:14]  # Extract 20x10 playable area
            elif board.shape == (20, 10):
                # Already correct size
                playable = board
            elif board.shape[0] >= 20 and board.shape[1] >= 10:
                # Has walls - extract middle 20x10
                playable = board[:20, :10]
            else:
                # Unexpected shape - use as is and hope for the best
                playable = board
            
            # ✅ FIXED: Add channel dimension (20, 10) → (20, 10, 1)
            playable_3d = np.expand_dims((playable > 0).astype(np.uint8), axis=-1)
            
            return playable_3d.astype(np.uint8)
        else:
            # If not a dict, assume it's already the board
            board = np.array(obs_dict)
            # Add channel dimension if needed
            if len(board.shape) == 2:
                board = np.expand_dims(board, axis=-1)
            return board.astype(np.uint8)


def test_environment():
    """Test function to verify environment works"""
    print("\n🧪 Testing environment...")
    
    env = make_env(render_mode="rgb_array", use_complete_vision=True)
    
    # Test reset
    obs, info = env.reset(seed=42)
    print(f"✅ Reset successful")
    print(f"   Observation shape: {obs.shape}")
    print(f"   Observation dtype: {obs.dtype}")
    print(f"   Observation range: [{obs.min()}, {obs.max()}]")
    
    # Verify 3D shape
    assert len(obs.shape) == 3, f"Expected 3D observation, got shape {obs.shape}"
    assert obs.shape == (20, 10, 1), f"Expected (20, 10, 1), got {obs.shape}"
    
    # Test steps
    total_reward = 0
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Verify shape consistency
        assert obs.shape == (20, 10, 1), f"Shape changed to {obs.shape}!"
        
        if terminated or truncated:
            obs, info = env.reset()
            break
    
    print(f"✅ Environment steps work - Total reward: {total_reward:.2f}")
    print(f"✅ Shape consistency verified: all observations are (20, 10, 1)")
    
    env.close()
    return True


if __name__ == "__main__":
    # Test the configuration
    print("="*60)
    print("Testing Tetris Configuration")
    print("="*60)
    
    test_environment()
    
    print("\n✅ Configuration test passed!")
    print(f"\nConfiguration values:")
    print(f"  ENV_NAME: {ENV_NAME}")
    print(f"  LR: {LR}")
    print(f"  GAMMA: {GAMMA}")
    print(f"  BATCH_SIZE: {BATCH_SIZE}")
    print(f"  MAX_EPISODES: {MAX_EPISODES}")
    print(f"  MODEL_DIR: {MODEL_DIR}")
    print(f"  LOG_DIR: {LOG_DIR}")