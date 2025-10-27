# config.py
"""Configuration for Tetris RL Training - FIXED with action discovery"""

import numpy as np
import gymnasium as gym
import tetris_gymnasium.envs

# Environment name
ENV_NAME = 'tetris_gymnasium/Tetris'

# Training Configuration
LR = 0.0001
MAX_EPISODES = 10000
MODEL_DIR = "models/"
LOG_DIR = "logs/"
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

# ACTION MAPPINGS FOR TETRIS-GYMNASIUM
# These are the typical action IDs - will be verified at runtime
ACTION_NOOP = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2
ACTION_DOWN = 3
ACTION_ROTATE_CW = 4
ACTION_ROTATE_CCW = 5
ACTION_HARD_DROP = 6
ACTION_SWAP = 7 

# Action meanings will be discovered and updated at runtime
ACTION_MEANINGS = None

#Added by GPT def _board_from_obs(obs)
def _board_from_obs(obs):
    # Plocka ut 2D-board ur obs (oavsett dict eller array)
    board = obs.get('board') if isinstance(obs, dict) else obs
    if board is None:
        board = np.zeros((20, 10), dtype=np.uint8)
    if len(board.shape) == 3:
        board = board[:, :, 0]
    return board

def discover_action_meanings(env):
    """Discover the actual action meanings from the environment"""
    global ACTION_MEANINGS, ACTION_LEFT, ACTION_RIGHT, ACTION_DOWN
    global ACTION_ROTATE_CW, ACTION_ROTATE_CCW, ACTION_HARD_DROP, ACTION_NOOP, ACTION_SWAP
    
    # Try to get action meanings from environment
    try:
        if hasattr(env, 'get_action_meanings'):
            meanings = env.get_action_meanings()
        elif hasattr(env.unwrapped, 'get_action_meanings'):
            meanings = env.unwrapped.get_action_meanings()
        else:
            meanings = None
            
        if meanings:
            ACTION_MEANINGS = meanings
            print(f"✅ Discovered {len(meanings)} actions from environment")
            
            # Update action constants
            for i, meaning in enumerate(meanings):
                if 'LEFT' in meaning.upper():
                    ACTION_LEFT = i
                elif 'RIGHT' in meaning.upper():
                    ACTION_RIGHT = i
                elif 'DOWN' in meaning.upper() and 'HARD' not in meaning.upper():
                    ACTION_DOWN = i
                elif 'ROTATE' in meaning.upper() and ('CW' in meaning.upper() or 'CLOCKWISE' in meaning.upper()):
                    ACTION_ROTATE_CW = i
                elif 'ROTATE' in meaning.upper() and ('CCW' in meaning.upper() or 'COUNTER' in meaning.upper()):
                    ACTION_ROTATE_CCW = i
                elif 'HARD' in meaning.upper() and 'DROP' in meaning.upper():
                    ACTION_HARD_DROP = i
                elif 'SWAP' in meaning.upper() or 'HOLD' in meaning.upper():
                    ACTION_SWAP = i
                elif 'NOOP' in meaning.upper() or 'NOTHING' in meaning.upper():
                    ACTION_NOOP = i
                    
            return ACTION_MEANINGS
            
    except Exception as e:
        print(f"⚠️  Could not get action meanings: {e}")
    
    # Fallback to standard Tetris actions (INCLUDING ACTION 7)
    print("   Using standard Tetris action mapping")
    ACTION_MEANINGS = [
        'NOOP',       # 0
        'LEFT',       # 1
        'RIGHT',      # 2
        'DOWN',       # 3
        'ROTATE_CW',  # 4
        'ROTATE_CCW', # 5
        'HARD_DROP',  # 6
        'SWAP'        # 7 <-- ADD THIS
    ]
    
    ACTION_NOOP = 0
    ACTION_LEFT = 1
    ACTION_RIGHT = 2
    ACTION_DOWN = 3
    ACTION_ROTATE_CW = 4
    ACTION_ROTATE_CCW = 5
    ACTION_HARD_DROP = 6
    ACTION_SWAP = 7  # <-- ADD THIS
    
    return ACTION_MEANINGS

def make_env(render_mode="rgb_array", use_complete_vision=True, use_cnn=False):
    """Create Tetris environment with complete vision"""
    # Create base environment
    env = gym.make(
        ENV_NAME,
        render_mode=render_mode,
        **ENV_CONFIG
    )
    
    print(f"✅ Environment created: {env.spec.id}")
    print(f"   Action space: {env.action_space} (n={env.action_space.n})")
    
    # Discover action meanings
    discover_action_meanings(env)
    
    # Wrap environment to convert dict observations to arrays
    if use_complete_vision:
        env = CompleteVisionWrapper(env)
        print(f"   Observation space: {env.observation_space}")
    
    return env


class CompleteVisionWrapper(gym.ObservationWrapper):
    """Wrapper to convert dict observations to 3D array for CNN"""
    
    def __init__(self, env):
        super().__init__(env)
        
        # Define new observation space - 3D with 1 channel
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(20, 10, 1),
            dtype=np.uint8
        )
    
    def observation(self, obs):
        """Convert dict observation to 3D array"""
        if isinstance(obs, dict):
            # Extract board from dict observation
            board = obs.get('board', np.zeros((20, 10)))
            
            # Ensure binary values (0 or 1)
            board = (board > 0).astype(np.uint8) * 255
            
            # Add channel dimension
            board = board[:, :, np.newaxis]
            
            return board
        else:
            # Already an array, just ensure correct shape
            if len(obs.shape) == 2:
                obs = obs[:, :, np.newaxis]
            return obs


def test_environment():
    """Test that environment is working correctly"""
    print("\n" + "="*60)
    print("Testing Tetris Environment")
    print("="*60)
    
    env = make_env(use_complete_vision=True)
    obs, info = env.reset()
    
    print(f"\n✅ Environment test:")
    print(f"   Observation shape: {obs.shape}")
    print(f"   Expected: (20, 10, 1)")
    
    # Test a few steps with different actions
    test_actions = [ACTION_LEFT, ACTION_RIGHT, ACTION_HARD_DROP]
    for action in test_actions:
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()
    print(f"✅ Environment works correctly!")
    
    return True


if __name__ == "__main__":
    test_environment()
    
    print("\n✅ Configuration ready!")
    print(f"\nAction mappings discovered:")
    print(f"  LEFT: {ACTION_LEFT}")
    print(f"  RIGHT: {ACTION_RIGHT}")
    print(f"  HARD_DROP: {ACTION_HARD_DROP}")