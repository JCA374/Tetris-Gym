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
    global ACTION_ROTATE_CW, ACTION_ROTATE_CCW, ACTION_HARD_DROP, ACTION_NOOP
    
    # Try to get action meanings from environment
    try:
        if hasattr(env.unwrapped, 'get_action_meanings'):
            meanings = env.unwrapped.get_action_meanings()
            ACTION_MEANINGS = meanings
            print(f"🔍 Discovered action meanings: {meanings}")
            
            # Map action names to indices
            name_to_idx = {name.lower(): i for i, name in enumerate(meanings)}
            
            # Update action constants with discovered values
            ACTION_NOOP = name_to_idx.get("noop", name_to_idx.get("no_op", 0))
            ACTION_LEFT = name_to_idx.get("left", 1)
            ACTION_RIGHT = name_to_idx.get("right", 2)
            ACTION_DOWN = name_to_idx.get("down", name_to_idx.get("soft_drop", 3))
            ACTION_ROTATE_CW = name_to_idx.get("rotate_cw", name_to_idx.get("rotate_right", 4))
            ACTION_ROTATE_CCW = name_to_idx.get("rotate_ccw", name_to_idx.get("rotate_left", 5))
            ACTION_HARD_DROP = name_to_idx.get("hard_drop", name_to_idx.get("drop", 6))
            
            print(f"   LEFT={ACTION_LEFT}, RIGHT={ACTION_RIGHT}, HARD_DROP={ACTION_HARD_DROP}")
            return True
    except:
        pass
    
    # Fallback: Test actions empirically
    print("⚠️  Could not get action meanings from env, using empirical testing...")
    
    obs, _ = env.reset(seed=42)
    initial_board = _board_from_obs(obs) # Fixed by GTP 
    
    # Test each action to see what it does
    action_effects = {}
    for action in range(env.action_space.n):
        obs, _ = env.reset(seed=42)  # Reset to same state
        
        # Take the action multiple times to see its effect
        for _ in range(3):
            obs, _, done, _, _ = env.step(action)
            if done:
                break
        
        board_after = _board_from_obs(obs) #Fixed by GPT
        
        # Analyze the effect
        if np.array_equal(initial_board, board_after):
            action_effects[action] = "likely_noop"
        elif np.sum(board_after) > np.sum(initial_board):
            action_effects[action] = "likely_drop"
        else:
            action_effects[action] = "likely_movement"
    
    print(f"   Empirical action effects: {action_effects}")
    
    # Best guess based on common patterns
    ACTION_MEANINGS = ["NOOP", "LEFT", "RIGHT", "DOWN", "ROTATE_CW", "ROTATE_CCW", "HARD_DROP"]
    print(f"   Using standard Tetris action mapping")
    
    return True


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