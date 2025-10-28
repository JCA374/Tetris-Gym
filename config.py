# config.py

import numpy as np
import gymnasium as gym
import tetris_gymnasium.envs

from src.env_wrapper import CompleteVisionWrapper 

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
import numpy as np

def crop_playfield(board):
    """
    Tar en binär/uint8 board med väggar (t.ex. 24x18) och returnerar en 20x10 spelplan.
    Upptäcker väggkolumner/väggrader automatiskt genom att leta efter kolumner/rader som
    nästan alltid är fyllda (”vägg”) och tar den största 10-breda inre regionen.
    """
    arr = np.array(board)
    if arr.ndim == 3:  # H, W, C
        arr2d = arr[..., 0]
    else:
        arr2d = arr

    H, W = arr2d.shape
    # Snabb exit om redan 20x10
    if (H, W) == (20, 10):
        return arr

    # Heuristik: väggar är kolumner/rader med hög andel fyllda celler.
    # Skapa mask över "fylld" (värde > 0)
    filled = (arr2d > 0).astype(np.uint8)

    col_fill_ratio = filled.mean(axis=0)  # shape (W,)
    row_fill_ratio = filled.mean(axis=1)  # shape (H,)

    # Väggkolumner ~ nästan alltid fyllda (t.ex. > 0.9)
    wall_cols = np.where(col_fill_ratio > 0.9)[0]
    wall_rows = np.where(row_fill_ratio > 0.9)[0]

    # Inre kolumner = de som INTE är väggar
    inner_cols = [c for c in range(W) if c not in set(wall_cols)]
    inner_rows = [r for r in range(H) if r not in set(wall_rows)]

    # Om inre fönster är större än 10/20, ta mittsektionen (vanligt i Tetris: 24x18 → ta 20x10 i mitten)
    def center_slice(indices, want_len):
        if len(indices) <= want_len:
            return indices
        start = (len(indices) - want_len) // 2
        return indices[start:start + want_len]

    inner_cols = center_slice(inner_cols, 10)
    inner_rows = center_slice(inner_rows, 20)

    cropped2d = arr2d[np.ix_(inner_rows, inner_cols)]

    # Lägg ev. tillbaka kanal
    if arr.ndim == 3:
        cropped = np.expand_dims(cropped2d.astype(np.uint8), axis=-1)
    else:
        cropped = cropped2d.astype(np.uint8)

    return cropped

def make_env(render_mode="rgb_array", use_complete_vision=True, use_cnn=False):
    env = gym.make(ENV_NAME, render_mode=render_mode, **ENV_CONFIG)
    print(f"✅ Environment created: {env.spec.id}")
    print(f"   Action space: {env.action_space} (n={env.action_space.n})")

    discover_action_meanings(env)

    if use_complete_vision:
        env = CompleteVisionWrapper(env)
        print(f"   Observation space: {env.observation_space}")

    return env

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

def get_action_meanings():
    """
    Return action meanings as a dictionary mapping action_id -> meaning string
    Must be called after discover_action_meanings(env) has been called
    
    Returns:
        dict: {action_id: action_meaning_string}
    """
    global ACTION_MEANINGS
    
    if ACTION_MEANINGS is None:
        # Return default mappings if not yet discovered
        return {
            0: 'NOOP',
            1: 'LEFT',
            2: 'RIGHT',
            3: 'DOWN',
            4: 'ROTATE_CW',
            5: 'ROTATE_CCW',
            6: 'HARD_DROP',
            7: 'SWAP'
        }
    
    # Convert list to dict
    return {i: meaning for i, meaning in enumerate(ACTION_MEANINGS)}


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

def make_env(render_mode="rgb_array", use_complete_vision=True, use_cnn=False):
    env = gym.make(ENV_NAME, render_mode=render_mode, **ENV_CONFIG)
    print(f"✅ Environment created: {env.spec.id}")
    print(f"   Action space: {env.action_space} (n={env.action_space.n})")

    discover_action_meanings(env)

    if use_complete_vision:
        env = CompleteVisionWrapper(env)
        print(f"   Observation space: {env.observation_space}")

    return env
