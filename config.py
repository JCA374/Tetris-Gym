# config.py
"""Configuration for Tetris RL Training with Complete Vision"""

import numpy as np
import gymnasium as gym

# Import tetris_gymnasium to register environments
import tetris_gymnasium.envs  # This registers the Tetris environment

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
    # obs_type removed - not supported by tetris-gymnasium
}

# Reward shaping modes: 'aggressive', 'positive', 'balanced'
REWARD_SHAPING_MODE = 'balanced'

# Logging
LOG_INTERVAL = 10
SAVE_INTERVAL = 500
MILESTONE_INTERVAL = 1000

def make_env(render_mode="rgb_array", use_complete_vision=True):
    """
    Create Tetris environment with complete vision
    
    Args:
        render_mode: Rendering mode ('rgb_array', 'human', None)
        use_complete_vision: If True, enable complete vision with 4-channel observations
    
    Returns:
        Gymnasium environment
    """
    # Create base environment
    env = gym.make(
        'tetris_gymnasium/Tetris',
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
    Wrapper to convert dict observations to playable area only.
    
    CRITICAL FIX: tetris-gymnasium adds walls around the playable area!
    - Board is 24x18 but playable area is only 20x10 (middle section)
    - Walls at columns 0-3 and 14-17, bottom rows 20-23
    - These walls make line clearing IMPOSSIBLE!
    
    This wrapper extracts ONLY the playable 20x10 area.
    """
    
    def __init__(self, env):
        super().__init__(env)
        
        base_space = env.observation_space
        
        print(f"   Original observation: {type(base_space)}")
        
        if isinstance(base_space, gym.spaces.Dict):
            # The board includes walls - we need to extract playable area
            # Full board is 24x18, playable is 20x10
            # Walls: columns 0-3 (left), 14-17 (right), rows 20-23 (bottom)
            
            playable_height = 20
            playable_width = 10
            
            # Create observation space for playable area only
            self.observation_space = gym.spaces.Box(
                low=0, high=1,
                shape=(playable_height, playable_width, 2),  # 2 channels
                dtype=np.float32
            )
            
            self.use_dict_obs = True
            self.wall_left = 4  # Skip first 4 columns
            self.wall_right = 14  # Stop at column 14
            self.wall_bottom = 20  # Only use rows 0-19
            
            print(f"   ✅ Extracting playable area: {playable_height}x{playable_width}")
            print(f"   ✅ Removing walls (cols 0-3, 14-17, rows 20-23)")
            print(f"   ✅ Final shape: {self.observation_space.shape}")
        else:
            self.observation_space = base_space
            self.use_dict_obs = False
    
    def observation(self, obs):
        """
        Extract playable area from dict observation.
        """
        if not self.use_dict_obs:
            obs = np.array(obs, dtype=np.float32)
            if obs.max() > 1.0:
                obs = obs / obs.max()
            return obs
        
        # Extract board and active piece mask
        board_full = obs['board'].astype(np.float32)
        active_full = obs['active_tetromino_mask'].astype(np.float32)
        
        # Extract ONLY the playable area (remove walls)
        # Playable area: rows 0-19, columns 4-13
        board_playable = board_full[:self.wall_bottom, self.wall_left:self.wall_right]
        active_playable = active_full[:self.wall_bottom, self.wall_left:self.wall_right]
        
        # Normalize board
        if board_playable.max() > 1.0:
            board_playable = board_playable / board_playable.max()
        
        # Stack channels: board + active piece
        combined = np.stack([board_playable, active_playable], axis=-1)
        
        return combined


def print_config():
    """Print current configuration"""
    print("\n" + "="*80)
    print("TETRIS RL TRAINING CONFIGURATION")
    print("="*80)
    print(f"Episodes:              {EPISODES}")
    print(f"Batch size:            {BATCH_SIZE}")
    print(f"Gamma (discount):      {GAMMA}")
    print(f"Learning rate:         {LEARNING_RATE}")
    print(f"Memory size:           {MEMORY_SIZE}")
    print(f"Epsilon start:         {EPSILON_START}")
    print(f"Epsilon end:           {EPSILON_END}")
    print(f"Epsilon decay:         {EPSILON_DECAY}")
    print(f"Hidden units:          {HIDDEN_UNITS}")
    print(f"Reward shaping mode:   {REWARD_SHAPING_MODE}")
    print(f"Board size:            {ENV_CONFIG['height']}x{ENV_CONFIG['width']}")
    print("="*80 + "\n")
