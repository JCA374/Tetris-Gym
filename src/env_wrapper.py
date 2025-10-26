# src/env_wrapper.py
import numpy as np
import gymnasium as gym
import tetris_gymnasium  # This registers the environment

class CompleteVisionWrapper(gym.ObservationWrapper):
    """
    Wrapper that converts the dictionary observation to a simple board array
    suitable for DQN input.
    """
    def __init__(self, env):
        super().__init__(env)
        # Define the observation space as a 20x10x1 array
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(20, 10, 1), dtype=np.uint8
        )
        print(f"🎯 CompleteVisionWrapper initialized:")
        print(f"   Input space: {env.observation_space}")
        print(f"   Output space: {self.observation_space}")
    
    def observation(self, obs):
        """
        Convert dictionary observation to board array
        """
        if isinstance(obs, dict) and 'board' in obs:
            # Get the board from the observation
            full_board = obs['board']
            
            # Extract the playable area (20x10) from the full board
            if full_board.shape == (24, 18):
                # Tetris Gymnasium uses 24x18 with padding
                board = full_board[2:22, 4:14]  # Extract 20x10 playable area
            elif full_board.shape == (20, 10):
                board = full_board
            else:
                # Fallback: try to extract 20x10 from whatever shape
                h, w = full_board.shape[:2]
                h_start = max(0, (h - 20) // 2)
                w_start = max(0, (w - 10) // 2)
                board = full_board[h_start:h_start+20, w_start:w_start+10]
                
                # Ensure it's exactly 20x10
                if board.shape != (20, 10):
                    target = np.zeros((20, 10), dtype=np.uint8)
                    h, w = min(20, board.shape[0]), min(10, board.shape[1])
                    target[:h, :w] = board[:h, :w]
                    board = target
            
            # Convert to binary (0 or 1) instead of piece IDs
            board = (board > 0).astype(np.uint8)
            
            # Add channel dimension
            board = np.expand_dims(board, axis=-1)  # Add channel dimension
            
            return board
        else:
            # Return empty board if observation format is unexpected
            return np.zeros((20, 10, 1), dtype=np.uint8)

def create_tetris_env(complete_vision=True):
    """
    Create a Tetris environment with appropriate wrappers
    """
    try:
        # Import and create the Tetris environment directly
        from tetris_gymnasium.envs import Tetris
        env = Tetris(render_mode=None)
        
        print(f"✅ Environment created: Tetris")
        print(f"   Observation space: {env.observation_space}")
        print(f"   Action space: {env.action_space}")
        print(f"   Complete vision: {complete_vision}")
        
        # Apply wrapper if using complete vision
        if complete_vision:
            env = CompleteVisionWrapper(env)
        
        return env
    except Exception as e:
        print(f"❌ Error creating environment: {e}")
        raise