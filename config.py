# Hyperparameter configuration for Tetris AI

# Environment settings
ENV_NAME = "ALE/Tetris-v5"  # Atari Tetris environment
RENDER_MODE = None  # "human" for visualization, None for training
FRAME_STACK = 4  # Number of frames to stack
PREPROCESS = True  # Apply image preprocessing
REWARD_SHAPING = True  # Apply custom reward shaping

# Training hyperparameters
LR = 1e-4  # Learning rate
GAMMA = 0.99  # Discount factor
BATCH_SIZE = 32  # Batch size for training
REPLAY_BUFFER_SIZE = 100000  # Experience replay buffer size
MIN_REPLAY_SIZE = 10000  # Minimum buffer size before training starts
EPSILON_START = 1.0  # Initial exploration rate
EPSILON_END = 0.1  # Final exploration rate
EPSILON_DECAY = 100000  # Decay steps for epsilon
TARGET_UPDATE_FREQ = 1000  # Steps between target network updates

# Training settings
MAX_EPISODES = 1000  # Maximum number of episodes
MAX_STEPS_PER_EPISODE = 10000  # Maximum steps per episode
SAVE_FREQUENCY = 50  # Save model every N episodes
LOG_FREQUENCY = 10  # Log metrics every N episodes

# Model architecture
CONV_CHANNELS = [32, 64, 64]  # Convolutional layer channels
CONV_KERNEL_SIZES = [8, 4, 3]  # Kernel sizes for conv layers
CONV_STRIDES = [4, 2, 1]  # Strides for conv layers
HIDDEN_SIZE = 512  # Hidden layer size

# Directories
MODEL_DIR = "models/"
LOG_DIR = "logs/"
CHECKPOINT_DIR = "checkpoints/"

# Device settings
DEVICE = "cuda"  # "cuda" or "cpu"

# Random seed for reproducibility
SEED = 42
