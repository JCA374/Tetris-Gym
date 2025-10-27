# DOCUMENTATION.md

# Tetris Reinforcement Learning Project - Complete Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Component Documentation](#component-documentation)
4. [Data Flow](#data-flow)
5. [Training Pipeline](#training-pipeline)
6. [Key Algorithms](#key-algorithms)
7. [Integration Logic](#integration-logic)

---

## 1. Project Overview

### Purpose
This project implements a Deep Q-Network (DQN) reinforcement learning agent that learns to play Tetris using the `tetris-gymnasium` environment. The agent learns through trial and error, optimizing its actions to maximize game score and clear lines.

### Key Technologies
- **Environment**: Tetris Gymnasium (`tetris-gymnasium==0.2.1`)
- **Deep Learning**: PyTorch for neural networks
- **RL Algorithm**: Deep Q-Network (DQN) with experience replay
- **Visualization**: Matplotlib for training curves
- **Logging**: CSV and JSON for metrics tracking

### Project Goals
1. Train an agent to play Tetris autonomously
2. Learn effective piece placement strategies
3. Maximize lines cleared and game duration
4. Explore different reward shaping techniques
5. Provide comprehensive training analytics

---

## 2. System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐      ┌──────────────┐                    │
│  │   Config     │──────▶│     Env      │                    │
│  │   Setup      │      │  (Tetris)    │                    │
│  └──────────────┘      └──────┬───────┘                    │
│                               │                             │
│                               ▼                             │
│                        ┌──────────────┐                     │
│                        │  Observation │                     │
│                        └──────┬───────┘                     │
│                               │                             │
│                               ▼                             │
│  ┌──────────────┐      ┌──────────────┐                    │
│  │    Agent     │◀─────┤  Preprocessor│                    │
│  │   (DQN)      │      └──────────────┘                    │
│  └──────┬───────┘                                           │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────┐      ┌──────────────┐                    │
│  │  Q-Network   │      │ Target Net   │                    │
│  │  (Policy)    │      │              │                    │
│  └──────┬───────┘      └──────────────┘                    │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────┐      ┌──────────────┐                    │
│  │    Action    │──────▶│     Env      │                    │
│  │  Selection   │      │   Step       │                    │
│  └──────────────┘      └──────┬───────┘                    │
│                               │                             │
│                               ▼                             │
│                        ┌──────────────┐                     │
│                        │    Reward    │                     │
│                        │   Shaping    │                     │
│                        └──────┬───────┘                     │
│                               │                             │
│                               ▼                             │
│  ┌──────────────┐      ┌──────────────┐                    │
│  │   Replay     │◀─────┤  Experience  │                    │
│  │   Buffer     │      │   Storage    │                    │
│  └──────┬───────┘      └──────────────┘                    │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────┐      ┌──────────────┐                    │
│  │  Training    │──────▶│   Logger     │                    │
│  │   Loop       │      │              │                    │
│  └──────────────┘      └──────────────┘                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Component Hierarchy

```
tetris-rl-project/
│
├── config.py                  # Environment configuration & action discovery
├── train.py                   # Main training orchestration
├── evaluate.py                # Model evaluation & visualization
├── requirements.txt           # Python dependencies
│
├── src/                       # Core implementation modules
│   ├── agent.py              # DQN agent with ε-greedy exploration
│   ├── model.py              # Neural network architectures (Conv/MLP)
│   ├── reward_shaping.py     # Custom reward functions
│   ├── training_logger.py    # Metrics tracking & visualization
│   └── utils.py              # Helper functions
│
├── tests/                     # Testing & diagnostic scripts
│   ├── test_setup.py         # Environment validation
│   ├── test_observation_integrity.py  # Data integrity checks
│   └── test_reward_helpers.py # Reward function validation
│
├── models/                    # Saved model checkpoints (generated)
└── logs/                      # Training logs & metrics (generated)
```

---

## 3. Component Documentation

### 3.1 Configuration Module (`config.py`)

**Purpose**: Centralized configuration and environment setup with dynamic action discovery.

**Key Functions**:
- `discover_action_meanings(env)`: Dynamically discovers Tetris actions from environment
- `make_env()`: Creates and configures Tetris Gymnasium environment
- `_board_from_obs(obs)`: Extracts board state from observations

**Key Constants**:
```python
# Training hyperparameters
LR = 0.0001                    # Learning rate
GAMMA = 0.99                   # Discount factor
BATCH_SIZE = 64                # Training batch size
MEMORY_SIZE = 100000           # Replay buffer capacity
EPSILON_START = 1.0            # Initial exploration rate
EPSILON_END = 0.05             # Minimum exploration rate
EPSILON_DECAY = 0.9999         # Exploration decay rate

# Model architecture
HIDDEN_UNITS = [512, 256, 128] # Neural network layer sizes

# Environment settings
ENV_CONFIG = {
    'height': 20,              # Board height
    'width': 10                # Board width
}
```

**Action Discovery Logic**:
The environment dynamically discovers available actions to ensure compatibility across different Tetris Gymnasium versions:

```python
def discover_action_meanings(env):
    """
    Discovers action mappings from environment.
    Handles different Tetris Gymnasium versions.
    
    Returns: List of action meanings (e.g., ['NOOP', 'LEFT', 'RIGHT', ...])
    """
    # Try multiple methods to get action meanings
    # 1. Check env.get_action_meanings()
    # 2. Check env.unwrapped.get_action_meanings()
    # 3. Fall back to standard Tetris actions
```

**Critical Features**:
- **Version Compatibility**: Handles multiple Tetris Gymnasium versions
- **Action Validation**: Ensures correct action-to-ID mappings
- **Board Extraction**: Normalizes observations to [0, 1] range
- **Environment Wrapping**: Adds preprocessing for observation flattening

---

### 3.2 Agent Module (`src/agent.py`)

**Purpose**: Implements the DQN reinforcement learning agent with experience replay.

**Class: Agent**

**Core Attributes**:
```python
self.q_network          # Main policy network
self.target_network     # Target network (stabilizes training)
self.optimizer          # Adam optimizer
self.memory             # Experience replay buffer (deque)
self.epsilon            # Current exploration rate
self.steps_done         # Total training steps
self.episodes_done      # Total episodes completed
```

**Key Methods**:

1. **`select_action(state, training=True)`**
   - Implements ε-greedy action selection
   - During training: explores with probability ε
   - During evaluation: always exploits (greedy)
   - Returns: Action index (int)

2. **`remember(state, action, reward, next_state, done)`**
   - Stores experiences in replay buffer
   - Buffer type: Deque with max size
   - Experience tuple: (s, a, r, s', done)

3. **`learn()`**
   - Samples batch from replay buffer
   - Computes Q-value targets using target network
   - Performs gradient descent on Q-network
   - Updates target network periodically
   - Returns: Training metrics (loss, etc.)

4. **`update_epsilon(episode)`**
   - Decays exploration rate
   - Methods: exponential, linear, adaptive
   - Ensures minimum epsilon (prevents purely greedy)

5. **`save_checkpoint(filename, model_dir)`**
   - Saves complete agent state
   - Includes: networks, optimizer, epsilon, step count
   - Format: PyTorch .pth file

6. **`load_checkpoint(filename, model_dir)`**
   - Restores agent from saved state
   - Enables training resumption
   - Returns: True if successful

**ε-Greedy Exploration Strategy**:
```
if random() < epsilon:
    action = random_action()  # Explore
else:
    action = argmax(Q(state)) # Exploit
```

**Experience Replay Logic**:
```
1. Store experience: (s, a, r, s', done) → buffer
2. Sample random batch: {(s_i, a_i, r_i, s'_i, done_i)}
3. Compute targets: y_i = r_i + γ * max Q_target(s'_i) * (1 - done_i)
4. Update Q-network: minimize MSE(Q(s_i, a_i), y_i)
5. Periodically: Q_target ← Q_policy
```

---

### 3.3 Model Module (`src/model.py`)

**Purpose**: Neural network architectures for Q-value approximation.

**Key Functions**:

1. **`create_model(obs_space, action_space, model_type)`**
   - Factory function for creating models
   - Supports: "dqn" (ConvDQN), "mlp" (MLPDQN), "dueling_dqn"
   - Returns: PyTorch nn.Module

2. **`_infer_input_shape(obs_space)`**
   - Infers correct input dimensions from observation space
   - Handles: (H,W,C) and (C,H,W) formats
   - Returns: (C, H, W) tuple

3. **`_to_nchw(x)`**
   - Converts tensors to NCHW format (PyTorch standard)
   - Handles various input formats
   - Returns: Properly formatted tensor

**Model Architectures**:

**A. ConvDQN (Convolutional DQN)**
```python
Input: (Batch, Channels, Height, Width)
│
├─ Conv2d(C, 32, kernel=3, padding=1) + ReLU
├─ Conv2d(32, 64, kernel=3, padding=1) + ReLU
├─ AdaptiveAvgPool2d(20, 10)  # Normalize spatial dimensions
├─ Flatten() → 12,800 features
│
├─ Linear(12800, 256) + ReLU
├─ Linear(256, n_actions)
│
Output: Q-values for each action
```

**B. MLPDQN (Multi-Layer Perceptron)**
```python
Input: (Batch, Features)
│
├─ Flatten()
├─ Linear(in_dim, 512) + ReLU
├─ Linear(512, 256) + ReLU
├─ Linear(256, n_actions)
│
Output: Q-values for each action
```

**Architecture Selection**:
- **ConvDQN**: Used for spatial board representations (default)
- **MLPDQN**: Used for flattened feature vectors
- **Dueling DQN**: Separates value and advantage streams (future work)

**Input Normalization**:
- Board values normalized to [0, 1]
- Empty cells: 0
- Filled cells: 1
- Prevents gradient instability

---

### 3.4 Reward Shaping Module (`src/reward_shaping.py`)

**Purpose**: Custom reward functions to guide learning beyond sparse game rewards.

**Core Concept**: 
Tetris provides sparse rewards (points only when lines clear). Reward shaping adds intermediate feedback to accelerate learning.

**Key Functions**:

1. **`extract_board_from_obs(obs)`**
   - Extracts 2D board from observation dict
   - Normalizes values to [0, 1]
   - Returns: numpy array (H, W)

2. **`get_column_heights(board)`**
   - Calculates height of each column
   - Returns: list of 10 heights

3. **`count_holes(board)`**
   - Counts empty cells with filled cells above
   - Indicator of poor placement
   - Returns: hole count (int)

4. **`calculate_bumpiness(heights)`**
   - Measures surface irregularity
   - Sum of absolute height differences between adjacent columns
   - Returns: bumpiness score (int)

5. **`horizontal_distribution(board)`**
   - Rewards spreading pieces across columns
   - Prevents clustering in center
   - Returns: distribution bonus (float)

**Reward Shaping Functions**:

**A. Balanced Reward Shaping** (Recommended)
```python
def balanced_reward_shaping(obs, action, reward, done, info):
    """
    Balanced approach combining multiple factors:
    - Line clearing: +500 per line
    - Holes penalty: -1 per hole
    - Height penalty: -0.5 per unit of max height
    - Bumpiness penalty: -0.3 per bumpiness unit
    - Distribution bonus: +10 for good spread
    - Death penalty: -50
    """
```

**B. Aggressive Reward Shaping**
```python
def aggressive_reward_shaping(obs, action, reward, done, info):
    """
    Strong penalties for bad play:
    - Holes: -3 each
    - Height: -1.0 per unit
    - Bumpiness: -0.5 per unit
    - Death: -100
    """
```

**C. Positive Reward Shaping**
```python
def positive_reward_shaping(obs, action, reward, done, info):
    """
    Focuses on positive reinforcement:
    - Lines: +1000 each
    - Low height bonus: +5
    - Smooth surface bonus: +3
    - Distribution bonus: +15
    """
```

**Reward Formula (Balanced)**:
```
Total Reward = Base Reward 
             + (Lines Cleared × 500)
             - (Holes × 1)
             - (Max Height × 0.5)
             - (Bumpiness × 0.3)
             + (Distribution Bonus)
             - (Death Penalty)
```

**Critical Design Decisions**:
- **Line Clearing**: Largest positive reward (primary objective)
- **Holes**: Strong negative signal (hard to recover from)
- **Height Management**: Encourages keeping board low
- **Bumpiness**: Promotes smooth, manageable surface
- **Distribution**: Prevents over-clustering (new feature)

---

### 3.5 Training Logger Module (`src/training_logger.py`)

**Purpose**: Comprehensive metrics tracking, logging, and visualization.

**Class: TrainingLogger**

**Key Attributes**:
```python
self.experiment_dir     # Unique folder for this experiment
self.metrics_file       # JSON file with all metrics
self.csv_file          # CSV file for episode data
self.plot_file         # PNG file for training curves
self.episode_data      # List of episode records
```

**Key Methods**:

1. **`log_episode(...)`**
   - Records episode metrics
   - Calculates moving averages
   - Writes to CSV file
   - Tracks: reward, steps, lines, epsilon

2. **`plot_progress()`**
   - Creates 6-panel visualization:
     * Episode rewards (with moving average)
     * Lines cleared per episode
     * Steps per episode
     * Epsilon decay curve
     * Cumulative lines cleared
     * Lines per 100 episodes (performance over time)

3. **`save_logs()`**
   - Writes comprehensive JSON with:
     * Experiment config
     * All episode data
     * Summary statistics

4. **`_get_summary()`**
   - Computes summary statistics:
     * Best/worst/average rewards
     * Total lines cleared
     * First line-clear episode
     * Recent 100-episode performance

**Logged Metrics**:
```python
{
    'episode': int,              # Episode number
    'reward': float,             # Total episode reward
    'steps': int,                # Steps in episode
    'epsilon': float,            # Current exploration rate
    'lines': int,                # Lines cleared this episode
    'total_lines': int,          # Cumulative lines cleared
    'avg_reward': float,         # MA-100 reward
    'avg_steps': float,          # MA-100 steps
    'avg_lines': float,          # MA-100 lines
    'timestamp': str,            # ISO timestamp
    'original_reward': float,    # Pre-shaping reward (optional)
    'shaped_reward_used': bool   # Whether shaping applied
}
```

**Visualization Output**:
- High-resolution PNG (150 DPI)
- 6-panel grid layout
- Raw data + moving averages
- Clear labels and legends
- Saved to experiment directory

---

### 3.6 Utility Module (`src/utils.py`)

**Purpose**: Helper functions for file I/O, plotting, and misc operations.

**Key Functions**:

1. **`make_dir(path)`**
   - Creates directory if doesn't exist
   - Returns: path string

2. **`save_json(data, filepath)`** & **`load_json(filepath)`**
   - JSON file operations
   - Handles serialization/deserialization

3. **`plot_training_curves(episode_data, save_path)`**
   - Creates 2x2 subplot grid:
     * Rewards
     * Lines cleared
     * Steps per episode
     * Epsilon decay

4. **`moving_average(values, window)`**
   - Computes rolling average
   - Uses numpy convolution
   - Smooths noisy training curves

5. **`format_time(seconds)`**
   - Converts seconds to human-readable format
   - Returns: "1.5m" or "2.3h" etc.

6. **`format_large_number(num)`**
   - Formats large numbers with K/M suffixes
   - Example: 1500 → "1.5K"

---

## 4. Data Flow

### 4.1 Observation Flow

```
Tetris Environment
       ↓
Raw Observation (dict or array)
       ↓
extract_board_from_obs()
       ↓
Normalized Board [0, 1]
       ↓
Flatten (if needed)
       ↓
Agent Preprocessing
       ↓
Q-Network Input
```

**Observation Structure**:
```python
# Tetris Gymnasium returns:
obs = {
    'board': np.array(shape=(20, 10), dtype=uint8),  # Main board
    'holder': int,                                    # Held piece
    'queue': np.array(shape=(5,), dtype=uint8)      # Next pieces
}

# After preprocessing:
flattened_obs = np.array(shape=(944,), dtype=float32)
# 944 = 20*10 (board) + 1 (holder) + 5 (queue) + ...
```

### 4.2 Training Loop Flow

```
START EPISODE
    ↓
Reset Environment → obs
    ↓
    ┌─────────────────────┐
    │   EPISODE LOOP      │
    │                     │
    │ 1. Select Action    │
    │    (ε-greedy)       │
    │         ↓           │
    │ 2. Execute Action   │
    │    in Environment   │
    │         ↓           │
    │ 3. Get Reward       │
    │    & Next State     │
    │         ↓           │
    │ 4. Apply Reward     │
    │    Shaping          │
    │         ↓           │
    │ 5. Store in         │
    │    Replay Buffer    │
    │         ↓           │
    │ 6. Sample Batch &   │
    │    Train Q-Network  │
    │         ↓           │
    │ 7. Update Target    │
    │    Network          │
    │    (periodic)       │
    │         ↓           │
    │ 8. Decay Epsilon    │
    │         ↓           │
    │ Check if Done?      │
    │                     │
    └─────────────────────┘
         ↓ (if done)
Log Episode Metrics
    ↓
Save Checkpoint (periodic)
    ↓
NEXT EPISODE
```

### 4.3 Q-Learning Update Flow

```
Sample Batch: {(s_i, a_i, r_i, s'_i, done_i)}
    ↓
Compute Current Q-values:
Q_current = Q_network(s_i)[a_i]
    ↓
Compute Target Q-values:
Q_next = max(Q_target(s'_i))
Q_target = r_i + γ * Q_next * (1 - done_i)
    ↓
Compute Loss:
loss = MSE(Q_current, Q_target)
    ↓
Backpropagation:
loss.backward()
    ↓
Gradient Clipping (prevent exploding gradients)
    ↓
Optimizer Step:
optimizer.step()
    ↓
Update Step Counter
    ↓
If step % target_update == 0:
    Q_target ← Q_network
```

---

## 5. Training Pipeline

### 5.1 Training Script Flow (`train.py`)

**Initialization Phase**:
```python
1. Parse command-line arguments
2. Create directories (models/, logs/)
3. Initialize environment with action discovery
4. Create agent with specified hyperparameters
5. Load checkpoint (if resuming)
6. Initialize logger
7. Select reward shaping function
```

**Training Loop**:
```python
for episode in range(start_episode, max_episodes):
    # Episode initialization
    obs, info = env.reset()
    episode_reward = 0
    episode_steps = 0
    episode_lines = 0
    
    # Episode loop
    while not done:
        # 1. Action selection
        action = agent.select_action(obs)
        
        # 2. Environment step
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # 3. Track original reward
        original_reward = reward
        
        # 4. Apply reward shaping
        reward = shaper_fn(obs, action, reward, done, info)
        
        # 5. Store experience
        agent.remember(obs, action, reward, next_obs, done)
        
        # 6. Learn from experience
        if len(agent.memory) >= agent.batch_size:
            agent.learn()
        
        # 7. Update state and metrics
        obs = next_obs
        episode_reward += reward
        episode_steps += 1
    
    # Episode completed
    # 8. Extract lines cleared
    episode_lines = extract_lines_from_info(info)
    
    # 9. Update exploration rate
    agent.update_epsilon(episode)
    
    # 10. Log episode
    logger.log_episode(...)
    
    # 11. Print progress (periodic)
    if episode % log_interval == 0:
        print_training_status(...)
    
    # 12. Save checkpoint (periodic)
    if episode % save_interval == 0:
        agent.save_checkpoint(...)
    
    # 13. Plot progress (periodic)
    if episode % plot_interval == 0:
        logger.plot_progress()
```

### 5.2 Key Training Parameters

**Learning Parameters**:
- **Learning Rate (LR)**: 0.0001 - Small to prevent instability
- **Gamma (γ)**: 0.99 - Strong emphasis on future rewards
- **Batch Size**: 32-64 - Balance between speed and stability
- **Memory Size**: 100,000 - Large buffer for diverse experiences

**Exploration Schedule**:
- **Epsilon Start**: 1.0 - Pure exploration initially
- **Epsilon End**: 0.01-0.05 - Always maintain small exploration
- **Epsilon Decay**: 0.9999 - Slow decay over 10,000+ episodes

**Network Updates**:
- **Target Update Frequency**: 1000 steps - Stabilizes learning
- **Gradient Clipping**: 1.0 - Prevents exploding gradients

---

## 6. Key Algorithms

### 6.1 Deep Q-Learning (DQN)

**Core Equation**:
```
Q(s, a) ← Q(s, a) + α[r + γ max Q(s', a') - Q(s, a)]
```

**Neural Network Approximation**:
```
Q(s, a; θ) ≈ Q*(s, a)
```
Where θ are the network weights.

**Loss Function**:
```
L(θ) = E[(r + γ max Q(s', a'; θ⁻) - Q(s, a; θ))²]
```
Where θ⁻ are target network weights (frozen).

**Algorithm**:
```
Initialize Q-network with random weights θ
Initialize target network with weights θ⁻ = θ
Initialize replay buffer D

for episode in episodes:
    Initialize state s
    for step in episode:
        # ε-greedy action selection
        a = argmax Q(s, a; θ) with probability 1-ε
            random action       with probability ε
        
        # Execute action
        Execute a, observe r, s'
        
        # Store experience
        Store (s, a, r, s') in D
        
        # Sample mini-batch
        Sample random batch from D
        
        # Compute target
        y = r + γ max Q(s', a'; θ⁻) for non-terminal s'
        y = r                       for terminal s'
        
        # Gradient descent
        Update θ to minimize (y - Q(s, a; θ))²
        
        # Update target network
        Every C steps: θ⁻ ← θ
```

### 6.2 Experience Replay

**Purpose**: Break correlation between consecutive experiences.

**Benefits**:
1. **Data Efficiency**: Each experience used multiple times
2. **Stability**: Breaks temporal correlations
3. **Diverse Learning**: Samples from varied game states

**Implementation**:
```python
class ReplayBuffer(deque):
    def __init__(self, max_size):
        super().__init__(maxlen=max_size)
    
    def add(self, experience):
        self.append(experience)
    
    def sample(self, batch_size):
        return random.sample(self, batch_size)
```

### 6.3 Target Network

**Purpose**: Stabilize Q-learning by using fixed targets.

**Problem**: Without target network, Q-values chase a moving target:
```
Q(s, a) ← r + γ max Q(s', a')
         ↑            ↑
      same network!
```

**Solution**: Use separate target network:
```
Q(s, a) ← r + γ max Q_target(s', a')
         ↑              ↑
    policy network   frozen network
```

**Update Schedule**:
```python
if steps % target_update_freq == 0:
    Q_target.load_state_dict(Q_policy.state_dict())
```

### 6.4 ε-Greedy Exploration

**Purpose**: Balance exploration (learning new strategies) and exploitation (using known good strategies).

**Formula**:
```python
if random() < epsilon:
    action = random_action()  # Explore
else:
    action = argmax Q(state)  # Exploit
```

**Decay Schedule**:

1. **Exponential Decay**:
```
ε_t = max(ε_min, ε_start * decay^t)
```

2. **Linear Decay**:
```
ε_t = max(ε_min, ε_start - t * decay_rate)
```

3. **Adaptive Decay**:
```
ε_t = ε_schedule[episode]  # Pre-computed schedule
```

---

## 7. Integration Logic

### 7.1 How Components Work Together

**Startup Sequence**:
```
1. train.py main()
       ↓
2. Parse arguments
       ↓
3. config.make_env()
       ↓
4. config.discover_action_meanings()
       ↓
5. Agent.__init__()
       ├→ model.create_model()  (Q-network)
       └→ model.create_model()  (Target network)
       ↓
6. TrainingLogger.__init__()
       ↓
7. Training loop starts
```

**Episode Execution**:
```
env.reset()
    ↓
agent.select_action(obs)
    ├→ Preprocess obs
    ├→ Q-network forward pass
    └→ ε-greedy selection
    ↓
env.step(action)
    ↓
reward_shaping.balanced_reward_shaping()
    ├→ extract_board_from_obs()
    ├→ get_column_heights()
    ├→ count_holes()
    ├→ calculate_bumpiness()
    └→ horizontal_distribution()
    ↓
agent.remember(experience)
    ↓
agent.learn()
    ├→ Sample batch from replay buffer
    ├→ Q-network forward pass
    ├→ Target network forward pass
    ├→ Compute loss
    ├→ Backpropagate
    └→ Update weights
    ↓
logger.log_episode()
    ↓
agent.save_checkpoint() (periodic)
```

### 7.2 Data Transformations

**Observation Path**:
```
Tetris Env Output:
  dict{'board': (20,10), 'holder': int, 'queue': (5,)}
       ↓
config._board_from_obs():
  np.array(20, 10) normalized to [0, 1]
       ↓
flatten (if needed):
  np.array(944,) 
       ↓
torch.FloatTensor:
  tensor(1, 944) with batch dimension
       ↓
Q-Network:
  Conv or MLP processing
       ↓
Output:
  tensor(1, n_actions) Q-values
```

**Reward Path**:
```
Env Step:
  reward = game_score_delta
       ↓
Extract board state:
  board = extract_board_from_obs(obs)
       ↓
Compute metrics:
  heights = get_column_heights(board)
  holes = count_holes(board)
  bumpiness = calculate_bumpiness(heights)
  distribution = horizontal_distribution(board)
       ↓
Reward shaping:
  shaped_reward = base_reward
                + line_bonus
                - hole_penalty
                - height_penalty
                - bumpiness_penalty
                + distribution_bonus
                - death_penalty
       ↓
Store in replay buffer:
  memory.append((s, a, shaped_reward, s', done))
```

### 7.3 Checkpoint System

**Saved State**:
```python
checkpoint = {
    'episode': agent.episodes_done,
    'q_network_state': agent.q_network.state_dict(),
    'target_network_state': agent.target_network.state_dict(),
    'optimizer_state': agent.optimizer.state_dict(),
    'epsilon': agent.epsilon,
    'steps': agent.steps_done,
    'hyperparameters': {
        'lr': agent.lr,
        'gamma': agent.gamma,
        'batch_size': agent.batch_size,
        # ... etc
    }
}
```

**Resume Training**:
```python
if agent.load_checkpoint(latest=True):
    start_episode = agent.episodes_done
    print(f"Resuming from episode {start_episode}")
else:
    start_episode = 0
    print("Starting fresh training")
```

### 7.4 Logging and Monitoring

**Real-time Monitoring**:
```python
# Every N episodes, print:
Episode: 1000/10000
  Reward: 123.45 (avg: 98.76 over last 100)
  Steps: 234 (avg: 189 over last 100)
  Lines: 2 (total: 156, avg: 0.16 per episode)
  Epsilon: 0.3678
  Loss: 0.0234
  Time: 12.3s per episode
```

**Saved Artifacts**:
```
logs/experiment_name/
├── config.json              # Training configuration
├── metrics.json             # Complete training history
├── episodes.csv             # Tabular episode data
└── training_curves.png      # Visualization
```

---

## 8. Common Issues and Solutions

### 8.1 Poor Performance (No Lines Cleared)

**Symptoms**: Agent plays randomly, never clears lines, short episodes.

**Diagnosis**:
1. Check action mappings: `python tests/test_setup.py`
2. Verify board state extraction
3. Confirm reward shaping is applied
4. Check epsilon decay (should start high)

**Solutions**:
- Ensure `discover_action_meanings()` correctly maps actions
- Verify `extract_board_from_obs()` normalizes values
- Use "balanced" reward shaping initially
- Increase exploration duration (higher episodes or slower decay)

### 8.2 Training Instability

**Symptoms**: Loss spikes, performance degrades, NaN values.

**Diagnosis**:
1. Check reward magnitudes (should be -100 to +500 range)
2. Verify gradient clipping is enabled
3. Check learning rate

**Solutions**:
- Normalize observations to [0, 1]
- Enable gradient clipping (max_norm=1.0)
- Reduce learning rate
- Increase target network update frequency

### 8.3 Slow Learning

**Symptoms**: No improvement after thousands of episodes.

**Diagnosis**:
1. Check replay buffer size (should be >> batch size)
2. Verify experiences are diverse
3. Check exploration rate

**Solutions**:
- Increase memory size (100K+)
- Slow down epsilon decay
- Use aggressive reward shaping initially
- Increase training frequency (learn every step)

---

## 9. Extension Points

### 9.1 Adding New Reward Functions

**Steps**:
1. Create function in `src/reward_shaping.py`:
```python
def my_custom_shaping(obs, action, reward, done, info):
    board = extract_board_from_obs(obs)
    # Your custom logic here
    bonus = compute_my_bonus(board)
    return reward + bonus
```

2. Add to shaping functions dict in `train.py`:
```python
shaping_functions = {
    'my_custom': my_custom_shaping,
    # ...
}
```

3. Use: `python train.py --reward_shaping my_custom`

### 9.2 Adding New Model Architectures

**Steps**:
1. Create model class in `src/model.py`:
```python
class MyCustomDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        # Your architecture
    
    def forward(self, x):
        # Forward pass
```

2. Update `create_model()` factory:
```python
if model_type == "my_custom":
    return MyCustomDQN(input_shape, n_actions)
```

3. Use: `python train.py --model_type my_custom`

### 9.3 Adding New Logging Metrics

**Steps**:
1. Update `TrainingLogger.log_episode()` to accept new metric
2. Add to CSV headers in `_init_csv()`
3. Update visualization in `plot_progress()`
4. Pass new metric from training loop

---

## 10. Testing and Validation

### 10.1 Test Suite

**Available Tests**:
```bash
# Comprehensive setup validation
python tests/test_setup.py

# Observation integrity check
python tests/test_observation_integrity.py

# Reward function validation
python tests/test_reward_helpers.py

# Action mapping verification
python test_actions.py

# Current training analysis
python tests/analyze_current_training.py
```

### 10.2 Validation Checklist

Before training:
- [ ] Environment creates successfully
- [ ] Actions are correctly mapped
- [ ] Observations are normalized
- [ ] Agent can select actions
- [ ] Replay buffer stores experiences
- [ ] Networks perform forward passes
- [ ] Reward shaping produces reasonable values
- [ ] Logger creates files

During training:
- [ ] Epsilon decreases over time
- [ ] Loss values are reasonable (< 10.0)
- [ ] Episodes have varying lengths
- [ ] Some episodes clear lines
- [ ] Checkpoints save successfully

After training:
- [ ] Logs contain complete data
- [ ] Plots are generated
- [ ] Model can be loaded
- [ ] Evaluation works

---

## 11. Performance Expectations

### 11.1 Typical Training Progression

**Episodes 1-1000**: Random exploration
- Avg reward: -50 to 0
- Lines cleared: 0-0.01 per episode
- Episode length: 10-50 steps

**Episodes 1000-5000**: Learning begins
- Avg reward: 0 to 50
- Lines cleared: 0.01-0.1 per episode
- Episode length: 50-150 steps

**Episodes 5000-10000**: Skill development
- Avg reward: 50 to 200
- Lines cleared: 0.1-1.0 per episode
- Episode length: 150-500 steps

**Episodes 10000+**: Mastery (if successful)
- Avg reward: 200+
- Lines cleared: 1.0+ per episode
- Episode length: 500+ steps

### 11.2 Hardware Considerations

**CPU Training**:
- ~5-15 episodes/second
- 10,000 episodes: 11-33 hours

**GPU Training** (recommended):
- ~20-100 episodes/second
- 10,000 episodes: 1.7-8.3 hours

**Memory Requirements**:
- RAM: 4-8 GB
- VRAM (GPU): 2-4 GB
- Disk: ~500 MB per checkpoint

---

## 12. Conclusion

This documentation covers the complete architecture and implementation of the Tetris RL project. The system integrates:

1. **Robust Environment Setup**: Dynamic action discovery and observation processing
2. **DQN Implementation**: Experience replay, target networks, ε-greedy exploration
3. **Reward Engineering**: Multiple shaping strategies for faster learning
4. **Comprehensive Logging**: Detailed metrics and visualizations
5. **Extensibility**: Easy to add new models, rewards, and features

**Key Success Factors**:
- Proper observation normalization
- Correct action mappings
- Balanced reward shaping
- Sufficient exploration
- Adequate training duration

**Next Steps**:
1. Run test suite: `python tests/test_setup.py`
2. Start training: `python train.py --episodes 10000 --reward_shaping balanced`
3. Monitor progress: Check logs/ directory
4. Evaluate: `python evaluate.py --model_path models/latest.pth --render`

For issues or improvements, refer to specific component documentation above.

---

*End of Documentation*
