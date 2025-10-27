# REWARD_SHAPING_DOCUMENTATION.md

# Reward Shaping Module - Detailed Documentation

## Overview

The Reward Shaping module (`src/reward_shaping.py`) provides custom reward functions that guide the agent's learning beyond the sparse rewards provided by the Tetris environment. This is critical for accelerating learning and achieving good performance.

---

## Table of Contents
1. [Problem: Sparse Rewards](#problem-sparse-rewards)
2. [Solution: Reward Shaping](#solution-reward-shaping)
3. [Helper Functions](#helper-functions)
4. [Reward Functions](#reward-functions)
5. [Design Principles](#design-principles)
6. [Usage Guide](#usage-guide)
7. [Customization](#customization)

---

## Problem: Sparse Rewards

### Tetris Natural Rewards

The Tetris environment provides rewards only when specific events occur:

```python
# Points awarded by environment
- Clear 1 line:  +40 points
- Clear 2 lines: +100 points
- Clear 3 lines: +300 points
- Clear 4 lines: +1200 points
- Game over:     0 points
- Per step:      0 points
```

### Why This Is Problematic

**1. Sparse Feedback**
```
Episode steps: 1, 2, 3, 4, ..., 98, 99, 100
Rewards:       0, 0, 0, 0, ..., 0,  40,  0
                                    ↑
                           Only 1 meaningful reward!
```

**2. Credit Assignment Problem**
- Which of the 99 actions led to clearing the line?
- Agent receives no feedback on most actions
- Hard to learn cause-and-effect relationships

**3. Exploration Challenge**
- Random exploration rarely clears lines
- No signal to guide learning
- Agent may never discover good strategies

**4. Long Training Time**
- Takes thousands of episodes to stumble upon line clears
- Most learning happens by accident
- Inefficient use of training time

---

## Solution: Reward Shaping

### Core Concept

**Add intermediate rewards** that provide feedback on every action, even when no lines are cleared.

```python
shaped_reward = base_reward + additional_feedback
```

### Benefits

1. **Dense Feedback**: Every action gets a signal
2. **Faster Learning**: Agent discovers strategies sooner
3. **Better Exploration**: Rewards guide toward good states
4. **Improved Performance**: Higher final skill level

### Potential Reward Shaping

**Principle**: Add rewards that don't change optimal policy.

```
Potential-based shaping: R'(s, a, s') = R(s, a, s') + γΦ(s') - Φ(s)

Where Φ(s) is a potential function on states.
```

**Our Approach**: Heuristic-based shaping (simpler, effective for Tetris)

---

## Helper Functions

### 1. Board Extraction

#### `extract_board_from_obs(obs)`

**Purpose**: Extract and normalize 2D board state from observation.

```python
def extract_board_from_obs(obs):
    """
    Extract board from observation and normalize to [0, 1].
    
    Args:
        obs: Observation from environment (dict or array)
    
    Returns:
        board: np.array of shape (H, W) with values in [0, 1]
    """
```

**Logic**:
```python
# Handle dict observations
if isinstance(obs, dict):
    board = obs.get('board')
    if board is None:
        return np.zeros((20, 10), dtype=np.float32)
else:
    board = obs

# Handle 3D arrays (with channel dimension)
if len(board.shape) == 3:
    board = board[:, :, 0]  # Take first channel

# Normalize to [0, 1]
if board.max() > 1:
    board = board.astype(np.float32) / 255.0

return board
```

**Why Normalize?**
- Consistent value ranges for reward calculation
- Prevents scale issues in neural networks
- Makes reward magnitudes predictable

### 2. Column Heights

#### `get_column_heights(board)`

**Purpose**: Calculate height of filled cells in each column.

```python
def get_column_heights(board):
    """
    Get height of each column (top filled cell).
    
    Args:
        board: 2D numpy array (H, W)
    
    Returns:
        heights: List of 10 heights (one per column)
    """
    heights = []
    for col in range(board.shape[1]):
        column = board[:, col]
        filled_indices = np.where(column > 0)[0]
        
        if len(filled_indices) > 0:
            # Height = board_height - topmost_filled_index
            height = board.shape[0] - filled_indices[0]
        else:
            height = 0
        
        heights.append(height)
    
    return heights
```

**Visualization**:
```
Board (20x10):           Heights:
│ │ │█│ │ │ │█│ │ │      [0, 0, 3, 0, 0, 0, 5, 0, 0, 0]
│ │ │█│ │ │ │█│ │ │           ↑        ↑
│ │ │█│ │ │ │█│ │ │           3        5
│ │ │ │ │ │ │█│ │ │
│ │ │ │ │ │ │█│ │ │
Bottom row

Column: 0  1  2  3  4  5  6  7  8  9
```

### 3. Hole Counting

#### `count_holes(board)`

**Purpose**: Count empty cells with filled cells above them.

```python
def count_holes(board):
    """
    Count holes (empty cells with filled cells above).
    
    Holes are bad because they're hard to fill.
    
    Args:
        board: 2D numpy array (H, W)
    
    Returns:
        holes: Total number of holes
    """
    holes = 0
    
    for col in range(board.shape[1]):
        column = board[:, col]
        
        # Find topmost filled cell
        filled_indices = np.where(column > 0)[0]
        if len(filled_indices) == 0:
            continue
        
        top_filled = filled_indices[0]
        
        # Count empty cells below topmost filled
        for row in range(top_filled + 1, board.shape[0]):
            if board[row, col] == 0:
                holes += 1
    
    return holes
```

**Why Holes Are Bad**:
```
Good (no holes):        Bad (2 holes):
│ │ │ │ │               │ │█│ │ │
│ │ │█│ │               │ │█│ │ │
│ │█│█│ │               │ │ │ │ │  ← Hole!
│█│█│█│█│               │█│ │█│█│  ← Hole!
└─┴─┴─┴─┘               └─┴─┴─┴─┘

Easy to clear           Hard/impossible to clear
```

### 4. Bumpiness

#### `calculate_bumpiness(heights)`

**Purpose**: Measure surface irregularity.

```python
def calculate_bumpiness(heights):
    """
    Calculate bumpiness (sum of absolute height differences).
    
    Smoother surfaces are easier to manage.
    
    Args:
        heights: List of column heights
    
    Returns:
        bumpiness: Sum of |h[i] - h[i+1]|
    """
    bumpiness = 0
    
    for i in range(len(heights) - 1):
        bumpiness += abs(heights[i] - heights[i+1])
    
    return bumpiness
```

**Example**:
```
Heights:  [2, 5, 3, 7, 2, 8, 3]
Diffs:      3  2  4  5  6  5
Bumpiness: 3+2+4+5+6+5 = 25

Smooth:   [2, 2, 3, 3, 3, 2, 2]
Diffs:      0  1  0  0  1  0
Bumpiness: 0+1+0+0+1+0 = 2

Bumpy surface = harder to play!
```

### 5. Aggregate Height

#### `calculate_aggregate_height(heights)`

**Purpose**: Sum of all column heights.

```python
def calculate_aggregate_height(heights):
    """
    Sum of all column heights.
    
    Lower aggregate height = more room to play.
    
    Args:
        heights: List of column heights
    
    Returns:
        total: Sum of heights
    """
    return sum(heights)
```

### 6. Maximum Height

#### `get_max_height(heights)`

**Purpose**: Find tallest column.

```python
def get_max_height(heights):
    """
    Maximum column height.
    
    Critical metric: game over when height reaches top.
    
    Args:
        heights: List of column heights
    
    Returns:
        max_height: Maximum value in heights
    """
    return max(heights) if heights else 0
```

### 7. Wells

#### `calculate_wells(heights)`

**Purpose**: Count deep gaps between columns.

```python
def calculate_wells(heights):
    """
    Count wells (deep gaps good for I-pieces).
    
    A well is a column significantly lower than neighbors.
    
    Args:
        heights: List of column heights
    
    Returns:
        wells: Number of wells
    """
    wells = 0
    
    for i in range(len(heights)):
        left_height = heights[i-1] if i > 0 else float('inf')
        current = heights[i]
        right_height = heights[i+1] if i < len(heights)-1 else float('inf')
        
        # Well if current is 3+ lower than both neighbors
        if current + 3 <= min(left_height, right_height):
            wells += 1
    
    return wells
```

**Visualization**:
```
Heights: [5, 2, 5]
         ┌─┐   ┌─┐
         │█│   │█│
         │█│   │█│
         │█│ │ │█│  ← Well here!
         │█│ │ │█│
         └─┴─┴─┴─┘
```

### 8. Horizontal Distribution

#### `horizontal_distribution(board)`

**Purpose**: Reward spreading pieces across the board width.

```python
def horizontal_distribution(board):
    """
    Reward for using full board width.
    
    Prevents clustering in center columns.
    
    Args:
        board: 2D numpy array (H, W)
    
    Returns:
        bonus: Reward based on column usage
    """
    heights = get_column_heights(board)
    
    # Count columns with non-zero height
    active_columns = sum(1 for h in heights if h > 0)
    
    # Bonus for using more columns
    if active_columns >= 8:
        bonus = 10.0
    elif active_columns >= 6:
        bonus = 5.0
    elif active_columns >= 4:
        bonus = 2.0
    else:
        bonus = 0.0
    
    return bonus
```

**Why Important**:
```
Bad (clustered):        Good (distributed):
│ │ │ │ │ │ │           │ │ │ │ │ │ │
│ │ │█│ │ │ │           │█│ │█│ │█│ │
│ │█│█│█│ │ │           │█│█│█│█│█│ │
│█│█│█│█│ │ │           │█│█│█│█│█│█│
└─┴─┴─┴─┴─┴─┘           └─┴─┴─┴─┴─┴─┘
Active: 4 columns        Active: 8 columns
Limited options         More flexibility
```

---

## Reward Functions

### 1. Balanced Reward Shaping (Recommended)

```python
def balanced_reward_shaping(obs, action, reward, done, info):
    """
    Balanced reward incorporating multiple factors.
    
    Philosophy: Balance positive and negative signals.
    
    Args:
        obs: Current observation
        action: Action taken
        reward: Original environment reward
        done: Episode termination flag
        info: Environment info dict
    
    Returns:
        shaped_reward: Modified reward
    """
```

**Components**:

1. **Line Clearing Bonus** (Primary objective)
```python
lines_cleared = info.get('lines_cleared', 0)
if lines_cleared > 0:
    shaped_reward += lines_cleared * 500
```

2. **Hole Penalty** (Discourage bad placements)
```python
holes = count_holes(board)
shaped_reward -= holes * 1.0
```

3. **Height Penalty** (Keep board low)
```python
max_height = get_max_height(heights)
shaped_reward -= max_height * 0.5
```

4. **Bumpiness Penalty** (Promote smooth surface)
```python
bumpiness = calculate_bumpiness(heights)
shaped_reward -= bumpiness * 0.3
```

5. **Distribution Bonus** (Use full width)
```python
distribution_bonus = horizontal_distribution(board)
shaped_reward += distribution_bonus
```

6. **Death Penalty** (Avoid game over)
```python
if done and lines_cleared == 0:
    shaped_reward -= 50
```

**Formula**:
```
R_shaped = R_base 
         + (lines × 500)
         - (holes × 1.0)
         - (max_height × 0.5)
         - (bumpiness × 0.3)
         + distribution_bonus
         - death_penalty
```

**Typical Values**:
```
Good play:  +400 to +600
OK play:    -10 to +50
Bad play:   -100 to -50
Death:      -200 to -100
```

### 2. Aggressive Reward Shaping

```python
def aggressive_reward_shaping(obs, action, reward, done, info):
    """
    Strong penalties for bad play.
    
    Philosophy: Harsh punishment for mistakes.
    
    Use when: Agent is too exploratory, making bad moves.
    """
```

**Key Differences from Balanced**:
- **Holes**: -3 each (vs -1)
- **Height**: -1.0 per unit (vs -0.5)
- **Bumpiness**: -0.5 per unit (vs -0.3)
- **Death**: -100 (vs -50)

**Effect**:
- Agent learns to avoid mistakes faster
- May be too conservative (won't take risks)
- Good for reducing bad behaviors quickly

**When to Use**:
- Agent is too chaotic
- Need to reduce bad placements
- Have stable exploration already

### 3. Positive Reward Shaping

```python
def positive_reward_shaping(obs, action, reward, done, info):
    """
    Focus on positive reinforcement.
    
    Philosophy: Reward good behaviors more than punish bad.
    
    Use when: Agent is too cautious, need more exploration.
    """
```

**Key Differences**:
- **Lines**: +1000 each (vs +500)
- **Low Height Bonus**: +5 (new)
- **Smooth Surface Bonus**: +3 (new)
- **Distribution**: +15 (vs +10)
- **Smaller penalties**

**Effect**:
- Agent more willing to explore
- May make risky moves
- Good for encouraging creativity

**When to Use**:
- Agent is too conservative
- Need more exploration
- Stuck in local optimum

---

## Design Principles

### 1. Reward Magnitudes

**Scale Guideline**:
```
Action-level rewards: -10 to +20
Episode-level rewards: -500 to +2000
```

**Why Important**:
- Too large: Gradient explosion, instability
- Too small: Slow learning, no signal
- Balanced: Smooth, stable learning

### 2. Reward Frequency

```python
# Every step (dense)
- Height penalty
- Hole penalty
- Bumpiness penalty

# Occasional (sparse but important)
- Line clearing bonus
- Distribution bonus

# Rare (episode-level)
- Death penalty
```

### 3. Signal Quality

**Good Signals**:
- Consistent: Same board state → same reward
- Informative: Different qualities → different rewards
- Actionable: Agent can improve based on signal

**Bad Signals**:
- Noisy: Random component in reward
- Ambiguous: Can't tell if action was good/bad
- Unactionable: Can't learn to improve

### 4. Potential-Based Considerations

**Ideally**: Φ(s) should estimate "goodness" of state

```python
Φ(s) = - (holes + height + bumpiness)

Then:
R'(s, a, s') = R(s,a,s') + γΦ(s') - Φ(s)
```

**Our Approach**: Simpler heuristics work well in practice

---

## Usage Guide

### In Training Script

```python
from src.reward_shaping import balanced_reward_shaping

# Select shaping function
shaper = balanced_reward_shaping

# Training loop
while not done:
    action = agent.select_action(obs)
    next_obs, env_reward, done, _, info = env.step(action)
    
    # Apply shaping
    shaped_reward = shaper(obs, action, env_reward, done, info)
    
    # Use shaped reward for learning
    agent.remember(obs, action, shaped_reward, next_obs, done)
    
    obs = next_obs
```

### Comparing Shaping Functions

```python
# Test different shaping on same episode
shapers = {
    'balanced': balanced_reward_shaping,
    'aggressive': aggressive_reward_shaping,
    'positive': positive_reward_shaping,
}

for name, shaper in shapers.items():
    total_shaped = 0
    obs, _ = env.reset(seed=42)  # Same seed
    done = False
    
    while not done:
        action = agent.select_action(obs)
        next_obs, reward, done, _, info = env.step(action)
        
        shaped_reward = shaper(obs, action, reward, done, info)
        total_shaped += shaped_reward
        
        obs = next_obs
    
    print(f"{name}: {total_shaped:.2f}")
```

### Ablation Study

**Test component importance**:

```python
def ablation_reward(obs, action, reward, done, info, 
                    use_holes=True, use_height=True, 
                    use_bumpiness=True, use_distribution=True):
    """Test what components matter most"""
    
    board = extract_board_from_obs(obs)
    shaped_reward = reward
    
    # Line bonus (always included)
    lines = info.get('lines_cleared', 0)
    shaped_reward += lines * 500
    
    # Optional components
    if use_holes:
        holes = count_holes(board)
        shaped_reward -= holes
    
    if use_height:
        heights = get_column_heights(board)
        shaped_reward -= get_max_height(heights) * 0.5
    
    if use_bumpiness:
        heights = get_column_heights(board)
        shaped_reward -= calculate_bumpiness(heights) * 0.3
    
    if use_distribution:
        shaped_reward += horizontal_distribution(board)
    
    return shaped_reward
```

---

## Customization

### Creating Custom Reward Function

```python
def my_custom_shaping(obs, action, reward, done, info):
    """
    Template for custom reward shaping.
    
    Steps:
    1. Extract board state
    2. Compute metrics
    3. Calculate bonuses/penalties
    4. Return modified reward
    """
    
    # 1. Extract board
    board = extract_board_from_obs(obs)
    
    # 2. Compute your metrics
    my_metric = calculate_my_feature(board)
    
    # 3. Build reward
    shaped_reward = reward  # Start with base reward
    
    # Add your components
    shaped_reward += my_metric * my_weight
    
    # 4. Return
    return shaped_reward
```

### Example: Add Tetris Bonus

```python
def tetris_bonus_shaping(obs, action, reward, done, info):
    """Bonus for clearing 4 lines (Tetris)"""
    
    shaped_reward = balanced_reward_shaping(obs, action, reward, done, info)
    
    # Extra bonus for Tetris (4 lines)
    lines = info.get('lines_cleared', 0)
    if lines == 4:
        shaped_reward += 200  # Big bonus!
    
    return shaped_reward
```

### Example: Combo Reward

```python
def combo_reward_shaping(obs, action, reward, done, info):
    """Reward consecutive line clears"""
    
    global last_lines_cleared, combo_count
    
    shaped_reward = balanced_reward_shaping(obs, action, reward, done, info)
    
    lines = info.get('lines_cleared', 0)
    
    if lines > 0:
        # Continue combo
        combo_count += 1
        shaped_reward += combo_count * 50  # Escalating bonus
    else:
        # Reset combo
        combo_count = 0
    
    return shaped_reward
```

---

## Testing

### Validate Helper Functions

```python
def test_hole_counting():
    # Board with 2 holes
    board = np.array([
        [0, 1, 0],
        [1, 0, 1],  # 2 holes
        [1, 1, 1],
    ])
    
    holes = count_holes(board)
    assert holes == 2, f"Expected 2 holes, got {holes}"

def test_bumpiness():
    heights = [2, 5, 3, 7]
    # Diffs: 3, 2, 4
    # Sum: 9
    
    bumpiness = calculate_bumpiness(heights)
    assert bumpiness == 9, f"Expected 9, got {bumpiness}"
```

### Validate Reward Functions

```python
def test_reward_magnitudes():
    """Ensure rewards are in expected range"""
    
    env = make_env()
    obs, _ = env.reset()
    
    total_reward = 0
    step_rewards = []
    
    for _ in range(100):
        action = env.action_space.sample()
        next_obs, env_reward, done, _, info = env.step(action)
        
        shaped_reward = balanced_reward_shaping(
            obs, action, env_reward, done, info
        )
        
        step_rewards.append(shaped_reward)
        total_reward += shaped_reward
        
        obs = next_obs
        if done:
            break
    
    # Check reasonable ranges
    assert -500 < total_reward < 2000
    assert all(-100 < r < 600 for r in step_rewards)
```

---

## Common Issues

### Issue 1: Rewards Too Negative

**Symptom**: Agent never improves, total rewards always < -1000

**Diagnosis**:
```python
# Check reward components
print(f"Base reward: {reward}")
print(f"Holes penalty: {-holes}")
print(f"Height penalty: {-max_height * 0.5}")
print(f"Bumpiness: {-bumpiness * 0.3}")
```

**Solutions**:
- Reduce penalty weights
- Increase bonus weights
- Check board normalization
- Use positive shaping

### Issue 2: No Learning

**Symptom**: Agent behavior doesn't change over time

**Diagnosis**:
```python
# Check if shaping is applied
print(f"Original reward: {env_reward}")
print(f"Shaped reward: {shaped_reward}")
print(f"Difference: {shaped_reward - env_reward}")
```

**Solutions**:
- Increase reward magnitude
- Check if shaping function is called
- Verify board extraction works
- Test with simpler shaping

### Issue 3: Overfitting to Shaping

**Symptom**: High shaped rewards but low actual game score

**Diagnosis**: Agent optimizing shaping rewards, not game objectives

**Solutions**:
- Reduce shaping weight
- Increase line clearing bonus
- Add more game-aligned metrics
- Use adaptive shaping (reduce over time)

---

## Summary

Reward shaping is critical for successful Tetris RL training:

**Key Points**:
1. **Sparse rewards are problematic** - need dense feedback
2. **Multiple metrics** - holes, height, bumpiness, distribution
3. **Balanced approach** - reward good, penalize bad
4. **Customizable** - easy to add new components
5. **Testable** - validate with unit tests

**Best Practices**:
- Start with balanced shaping
- Monitor both shaped and original rewards
- Adjust weights based on agent behavior
- Test components individually
- Gradually reduce shaping over time (optional)

**Integration**:
- Called every step in training loop
- Receives obs, action, reward, done, info
- Returns modified reward for learning
- Doesn't affect environment, only agent

---

*End of Reward Shaping Documentation*
