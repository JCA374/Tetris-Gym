# REWARD_SHAPING_DOCUMENTATION.md

# Reward Shaping - Complete Guide

## Table of Contents
1. [Overview](#overview)
2. [The Problem: Sparse Rewards](#the-problem-sparse-rewards)
3. [The Solution: Reward Shaping](#the-solution-reward-shaping)
4. [Helper Functions](#helper-functions)
5. [Reward Shaping Strategies](#reward-shaping-strategies)
6. [CRITICAL: Anti-Exploitation Fixes](#critical-anti-exploitation-fixes)
7. [Design Principles](#design-principles)
8. [Usage Guide](#usage-guide)
9. [Customization](#customization)
10. [Testing](#testing)
11. [Common Issues](#common-issues)
12. [Lessons Learned](#lessons-learned)

---

## Overview

**Reward shaping** is a technique to provide denser, more informative feedback to the reinforcement learning agent than the sparse rewards from the base Tetris environment.

**Why it matters**: Without reward shaping, Tetris RL is extremely difficult:
- Lines cleared (the goal) happen rarely
- Agent receives mostly zero rewards
- Hard to learn what actions lead to success

**What it does**: Adds intermediate rewards based on board quality metrics, guiding the agent toward good play even before clearing lines.

---

## The Problem: Sparse Rewards

### Base Tetris Environment Rewards

```python
# Typical Tetris environment:
step_reward = 0           # Most steps
line_reward = 100-400     # Only when lines cleared (rare!)
death_penalty = 0         # Game over

Problem: 99% of steps give zero reward!
```

### Why This is Hard to Learn

```
Episode 1: [0, 0, 0, 0, ..., 0] (50 steps, no feedback)
Episode 2: [0, 0, 0, 0, ..., 0] (73 steps, no feedback)
Episode 3: [0, 0, 0, 0, ..., 100] (35 steps, ONE reward at end)
```

**The agent has no idea:**
- Which actions were good vs bad
- How to improve
- What led to the line clear in episode 3

**Result**: Random exploration for thousands of episodes before any learning.

---

## The Solution: Reward Shaping

### Dense Feedback

Instead of waiting for rare line clears, give feedback every step based on board quality:

```python
# With reward shaping:
Good placement:    +10 (lower height, no holes)
OK placement:      -5 (slight height increase)
Bad placement:     -20 (created holes)
Line clear:        +1000 (JACKPOT!)
Death:             -50 (avoid this)
```

### Key Insight

**Goal**: Shape rewards to align with good Tetris strategy:
- Keep height low ✓
- Minimize holes ✓
- Smooth surface ✓
- Use full board width ✓
- Clear lines (primary goal) ✓✓✓

---

## Helper Functions

### 1. `extract_board_from_obs(obs)`

Extracts the 2D board array from observations (handles dict or array format).

```python
def extract_board_from_obs(obs):
    """
    Extract board from observation.
    Handles both dict and array formats.
    Returns normalized board [0, 1]
    """
    if isinstance(obs, dict):
        board = obs.get('board', None)
    elif isinstance(obs, np.ndarray):
        board = obs
    else:
        return None
    
    if board is None:
        return None
    
    # Normalize to [0, 1]
    if board.max() > 1:
        board = board / 255.0
    
    return board
```

---

### 2. `get_column_heights(board)`

Returns the height of each column (how many filled cells from bottom).

```python
def get_column_heights(board):
    """
    Get height of each column.
    
    Returns:
        List of heights, e.g., [5, 3, 8, 2, ...]
    """
    heights = []
    for col in range(board.shape[1]):
        column = board[:, col]
        filled_cells = np.where(column > 0)[0]
        if len(filled_cells) > 0:
            height = board.shape[0] - filled_cells[0]
        else:
            height = 0
        heights.append(height)
    return heights
```

**Example:**
```
Board:        Heights:
  ····        [0, 2, 4, 1, 0]
  ·█·█·
  ·███·
  ·███·
  ·███·
```

---

### 3. `count_holes(board)`

Counts empty cells that have filled cells above them (very bad for Tetris).

```python
def count_holes(board):
    """
    Count holes (empty cells with filled cells above).
    Holes are hard to fill and bad for gameplay.
    """
    holes = 0
    for col in range(board.shape[1]):
        column = board[:, col]
        filled_indices = np.where(column > 0)[0]
        if len(filled_indices) > 0:
            top_filled = filled_indices[0]
            holes += np.sum(column[top_filled:] == 0)
    return holes
```

**Example:**
```
Board:        Holes: 2
  ████·
  ██·█·  ← 1 hole
  ███··
  ██·█·  ← 1 hole
  █████
```

---

### 4. `calculate_bumpiness(heights)`

Measures how uneven the surface is (sum of height differences between adjacent columns).

```python
def calculate_bumpiness(heights):
    """
    Calculate bumpiness (surface irregularity).
    
    Args:
        heights: List of column heights (NOT the board!)
    
    Returns:
        Sum of absolute height differences
    """
    if not heights or len(heights) < 2:
        return 0
    
    bumpiness = sum(abs(heights[i] - heights[i+1]) 
                    for i in range(len(heights) - 1))
    return bumpiness
```

**Example:**
```
Heights: [3, 5, 2, 7, 6]
Differences: |3-5| + |5-2| + |2-7| + |7-6|
           = 2 + 3 + 5 + 1 = 11
Bumpiness: 11
```

---

### 5. `get_horizontal_distribution(board)`

Rewards spreading pieces across the full board width (prevents center clustering).

```python
def get_horizontal_distribution(board):
    """
    Measure how well pieces are distributed horizontally.
    Returns bonus for good distribution.
    """
    heights = get_column_heights(board)
    if not heights:
        return 0
    
    # Check which columns are used
    used_columns = sum(1 for h in heights if h > 0)
    total_columns = len(heights)
    
    # Bonus for using more columns
    distribution_score = (used_columns / total_columns) * 10
    
    return distribution_score
```

---

## Reward Shaping Strategies

### 1. Balanced Reward Shaping (RECOMMENDED) ⭐

**Updated and Fixed Version** - Prevents rotation exploitation!

```python
def balanced_reward_shaping(obs, action, reward, done, info):
    """
    Fixed balanced reward shaping - prevents rotation exploitation
    
    CRITICAL FIXES (October 2025):
    1. Removed per-step survival bonus that caused rotation exploit
    2. Added -1 penalty for steps without progress
    3. Always returns a float (never None)
    
    Philosophy: Balance positive and negative signals
    Primary Goal: Clear lines
    Secondary Goals: Maintain clean board state
    
    Args:
        obs: Current observation
        action: Action taken
        reward: Original environment reward (not used)
        done: Episode termination flag
        info: Environment info dict
    
    Returns:
        shaped_reward: Modified reward value (always a float)
    """
    import numpy as np
    
    # Start from zero (don't use base reward)
    shaped_reward = 0.0
    
    # Extract board state
    board = extract_board_from_obs(obs)
    
    # Safety checks
    if board is None or len(board.shape) != 2:
        return -1.0
    
    # Calculate board metrics
    heights = get_column_heights(board)
    if not heights:
        return -1.0
    
    holes = count_holes(board)
    max_height = max(heights)
    bumpiness = calculate_bumpiness(heights)
    distribution = get_horizontal_distribution(board)
    
    # ========================================================================
    # 1. LINE CLEARING (Primary Objective) - HUGE rewards
    # ========================================================================
    lines = (info.get('lines_cleared', 0) or 
             info.get('number_of_lines', 0) or 
             info.get('lines', 0) or 0)
    
    if lines > 0:
        # Exponential rewards for multiple lines
        line_rewards = {
            1: 1000,   # Single line
            2: 3000,   # Double  
            3: 6000,   # Triple
            4: 12000   # Tetris!
        }
        shaped_reward += line_rewards.get(lines, lines * 1000)
    else:
        # ⚠️ CRITICAL: Penalty for not making progress!
        # This prevents the rotation exploit
        shaped_reward -= 1.0
    
    # ========================================================================
    # 2. BOARD STATE PENALTIES
    # ========================================================================
    
    # Holes are very bad (hard to recover from)
    shaped_reward -= holes * 5.0
    
    # Height penalty (keep board low for survival)
    shaped_reward -= max_height * 1.0
    
    # Bumpiness penalty (smooth surface is easier to manage)
    shaped_reward -= bumpiness * 0.5
    
    # ========================================================================
    # 3. STRATEGIC BONUSES
    # ========================================================================
    
    # Horizontal distribution bonus (spread pieces across board)
    shaped_reward += distribution * 5.0
    
    # Low height bonus (encourages keeping board low)
    if max_height < 10:
        shaped_reward += 5.0
    
    # ========================================================================
    # 4. DEATH PENALTY
    # ========================================================================
    
    if done:
        if lines == 0:
            # Big penalty for dying without clearing any lines
            shaped_reward -= 50.0
        else:
            # Smaller penalty if at least cleared some lines
            shaped_reward -= 10.0
    
    # ========================================================================
    # 5. CLAMP AND RETURN
    # ========================================================================
    
    # Prevent extreme values
    shaped_reward = np.clip(shaped_reward, -500.0, 15000.0)
    
    # Always return a float
    return float(shaped_reward)
```

**Formula:**
```
R_shaped = 0 (baseline)
         + (lines × 1000-12000)     [exponential]
         - (no_progress × 1.0)       [NEW: anti-exploit]
         - (holes × 5.0)
         - (max_height × 1.0)
         - (bumpiness × 0.5)
         + (distribution × 5.0)
         + (low_height_bonus × 5.0)
         - (death_penalty)
```

**Typical Values:**
```
Line clear (1):    +1000
Line clear (4):    +12000
Good play:         -10 to +50
Bad placement:     -50 to -100
Death (no lines):  -50
Step with no progress: -1
```

---

### 2. Aggressive Reward Shaping

```python
def aggressive_reward_shaping(obs, action, reward, done, info):
    """
    Strong penalties for bad play.
    
    Philosophy: Harsh punishment for mistakes
    Use when: Agent is too exploratory, making bad moves
    """
    shaped_reward = 0.0
    board = extract_board_from_obs(obs)
    
    if board is None:
        return -1.0
    
    # Lines cleared
    lines = info.get('lines_cleared', 0) or info.get('number_of_lines', 0) or 0
    if lines > 0:
        shaped_reward += [1000, 3000, 6000, 20000][lines-1]
    
    # Heavy penalties
    shaped_reward -= get_max_height(board) * 5.0   # ← 5x stronger
    shaped_reward -= count_holes(board) * 10.0     # ← 10x stronger
    shaped_reward -= calculate_bumpiness(board) * 2.0
    shaped_reward += get_horizontal_distribution(board) * 20.0
    
    return max(shaped_reward, -200.0)
```

**Key Differences from Balanced:**
- **Holes**: -10 each (vs -5)
- **Height**: -5.0 per unit (vs -1.0)
- **Bumpiness**: -2.0 per unit (vs -0.5)
- **Distribution**: +20 (vs +5)

**Effect:**
- Agent learns to avoid mistakes faster
- May be too conservative (won't take risks)
- Good for reducing bad behaviors quickly

---

### 3. Positive Reward Shaping

```python
def positive_reward_shaping(obs, action, reward, done, info):
    """
    Focus on positive reinforcement.
    
    Philosophy: Reward good behaviors more than punish bad
    Use when: Agent is too cautious, needs more exploration
    """
    shaped_reward = 0.0
    board = extract_board_from_obs(obs)
    
    if board is None:
        return 0.0
    
    # Survival bonus
    shaped_reward += 10.0 if not done else -10.0
    
    # Line clear bonuses
    lines = info.get('lines_cleared', 0) or info.get('number_of_lines', 0) or 0
    if lines > 0:
        shaped_reward += [300, 900, 1800, 6000][lines-1]
    
    # Gentle penalties (only when severe)
    max_height = get_max_height(board)
    if max_height > 15:
        shaped_reward -= (max_height - 15) * 1.0
    
    holes = count_holes(board)
    if holes > 5:
        shaped_reward -= (holes - 5) * 1.0
    
    shaped_reward += get_horizontal_distribution(board) * 15.0
    
    # Low board bonuses
    if max_height < 5:
        shaped_reward += 15.0
    elif max_height < 10:
        shaped_reward += 5.0
    
    return max(shaped_reward, 0.0)
```

**Key Differences:**
- **Survival**: +10 each step (encourages staying alive)
- **Lines**: Lower rewards (300-6000)
- **Penalties**: Only when severe (height > 15, holes > 5)
- **Bonuses**: Additional low-board bonuses

---

## CRITICAL: Anti-Exploitation Fixes

### The Rotation Exploit Bug (Discovered October 2025)

**What Happened:**

During training, the agent discovered a clever exploit in the original reward shaping:

```python
# Original (BROKEN) balanced_reward_shaping:
shaped_reward += 2.0 if not done else -100  # ← BUG!

# Agent's "clever" strategy:
# "If I rotate pieces forever, I never die and get +2 every step!"
# "If I place pieces, I eventually die and get -100!"
# Result: Agent learned to rotate forever, never place pieces!
```

**Symptoms:**
- Agent uses 100% ROTATE_CCW action
- Zero lines cleared even after 2000+ episodes
- Very negative rewards (but "optimal" given the buggy reward)
- Board stays mostly empty (pieces never placed)
- Exploration works perfectly, but exploitation learned wrong behavior

**Root Cause:**

Per-step survival rewards (+2 for being alive) made infinite rotation more rewarding than actual gameplay:

```
Strategy A (Rotation Forever):
  Step 1: Rotate → +2 reward
  Step 2: Rotate → +2 reward
  Step 3: Rotate → +2 reward
  ...
  Total: +2n (grows forever!)

Strategy B (Play Tetris):
  Step 1-50: Play normally → variable rewards
  Step 51: Die → -100 reward
  Total: Usually negative

Agent learns: Rotation Forever > Playing Tetris
```

**The Fix:**

Remove per-step survival rewards and add penalty for time-wasting:

```python
# NEW (FIXED) balanced_reward_shaping:
if lines > 0:
    shaped_reward += lines * 1000  # Reward progress
else:
    shaped_reward -= 1.0  # ← Penalty for wasting time!

# Now:
# Rotation forever: -1, -1, -1, ... (gets worse!)
# Clearing lines: +1000! (much better!)
```

**Implementation Details:**

The fix required THREE changes:

1. **Remove survival bonus:**
   ```python
   # OLD:
   shaped_reward += 2.0 if not done else -100
   
   # NEW:
   # (removed entirely)
   ```

2. **Add progress penalty:**
   ```python
   # NEW:
   if lines > 0:
       shaped_reward += lines * 1000
   else:
       shaped_reward -= 1.0  # Penalize no progress!
   ```

3. **Adjust death penalty:**
   ```python
   # NEW:
   if done:
       if lines == 0:
           shaped_reward -= 50.0  # Big penalty if no progress
       else:
           shaped_reward -= 10.0  # Small penalty if made progress
   ```

**Results After Fix:**

After retraining with fixed reward shaping:
- Agent places pieces normally
- Uses all actions (not just rotation)
- Starts clearing lines within 500-1000 episodes
- Learns actual Tetris strategy

---

## Design Principles

### 1. Reward Magnitudes

**Scale Guideline:**
```
Per-step rewards:      -10 to +20
Board state penalties: -50 to -100
Line clearing:         +1000 to +12000
Episode-level:         -500 to +15000
```

**Why Important:**
- Too large: Gradient explosion, training instability
- Too small: Slow learning, no signal
- Balanced: Smooth, stable convergence

### 2. Reward Frequency

```python
# Every step (dense feedback)
- Progress penalty (-1 if no lines)
- Height penalty (varies)
- Hole penalty (varies)
- Bumpiness penalty (varies)
- Distribution bonus (varies)

# Occasional (sparse but important)
- Line clearing bonus (when lines cleared)

# Rare (episode-level)
- Death penalty (game over)
```

### 3. Signal Quality

**Good Signals:**
- **Consistent**: Same board state → same reward
- **Informative**: Different qualities → different rewards
- **Actionable**: Agent can improve based on signal
- **Non-exploitable**: Can't get infinite rewards without progress

**Bad Signals:**
- **Noisy**: Random component in reward
- **Ambiguous**: Can't tell if action was good/bad
- **Unactionable**: Can't learn to improve
- **Exploitable**: Can maximize reward without achieving goal

### 4. Anti-Exploitation Checklist

When designing reward functions, ensure:

✅ **Progress Required**: Can't get positive rewards without making progress
✅ **Time Penalty**: Wasting time is discouraged
✅ **Goal Aligned**: Highest rewards come from actual objectives
✅ **Bounded**: Can't accumulate infinite rewards
✅ **Tested**: Manually verify agent can't find exploits

---

## Usage Guide

### In Training Script

```python
from src.reward_shaping import balanced_reward_shaping

# Select shaping function
shaper_fn = balanced_reward_shaping

# Training loop
while not done:
    action = agent.select_action(obs)
    next_obs, env_reward, done, _, info = env.step(action)
    
    # Apply shaping
    shaped_reward = shaper_fn(obs, action, env_reward, done, info)
    
    # Use shaped reward for learning (NOT env_reward!)
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
    import numpy as np
    
    # 1. Extract board
    board = extract_board_from_obs(obs)
    if board is None:
        return -1.0
    
    shaped_reward = 0.0
    
    # 2. Compute your metrics
    my_metric = calculate_my_feature(board)
    
    # 3. Build reward
    shaped_reward += my_metric * my_weight
    
    # 4. Prevent exploits
    lines = info.get('lines_cleared', 0)
    if lines == 0:
        shaped_reward -= 1.0  # Progress penalty!
    
    # 5. Return (always a float!)
    return float(np.clip(shaped_reward, -500, 15000))
```

### Example: Add Tetris Bonus

```python
def tetris_bonus_shaping(obs, action, reward, done, info):
    """Extra bonus for clearing 4 lines (Tetris)"""
    
    # Start with balanced shaping
    shaped_reward = balanced_reward_shaping(obs, action, reward, done, info)
    
    # Add extra bonus for Tetris (4 lines)
    lines = info.get('lines_cleared', 0)
    if lines == 4:
        shaped_reward += 5000  # Huge bonus for Tetris!
    
    return shaped_reward
```

---

## Testing

### Validate Helper Functions

```python
def test_hole_counting():
    """Test hole detection"""
    # Board with 2 holes
    board = np.array([
        [0, 1, 0, 0, 0],
        [1, 0, 1, 0, 0],  # 1 hole at position (1,1)
        [1, 1, 1, 0, 0],
        [1, 0, 1, 1, 0],  # 1 hole at position (3,1)
        [1, 1, 1, 1, 0],
    ])
    
    holes = count_holes(board)
    assert holes == 2, f"Expected 2 holes, got {holes}"
    print("✅ Hole counting works!")

def test_bumpiness():
    """Test bumpiness calculation"""
    heights = [2, 5, 3, 7, 6]
    # Differences: |2-5|=3, |5-3|=2, |3-7|=4, |7-6|=1
    # Sum: 3 + 2 + 4 + 1 = 10
    
    bumpiness = calculate_bumpiness(heights)
    assert bumpiness == 10, f"Expected 10, got {bumpiness}"
    print("✅ Bumpiness calculation works!")
```

### Validate Reward Functions

```python
def test_no_exploit():
    """Ensure reward function can't be exploited"""
    
    # Simulate rotating forever
    total_rotation_reward = 0
    for _ in range(100):
        # No lines cleared
        reward = balanced_reward_shaping(
            obs, ACTION_ROTATE, 0, False, {'lines_cleared': 0}
        )
        total_rotation_reward += reward
    
    # Should be negative!
    assert total_rotation_reward < 0, "Rotation exploit not fixed!"
    print(f"✅ No rotation exploit: 100 rotations = {total_rotation_reward:.1f}")
    
    # Simulate clearing lines
    line_reward = balanced_reward_shaping(
        obs, ACTION_HARD_DROP, 0, False, {'lines_cleared': 1}
        )
    
    # Should be much better than rotating!
    assert line_reward > abs(total_rotation_reward), "Lines not rewarding enough!"
    print(f"✅ Line clearing better: 1 line = +{line_reward:.1f}")
```

---

## Common Issues

### Issue 1: Rewards Too Negative

**Symptom**: Agent never improves, total rewards always < -1000

**Diagnosis:**
```python
# Check reward components
print(f"Base: {reward}")
print(f"Holes: {-holes * 5}")
print(f"Height: {-max_height * 1}")
print(f"Bumpiness: {-bumpiness * 0.5}")
print(f"Total: {shaped_reward}")
```

**Solutions:**
- Reduce penalty weights
- Increase bonus weights
- Check board normalization (should be [0, 1])
- Try positive shaping mode

---

### Issue 2: No Learning

**Symptom**: Agent behavior doesn't change over time

**Diagnosis:**
```python
# Verify shaping is applied
print(f"Original reward: {env_reward}")
print(f"Shaped reward: {shaped_reward}")
print(f"Difference: {shaped_reward - env_reward}")

# Should see significant differences!
```

**Solutions:**
- Increase reward magnitude
- Verify shaping function is called
- Test board extraction works
- Check for None returns (should always return float!)

---

### Issue 3: Agent Exploits Reward Function

**Symptom**: High shaped rewards but poor actual performance

**Signs:**
- Uses one action 90%+ of the time
- Doesn't clear lines but gets positive rewards
- Finds unexpected "loopholes"

**Diagnosis:**
```python
# Watch agent and check action distribution
python watch_agent.py --model models/checkpoint_latest.pth

# Should see balanced action usage
```

**Solutions:**
1. Add progress penalty (e.g., -1 for no lines)
2. Remove per-step survival bonuses
3. Ensure highest rewards require actual progress
4. Test for exploits manually
5. Retrain from scratch after fixing

---

## Lessons Learned

### October 2025 Training Experience

**Key Discoveries:**

1. **Per-Step Rewards are Dangerous**
   - Giving +2 for "staying alive" caused rotation exploit
   - Agent optimized staying alive, not playing Tetris
   - Lesson: Only reward actual progress

2. **Always Add Progress Penalty**
   - Small penalty (-1) for steps without lines
   - Prevents time-wasting strategies
   - Forces agent to make progress

3. **Test for Exploits Early**
   - Watch agent after 500-1000 episodes
   - If using one action >80%, investigate
   - Don't wait for 2000 episodes to discover exploits

4. **Exploits Require Fresh Start**
   - Can't fix exploited model by changing rewards
   - Must train from scratch with fixed rewards
   - Old model "memorized" the wrong strategy

5. **Exploration vs Exploitation Split**
   - Just because exploration works doesn't mean learning works
   - Must verify Q-network learns correct behavior
   - Use verification scripts early and often

### Best Practices Established

✅ **DO:**
- Start rewards at 0, build up from progress
- Penalize time-wasting (-1 per step without lines)
- Make line clearing rewards dominant (1000+)
- Test exploitation scenarios manually
- Watch agent behavior frequently during training
- Verify all return paths return floats (never None)

❌ **DON'T:**
- Give per-step survival bonuses
- Allow positive rewards without progress
- Assume agent will "figure it out"
- Trust metrics alone - watch actual behavior
- Continue training if exploit discovered - restart fresh

---

## Summary

**Reward shaping is critical for Tetris RL success**

**Key Points:**
1. Sparse base rewards → Need dense shaping
2. Multiple metrics → Holes, height, bumpiness, distribution
3. Balanced approach → Reward progress, penalize waste
4. Anti-exploitation → Progress required, time penalties
5. Always returns float → No None values
6. Tested thoroughly → Manual verification required

**Current Recommended Function: `balanced_reward_shaping`**
- Fixed to prevent rotation exploit
- Balances positive and negative signals
- Progress penalty prevents time-wasting
- Exponential line rewards (1000-12000)
- Board quality penalties keep strategy sound

**Integration:**
- Called every step in training loop
- Receives obs, action, reward, done, info
- Returns shaped reward for agent learning
- Does not affect environment, only agent

---

*Last Updated: October 2025 - After discovering and fixing rotation exploit*
*Status: Production-ready with anti-exploitation measures*

---