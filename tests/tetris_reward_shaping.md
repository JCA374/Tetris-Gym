# Tetris Reward Shaping & Helper Functions

## Current Reward Structure Analysis

### **Tetris Gymnasium Default Rewards**
```python
# Standard Tetris scoring (what we currently get):
- Line clears: 40, 100, 300, 1200 (1, 2, 3, 4 lines)
- Soft drop: +1 per cell
- Hard drop: +2 per cell  
- Game over: Large negative penalty
- Step penalty: Small negative (time pressure)
```

### **Problems with Default Rewards**
1. **Sparse Rewards**: Only get reward when lines are cleared
2. **No Strategic Guidance**: No reward for good placement, hole avoidance
3. **Short-term Focus**: No incentive for long-term board management
4. **Hard Exploration**: Agent may never discover line clearing initially

## Recommended Reward Shaping Strategy

### **1. Core Tetris Heuristics**

#### **Height-Based Penalties**
```python
def calculate_height_penalty(board):
    """Penalize tall stacks - encourages keeping board low"""
    heights = []
    for col in range(board.shape[1]):
        # Find highest filled cell in each column
        height = 0
        for row in range(board.shape[0]):
            if board[row, col] != 0:
                height = board.shape[0] - row
                break
        heights.append(height)
    
    # Penalties
    max_height = max(heights) if heights else 0
    avg_height = sum(heights) / len(heights) if heights else 0
    
    return {
        'max_height_penalty': -max_height * 0.5,
        'avg_height_penalty': -avg_height * 0.1,
        'height_variance': -np.var(heights) * 0.1  # Prefer flat top
    }
```

#### **Hole Detection & Penalties**
```python
def count_holes(board):
    """Count holes (empty cells with filled cells above)"""
    holes = 0
    for col in range(board.shape[1]):
        found_filled = False
        for row in range(board.shape[0]):
            if board[row, col] != 0:
                found_filled = True
            elif found_filled and board[row, col] == 0:
                holes += 1
    return holes

def calculate_hole_penalty(board):
    """Heavy penalty for creating holes"""
    holes = count_holes(board)
    return -holes * 2.0  # Strong penalty
```

#### **Line Clear Bonuses**
```python
def calculate_line_clear_bonus(lines_cleared, total_lines_cleared):
    """Enhanced rewards for line clearing"""
    base_rewards = {0: 0, 1: 40, 2: 100, 3: 300, 4: 1200}
    
    # Base reward
    reward = base_rewards.get(lines_cleared, 0)
    
    # Combo bonuses
    if lines_cleared > 0:
        # Tetris bonus (4 lines)
        if lines_cleared == 4:
            reward += 200  # Extra Tetris bonus
        
        # Efficiency bonus (encourage fewer pieces per line)
        efficiency_bonus = lines_cleared * 10
        reward += efficiency_bonus
    
    return reward
```

#### **Board Quality Metrics**
```python
def calculate_board_quality(board):
    """Assess overall board quality"""
    
    # Bumpiness (height differences between adjacent columns)
    heights = get_column_heights(board)
    bumpiness = sum(abs(heights[i] - heights[i+1]) 
                   for i in range(len(heights)-1))
    
    # Well depth (deep single-column gaps)
    wells = 0
    for i in range(len(heights)):
        left_height = heights[i-1] if i > 0 else 0
        right_height = heights[i+1] if i < len(heights)-1 else 0
        well_depth = min(left_height, right_height) - heights[i]
        wells += max(0, well_depth)
    
    return {
        'bumpiness_penalty': -bumpiness * 0.1,
        'wells_penalty': -wells * 0.3,
        'smoothness_bonus': 1.0 / (1 + bumpiness)  # Reward smooth tops
    }
```

### **2. Strategic Helper Functions**

#### **Piece Placement Analysis**
```python
def analyze_placement_quality(board, piece_type, position, rotation):
    """Analyze quality of a piece placement"""
    
    # Simulate placement
    test_board = board.copy()
    place_piece(test_board, piece_type, position, rotation)
    
    # Calculate metrics
    lines_cleared = count_cleared_lines(test_board)
    holes_created = count_holes(test_board) - count_holes(board)
    height_increase = max_height(test_board) - max_height(board)
    
    # Quality score
    quality = 0
    quality += lines_cleared * 10  # Line clears are good
    quality -= holes_created * 5   # Holes are bad
    quality -= height_increase * 2 # Height increase is bad
    
    return quality

def get_all_possible_placements(board, piece_type):
    """Get all valid placements for current piece"""
    placements = []
    
    for rotation in range(4):  # Try all rotations
        for col in range(board.shape[1]):
            if is_valid_placement(board, piece_type, col, rotation):
                quality = analyze_placement_quality(board, piece_type, col, rotation)
                placements.append({
                    'position': col,
                    'rotation': rotation,
                    'quality': quality
                })
    
    return sorted(placements, key=lambda x: x['quality'], reverse=True)
```

#### **Future Planning Helpers**
```python
def calculate_accessibility(board):
    """Measure how accessible filled areas are"""
    # Count cells that are reachable from top
    accessible = 0
    visited = set()
    
    def dfs(row, col):
        if (row, col) in visited or row < 0 or col < 0:
            return 0
        if row >= board.shape[0] or col >= board.shape[1]:
            return 0
        if board[row, col] == 0:
            return 0
            
        visited.add((row, col))
        count = 1
        
        # Check 4 directions
        for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
            count += dfs(row + dr, col + dc)
        
        return count
    
    # Start from top row
    for col in range(board.shape[1]):
        if board[0, col] != 0:
            accessible += dfs(0, col)
    
    total_filled = np.sum(board != 0)
    return accessible / max(1, total_filled)  # Accessibility ratio
```

### **3. Advanced Reward Shaping**

#### **Multi-Objective Reward Function**
```python
class TetrisRewardShaper:
    def __init__(self):
        self.weights = {
            # Primary objectives
            'lines_cleared': 10.0,
            'game_over': -100.0,
            
            # Board management
            'max_height': -0.5,
            'holes': -2.0,
            'bumpiness': -0.1,
            'wells': -0.3,
            
            # Strategic bonuses
            'tetris_bonus': 5.0,
            'efficiency': 0.5,
            'accessibility': 1.0,
            
            # Survival
            'step_reward': 0.01,  # Small bonus for staying alive
        }
        
        self.previous_board = None
        self.total_lines = 0
    
    def calculate_reward(self, obs, action, reward, done, info):
        """Calculate shaped reward"""
        shaped_reward = reward  # Start with base reward
        
        # Extract board from observation
        board = self.extract_board(obs)
        
        if self.previous_board is not None:
            # Calculate changes
            lines_cleared = info.get('lines_cleared', 0)
            
            # Core penalties/bonuses
            height_penalty = calculate_height_penalty(board)
            hole_penalty = calculate_hole_penalty(board)
            quality_metrics = calculate_board_quality(board)
            
            # Apply weights
            for metric, value in height_penalty.items():
                shaped_reward += value * self.weights.get(metric.split('_')[0], 0)
            
            shaped_reward += hole_penalty * self.weights['holes']
            
            for metric, value in quality_metrics.items():
                key = metric.split('_')[0]
                shaped_reward += value * self.weights.get(key, 0)
            
            # Line clear bonuses
            if lines_cleared > 0:
                self.total_lines += lines_cleared
                
                # Tetris bonus
                if lines_cleared == 4:
                    shaped_reward += self.weights['tetris_bonus']
                
                # Efficiency bonus
                efficiency = lines_cleared / max(1, info.get('pieces_placed', 1))
                shaped_reward += efficiency * self.weights['efficiency']
            
            # Survival bonus
            if not done:
                shaped_reward += self.weights['step_reward']
        
        self.previous_board = board.copy()
        return shaped_reward
    
    def extract_board(self, obs):
        """Extract board from observation dict"""
        if isinstance(obs, dict) and 'board' in obs:
            return obs['board']
        else:
            # Handle flattened observations
            # This depends on your observation structure
            return self.reconstruct_board_from_flat(obs)
    
    def reset(self):
        """Reset for new episode"""
        self.previous_board = None
        self.total_lines = 0
```

### **4. Implementation Strategy**

#### **Phase 1: Basic Shaping**
```python
# Start with simple height and hole penalties
def basic_reward_shaping(obs, reward, done, info):
    if done:
        return reward - 50  # Game over penalty
    
    board = extract_board(obs)
    
    # Simple penalties
    max_height = get_max_height(board)
    holes = count_holes(board)
    
    shaped_reward = reward
    shaped_reward -= max_height * 0.2  # Height penalty
    shaped_reward -= holes * 1.0       # Hole penalty
    shaped_reward += 0.01              # Survival bonus
    
    return shaped_reward
```

#### **Phase 2: Strategic Shaping**
```python
# Add bumpiness, wells, and efficiency metrics
def strategic_reward_shaping(obs, reward, done, info):
    # ... basic shaping +
    
    bumpiness = calculate_bumpiness(board)
    wells = calculate_wells(board)
    lines_cleared = info.get('lines_cleared', 0)
    
    shaped_reward -= bumpiness * 0.1
    shaped_reward -= wells * 0.3
    
    # Tetris bonus
    if lines_cleared == 4:
        shaped_reward += 10
    
    return shaped_reward
```

#### **Phase 3: Advanced Shaping**
```python
# Full multi-objective shaping with learning
reward_shaper = TetrisRewardShaper()

def advanced_reward_shaping(obs, reward, done, info):
    return reward_shaper.calculate_reward(obs, action, reward, done, info)
```

### **5. Curriculum Learning**

#### **Progressive Difficulty**
```python
class TetrisCurriculum:
    def __init__(self):
        self.stage = 0
        self.episode_count = 0
    
    def get_env_config(self):
        """Return environment config for current stage"""
        stages = [
            # Stage 0: Small board, slow speed
            {'board_height': 10, 'board_width': 6, 'gravity': 1.0},
            
            # Stage 1: Normal board, slow speed  
            {'board_height': 20, 'board_width': 10, 'gravity': 1.0},
            
            # Stage 2: Normal board, normal speed
            {'board_height': 20, 'board_width': 10, 'gravity': 2.0},
            
            # Stage 3: Full complexity
            {'board_height': 20, 'board_width': 10, 'gravity': 3.0}
        ]
        
        return stages[min(self.stage, len(stages)-1)]
    
    def update(self, episode_reward, lines_cleared):
        """Update curriculum based on performance"""
        self.episode_count += 1
        
        # Advance stages based on performance thresholds
        if (self.stage == 0 and lines_cleared >= 5 and self.episode_count >= 100):
            self.stage = 1
            print("Advancing to Stage 1: Normal board size")
        elif (self.stage == 1 and lines_cleared >= 10 and self.episode_count >= 300):
            self.stage = 2
            print("Advancing to Stage 2: Normal speed")
        elif (self.stage == 2 and lines_cleared >= 20 and self.episode_count >= 500):
            self.stage = 3
            print("Advancing to Stage 3: Full complexity")
```

### **6. Integration with Existing Code**

#### **Modified Agent Class**
```python
# In src/agent.py - add reward shaping
class Agent:
    def __init__(self, ..., use_reward_shaping=True):
        # ... existing init ...
        
        if use_reward_shaping:
            self.reward_shaper = TetrisRewardShaper()
        else:
            self.reward_shaper = None
    
    def remember(self, state, action, reward, next_state, done, info=None):
        """Store experience with shaped reward"""
        if self.reward_shaper:
            reward = self.reward_shaper.calculate_reward(
                state, action, reward, done, info or {}
            )
        
        self.memory.append((state, action, reward, next_state, done))
```

#### **Modified Training Loop**
```python
# In train.py - integrate reward shaping
def train_with_shaping(args):
    # ... setup ...
    
    reward_shaper = TetrisRewardShaper()
    curriculum = TetrisCurriculum()
    
    for episode in range(start_episode, args.episodes):
        # Update environment based on curriculum
        env_config = curriculum.get_env_config()
        if episode % 100 == 0:  # Update env periodically
            env = make_env(**env_config)
        
        obs, info = env.reset()
        episode_reward = 0
        total_lines = 0
        
        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Apply reward shaping
            shaped_reward = reward_shaper.calculate_reward(
                obs, action, reward, terminated or truncated, info
            )
            
            agent.remember(obs, action, shaped_reward, next_obs, 
                          terminated or truncated, info)
            
            episode_reward += shaped_reward
            total_lines += info.get('lines_cleared', 0)
            
            # ... rest of training loop ...
        
        # Update curriculum
        curriculum.update(episode_reward, total_lines)
        reward_shaper.reset()
```

### **7. Recommended Implementation Order**

1. **Week 1**: Implement basic height and hole penalties
2. **Week 2**: Add bumpiness and wells calculations  
3. **Week 3**: Implement line clear bonuses and efficiency metrics
4. **Week 4**: Add curriculum learning
5. **Week 5**: Fine-tune weights and advanced features

### **8. Expected Impact**

With proper reward shaping, you should see:
- **Faster convergence**: 200-400 episodes vs 800-1200 without shaping
- **Better performance**: Higher line clearing rates and longer games
- **More stable learning**: Less variance in episode rewards
- **Strategic play**: Agent learns to avoid holes and manage height

The key is to start simple and gradually add complexity as the agent improves!