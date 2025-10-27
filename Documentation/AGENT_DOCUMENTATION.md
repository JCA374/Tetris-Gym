# AGENT_DOCUMENTATION.md

# DQN Agent Component - Detailed Documentation

## Overview

The Agent (`src/agent.py`) is the core learning component that implements the Deep Q-Network (DQN) algorithm. It manages action selection, experience storage, neural network training, and exploration strategy.

---

## Architecture

### Class Hierarchy
```
Agent
├── Q-Network (Policy Network)
├── Target Network (Frozen Copy)
├── Optimizer (Adam)
├── Replay Buffer (Deque)
└── Exploration Strategy (ε-greedy)
```

### Initialization Parameters

```python
Agent(
    obs_space,              # Observation space (gym.spaces)
    action_space,           # Action space (gym.spaces.Discrete)
    lr=0.001,              # Learning rate
    gamma=0.99,            # Discount factor
    batch_size=32,         # Training batch size
    memory_size=10000,     # Replay buffer capacity
    target_update=1000,    # Target network update frequency
    model_type="dqn",      # Neural network architecture
    epsilon_start=1.0,     # Initial exploration rate
    epsilon_end=0.01,      # Minimum exploration rate
    epsilon_decay=0.9999,  # Exploration decay factor
    epsilon_decay_method="exponential",  # Decay method
    reward_shaping="balanced",           # Reward shaping mode
    max_episodes=10000,    # Total training episodes
    device=None            # Compute device (auto-detect if None)
)
```

---

## Core Attributes

### Neural Networks

#### 1. Policy Network (`self.q_network`)
- **Purpose**: Estimates Q-values for action selection
- **Updated**: Every training step
- **Type**: PyTorch nn.Module (ConvDQN or MLPDQN)
- **Output**: Tensor of shape (batch_size, n_actions)

```python
Q(s, a) = q_network(s)[a]
```

#### 2. Target Network (`self.target_network`)
- **Purpose**: Provides stable Q-value targets
- **Updated**: Every `target_update` steps
- **Relationship**: Periodic copy of q_network
- **Why Needed**: Prevents chasing moving target during training

```python
y = r + γ * max(target_network(s'))
```

### Memory and Experience

#### Replay Buffer (`self.memory`)
- **Type**: `collections.deque` with `maxlen`
- **Capacity**: Defined by `memory_size` parameter
- **Contents**: Tuples of `(state, action, reward, next_state, done)`
- **Behavior**: FIFO (First In, First Out) when full

**Why Experience Replay?**
1. Breaks correlation between consecutive experiences
2. Improves data efficiency (reuse experiences)
3. Stabilizes training
4. Enables mini-batch learning

**Storage Logic**:
```python
def remember(self, state, action, reward, next_state, done):
    """
    Store experience in replay buffer.
    Oldest experience removed automatically when buffer full.
    """
    self.memory.append((state, action, reward, next_state, done))
```

### Exploration Strategy

#### Epsilon (ε) - Exploration Rate
- **Purpose**: Controls exploration vs exploitation trade-off
- **Range**: [epsilon_end, epsilon_start] (e.g., [0.01, 1.0])
- **Direction**: Decreases over time
- **Storage**: `self.epsilon`

**Three Decay Methods**:

1. **Exponential Decay** (default):
```python
ε_new = max(ε_min, ε * decay)
```
- Smooth, gradual decay
- Never reaches exactly ε_min
- Good for continuous exploration

2. **Linear Decay**:
```python
ε_new = max(ε_min, ε - decay_rate)
```
- Predictable, uniform decay
- Reaches ε_min at specific episode
- Good for time-limited training

3. **Adaptive Decay**:
```python
ε = schedule[episode]
```
- Pre-computed custom schedule
- Allows non-monotonic decay
- Can increase ε temporarily

### Training Counters

#### `self.steps_done`
- **Purpose**: Total number of training steps performed
- **Incremented**: After each `learn()` call
- **Used For**: 
  - Target network update timing
  - Learning rate scheduling
  - Checkpoint frequency

#### `self.episodes_done`
- **Purpose**: Total number of episodes completed
- **Incremented**: After each episode ends
- **Used For**:
  - Epsilon decay
  - Progress tracking
  - Checkpoint naming

---

## Core Methods

### 1. Action Selection

```python
def select_action(self, state, training=True):
    """
    Select action using ε-greedy policy.
    
    Args:
        state: Current game state (observation)
        training: If True, use exploration; if False, pure exploitation
    
    Returns:
        action: Integer action index
    """
```

**Logic Flow**:
```
1. Preprocess state → tensor
2. If training mode:
   a. Generate random number r ∈ [0, 1]
   b. If r < ε:
      - Select random action (explore)
   c. Else:
      - Forward pass through Q-network
      - Select action with max Q-value (exploit)
3. If evaluation mode:
   - Always select action with max Q-value
4. Return action
```

**Code Structure**:
```python
if training and random.random() < self.epsilon:
    # Explore: random action
    action = self.action_space.sample()
else:
    # Exploit: best known action
    state_tensor = self._preprocess_state(state)
    with torch.no_grad():
        q_values = self.q_network(state_tensor)
    action = q_values.max(1)[1].item()

return action
```

**Why ε-greedy?**
- **Early Training**: High ε → more exploration → discover strategies
- **Late Training**: Low ε → more exploitation → refine strategies
- **Evaluation**: ε = 0 → pure exploitation → test learned policy

### 2. Experience Storage

```python
def remember(self, state, action, reward, next_state, done):
    """
    Store experience tuple in replay buffer.
    
    Args:
        state: Current state
        action: Action taken
        reward: Reward received
        next_state: Resulting state
        done: Episode termination flag
    """
```

**Memory Management**:
- Automatic overflow handling (deque with maxlen)
- No manual deletion needed
- FIFO replacement when full

**Experience Tuple**:
```python
experience = (
    state,      # np.array or dict
    action,     # int
    reward,     # float
    next_state, # np.array or dict
    done        # bool
)
```

### 3. Learning from Experience

```python
def learn(self):
    """
    Train Q-network using experience replay.
    
    Returns:
        dict: Training metrics (loss, etc.)
    """
```

**Detailed Algorithm**:

#### Step 1: Check Buffer Size
```python
if len(self.memory) < self.batch_size:
    return  # Not enough experiences yet
```

#### Step 2: Sample Mini-Batch
```python
batch = random.sample(self.memory, self.batch_size)
states, actions, rewards, next_states, dones = zip(*batch)
```
- Random sampling breaks temporal correlation
- Batch size (e.g., 32) balances speed vs stability

#### Step 3: Convert to Tensors
```python
states = torch.FloatTensor(np.array(states)).to(self.device)
actions = torch.LongTensor(actions).to(self.device)
rewards = torch.FloatTensor(rewards).to(self.device)
next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
dones = torch.FloatTensor(dones).to(self.device)
```

#### Step 4: Compute Current Q-Values
```python
current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
```
- `gather()` selects Q-value for action taken
- Shape: (batch_size, 1)

#### Step 5: Compute Target Q-Values
```python
with torch.no_grad():
    next_q_values = self.target_network(next_states).max(1)[0]
    target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
```
- Use target network (frozen weights)
- Bellman equation: Q(s,a) = r + γ * max Q(s',a')
- Terminal states: Q(s,a) = r (no future)

#### Step 6: Compute Loss
```python
loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
```
- Mean Squared Error between predicted and target Q-values
- Measures how well network predicts future rewards

#### Step 7: Backpropagation
```python
self.optimizer.zero_grad()  # Clear gradients
loss.backward()             # Compute gradients
```

#### Step 8: Gradient Clipping
```python
torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
```
- Prevents exploding gradients
- Clips gradient norm to max of 1.0
- Stabilizes training

#### Step 9: Optimizer Step
```python
self.optimizer.step()  # Update weights
```

#### Step 10: Update Counters
```python
self.steps_done += 1
```

#### Step 11: Update Target Network
```python
if self.steps_done % self.target_update == 0:
    self.target_network.load_state_dict(self.q_network.state_dict())
```
- Periodic copying (e.g., every 1000 steps)
- Provides stable learning targets

### 4. Exploration Decay

```python
def update_epsilon(self, episode):
    """
    Decay exploration rate based on training progress.
    
    Args:
        episode: Current episode number
    """
```

**Exponential Decay Implementation**:
```python
if self.epsilon_decay_method == "exponential":
    self.epsilon = max(
        self.epsilon_end,
        self.epsilon * self.epsilon_decay
    )
```

**Why Decay?**
- **Early**: Need exploration to discover good strategies
- **Late**: Should exploit learned strategies
- **Balance**: Always maintain minimum exploration (avoid local optima)

**Decay Curve Example**:
```
Episode    Epsilon (decay=0.9999)
0          1.0000
1000       0.9048
2000       0.8187
5000       0.6065
10000      0.3679
20000      0.1353
```

### 5. Checkpoint Management

#### Save Checkpoint
```python
def save_checkpoint(self, filename=None, model_dir="models/"):
    """
    Save complete agent state to disk.
    
    Args:
        filename: Checkpoint name (auto-generated if None)
        model_dir: Directory to save checkpoint
    
    Returns:
        checkpoint_path: Path to saved file
    """
```

**Saved Components**:
```python
checkpoint = {
    'episode': self.episodes_done,
    'steps': self.steps_done,
    'epsilon': self.epsilon,
    'q_network_state': self.q_network.state_dict(),
    'target_network_state': self.target_network.state_dict(),
    'optimizer_state': self.optimizer.state_dict(),
    'hyperparameters': {
        'lr': self.lr,
        'gamma': self.gamma,
        'batch_size': self.batch_size,
        'memory_size': len(self.memory),
        # ...
    }
}
torch.save(checkpoint, path)
```

#### Load Checkpoint
```python
def load_checkpoint(self, filename=None, model_dir="models/", latest=False):
    """
    Load agent state from checkpoint.
    
    Args:
        filename: Checkpoint name
        model_dir: Directory containing checkpoint
        latest: If True, load most recent checkpoint
    
    Returns:
        success: True if loaded successfully
    """
```

**Loading Logic**:
```python
checkpoint = torch.load(path)
self.q_network.load_state_dict(checkpoint['q_network_state'])
self.target_network.load_state_dict(checkpoint['target_network_state'])
self.optimizer.load_state_dict(checkpoint['optimizer_state'])
self.epsilon = checkpoint['epsilon']
self.steps_done = checkpoint['steps']
self.episodes_done = checkpoint['episode']
```

---

## State Preprocessing

### Purpose
Convert raw observations into neural network-compatible tensors.

### Methods

#### `_preprocess_state(state)`
**For Single States**:
```python
def _preprocess_state(self, state):
    """
    Convert single state to tensor.
    
    Input: np.array or dict
    Output: torch.Tensor with shape (1, ...)
    """
    if not isinstance(state, np.ndarray):
        state = np.array(state)
    
    # Add batch dimension
    state = state[np.newaxis, ...]
    
    # Convert to tensor
    state_tensor = torch.FloatTensor(state).to(self.device)
    
    return state_tensor
```

#### `_preprocess_batch(batch)`
**For Batches**:
```python
def _preprocess_batch(self, batch):
    """
    Convert batch of states to tensor.
    
    Input: List of states
    Output: torch.Tensor with shape (batch_size, ...)
    """
    # Already properly shaped
    if len(batch.shape) == 4:  # (B, C, H, W)
        return batch
    
    # Need reshaping
    return batch.reshape(...)
```

---

## Advanced Features

### 1. Adaptive Epsilon Schedule

```python
def _create_adaptive_schedule(self, max_episodes):
    """
    Create custom epsilon schedule based on training phases.
    
    Returns:
        schedule: Dict mapping episode → epsilon
    """
```

**Example Schedule**:
```python
schedule = {}
# Phase 1: High exploration (0-20%)
for ep in range(int(0.2 * max_episodes)):
    schedule[ep] = 1.0

# Phase 2: Gradual decay (20-80%)
for ep in range(int(0.2 * max_episodes), int(0.8 * max_episodes)):
    progress = (ep - 0.2*max_episodes) / (0.6*max_episodes)
    schedule[ep] = 1.0 - progress * 0.9

# Phase 3: Low exploration (80-100%)
for ep in range(int(0.8 * max_episodes), max_episodes):
    schedule[ep] = 0.1
```

### 2. Prioritized Experience Replay (Future Extension)

**Concept**: Sample experiences based on TD-error magnitude.

**Benefits**:
- Focuses on surprising/important experiences
- Faster learning on critical situations
- Better data efficiency

**Implementation Skeleton**:
```python
def remember_prioritized(self, state, action, reward, next_state, done):
    """Store with priority based on TD-error"""
    td_error = self._compute_td_error(state, action, reward, next_state, done)
    priority = abs(td_error) + epsilon
    self.memory.append((state, action, reward, next_state, done, priority))

def sample_prioritized(self, batch_size):
    """Sample based on priorities"""
    priorities = np.array([exp[-1] for exp in self.memory])
    probs = priorities / priorities.sum()
    indices = np.random.choice(len(self.memory), batch_size, p=probs)
    return [self.memory[i] for i in indices]
```

### 3. Double DQN (Future Extension)

**Problem**: Standard DQN overestimates Q-values.

**Solution**: Use policy network for action selection, target network for evaluation.

```python
# Standard DQN
next_q_values = target_network(next_states).max(1)[0]

# Double DQN
best_actions = q_network(next_states).max(1)[1]
next_q_values = target_network(next_states).gather(1, best_actions)
```

---

## Usage Examples

### Basic Training Loop

```python
from src.agent import Agent
from config import make_env

# Setup
env = make_env()
agent = Agent(env.observation_space, env.action_space)

# Training
for episode in range(1000):
    obs, info = env.reset()
    done = False
    episode_reward = 0
    
    while not done:
        # Select action
        action = agent.select_action(obs)
        
        # Environment step
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Store experience
        agent.remember(obs, action, reward, next_obs, done)
        
        # Learn
        if len(agent.memory) >= agent.batch_size:
            agent.learn()
        
        obs = next_obs
        episode_reward += reward
    
    # Update exploration
    agent.update_epsilon(episode)
    
    # Save checkpoint
    if episode % 500 == 0:
        agent.save_checkpoint()

env.close()
```

### Evaluation Mode

```python
# Load trained agent
agent.load_checkpoint(latest=True)

# Evaluate
env = make_env(render_mode="human")
obs, info = env.reset()
done = False

while not done:
    # Pure exploitation (no exploration)
    action = agent.select_action(obs, training=False)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()
```

---

## Performance Considerations

### Memory Management

**Replay Buffer Size**:
- **Too Small**: Limited diversity, overfitting
- **Too Large**: Memory overhead, slow sampling
- **Recommended**: 10,000 - 100,000 experiences

**Typical Memory Usage**:
```python
experience_size = state_size + action_size + reward_size + next_state_size + done_size
                = 944 + 1 + 1 + 944 + 1
                = 1891 floats/ints

buffer_memory = experience_size * memory_size * 4 bytes
              = 1891 * 100,000 * 4
              = ~758 MB
```

### Training Speed

**Factors**:
1. **Batch Size**: Larger = more stable but slower per step
2. **Network Size**: Deeper/wider = better capacity but slower
3. **Device**: GPU ~10-50x faster than CPU
4. **Environment**: Rendering slows training significantly

**Optimization Tips**:
- Use GPU for networks (agent trained on GPU)
- Disable rendering during training (`render_mode=None`)
- Increase batch size on powerful hardware
- Use pin_memory for faster GPU transfers

### Target Network Update Frequency

**Trade-offs**:
- **More Frequent** (e.g., every 100 steps):
  - Faster adaptation to new strategies
  - Less stable learning
  - Higher variance
  
- **Less Frequent** (e.g., every 10,000 steps):
  - More stable learning
  - Slower adaptation
  - Smoother convergence

**Recommended**: 1000 steps (good balance)

---

## Debugging and Diagnostics

### Common Issues

#### 1. Agent Not Learning
**Symptoms**: Epsilon decays but performance doesn't improve.

**Diagnosis**:
```python
# Check if learning is happening
print(f"Memory size: {len(agent.memory)}")
print(f"Steps done: {agent.steps_done}")
print(f"Epsilon: {agent.epsilon}")

# Check Q-values
with torch.no_grad():
    q_values = agent.q_network(state)
    print(f"Q-values: {q_values}")
```

**Solutions**:
- Ensure `learn()` is being called
- Check reward shaping (not too negative)
- Verify state preprocessing
- Increase exploration duration

#### 2. Training Instability
**Symptoms**: Loss spikes, NaN values, divergence.

**Diagnosis**:
```python
# Monitor loss values
metrics = agent.learn()
print(f"Loss: {metrics['loss']}")

# Check Q-value magnitudes
print(f"Q-values range: [{q_values.min()}, {q_values.max()}]")

# Check gradients
for name, param in agent.q_network.named_parameters():
    if param.grad is not None:
        print(f"{name} grad norm: {param.grad.norm()}")
```

**Solutions**:
- Enable gradient clipping (already implemented)
- Reduce learning rate
- Check for NaN in observations
- Normalize rewards

#### 3. Memory Leaks
**Symptoms**: RAM usage grows indefinitely.

**Diagnosis**:
```python
import psutil
process = psutil.Process()
print(f"Memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")
```

**Solutions**:
- Detach tensors before storing in memory
- Use `torch.no_grad()` for inference
- Clear unused variables
- Limit replay buffer size

---

## Testing

### Unit Tests

```python
def test_agent_initialization():
    """Test agent creates successfully"""
    env = make_env()
    agent = Agent(env.observation_space, env.action_space)
    assert agent.q_network is not None
    assert agent.target_network is not None
    assert len(agent.memory) == 0

def test_action_selection():
    """Test action selection works"""
    env = make_env()
    agent = Agent(env.observation_space, env.action_space)
    obs, _ = env.reset()
    
    action = agent.select_action(obs)
    assert 0 <= action < agent.n_actions

def test_experience_storage():
    """Test experience replay buffer"""
    env = make_env()
    agent = Agent(env.observation_space, env.action_space, memory_size=100)
    
    obs, _ = env.reset()
    for _ in range(150):  # More than buffer size
        action = agent.select_action(obs)
        next_obs, reward, done, _, _ = env.step(action)
        agent.remember(obs, action, reward, next_obs, done)
        obs = next_obs if not done else env.reset()[0]
    
    assert len(agent.memory) == 100  # Buffer limit respected

def test_learning():
    """Test learning step works"""
    env = make_env()
    agent = Agent(env.observation_space, env.action_space, batch_size=16)
    
    # Fill buffer
    obs, _ = env.reset()
    for _ in range(32):
        action = agent.select_action(obs)
        next_obs, reward, done, _, _ = env.step(action)
        agent.remember(obs, action, reward, next_obs, done)
        obs = next_obs if not done else env.reset()[0]
    
    # Learn
    metrics = agent.learn()
    assert 'loss' in metrics
    assert not np.isnan(metrics['loss'])
```

---

## Summary

The Agent component is the brain of the RL system, implementing:

1. **DQN Algorithm**: Q-learning with neural network function approximation
2. **Experience Replay**: Breaks temporal correlations, improves data efficiency
3. **Target Network**: Stabilizes learning with frozen Q-value targets
4. **ε-Greedy Exploration**: Balances exploration and exploitation
5. **Checkpoint System**: Enables training resumption and model deployment

**Key Design Principles**:
- **Modularity**: Clear separation of concerns (action selection, learning, storage)
- **Flexibility**: Configurable hyperparameters and architectures
- **Robustness**: Gradient clipping, error handling, validation
- **Efficiency**: GPU support, batch processing, optimized data structures

**Integration Points**:
- **Environment**: Receives observations, sends actions
- **Model**: Uses neural networks for Q-value approximation
- **Logger**: Reports training metrics
- **Config**: Receives hyperparameters and environment setup

---

*End of Agent Documentation*
