# INTEGRATION_AND_WORKFLOW.md

# Integration and Workflow - Complete System Guide

## Overview

This document explains how all components of the Tetris RL project work together, providing a complete view of the system's operation from initialization through training to evaluation.

---

## Table of Contents
1. [System Initialization](#system-initialization)
2. [Training Workflow](#training-workflow)
3. [Component Interactions](#component-interactions)
4. [Data Flow Diagrams](#data-flow-diagrams)
5. [State Management](#state-management)
6. [Error Handling](#error-handling)
7. [Performance Optimization](#performance-optimization)
8. [Debugging Guide](#debugging-guide)

---

## System Initialization

### Complete Startup Sequence

```
1. train.py starts
   ↓
2. Parse command-line arguments
   ├─ Training parameters (episodes, lr, gamma)
   ├─ Model parameters (architecture)
   ├─ Exploration parameters (epsilon)
   └─ Reward shaping selection
   ↓
3. Create directories
   ├─ models/ (for checkpoints)
   └─ logs/ (for metrics)
   ↓
4. Initialize Environment
   ├─ config.make_env()
   │   ├─ Create Tetris Gymnasium environment
   │   ├─ Discover action meanings (dynamic)
   │   ├─ Wrap with preprocessing
   │   └─ Return configured environment
   ↓
5. Initialize Agent
   ├─ Agent.__init__()
   │   ├─ model.create_model() → Q-network
   │   ├─ model.create_model() → Target network
   │   ├─ Initialize optimizer (Adam)
   │   ├─ Initialize replay buffer (deque)
   │   └─ Set hyperparameters
   ↓
6. Load Checkpoint (if resuming)
   ├─ Agent.load_checkpoint()
   │   ├─ Load Q-network weights
   │   ├─ Load target network weights
   │   ├─ Load optimizer state
   │   ├─ Restore epsilon, steps, episodes
   │   └─ Return start episode
   ↓
7. Initialize Logger
   ├─ TrainingLogger.__init__()
   │   ├─ Create experiment directory
   │   ├─ Initialize CSV file
   │   └─ Setup plotting
   ↓
8. Select Reward Shaper
   ├─ Map argument to function
   └─ Store reference
   ↓
9. Print Configuration
   ├─ Environment info
   ├─ Agent parameters
   ├─ Action mappings
   └─ Device (CPU/GPU)
   ↓
10. Start Training Loop
```

### Code Flow

```python
# train.py main execution

def main():
    # 1. Parse arguments
    args = parse_args()
    
    # 2. Create directories
    make_dir(MODEL_DIR)
    make_dir(LOG_DIR)
    
    # 3. Initialize environment
    env = make_env(render_mode="rgb_array")
    discover_action_meanings(env)
    
    # 4. Initialize agent
    agent = Agent(
        obs_space=env.observation_space,
        action_space=env.action_space,
        lr=args.lr,
        gamma=args.gamma,
        # ... other params
    )
    
    # 5. Load checkpoint (optional)
    start_episode = 0
    if args.resume:
        if agent.load_checkpoint(latest=True):
            start_episode = agent.episodes_done
    
    # 6. Initialize logger
    experiment_name = args.experiment_name or generate_name()
    logger = TrainingLogger(LOG_DIR, experiment_name)
    
    # 7. Select reward shaper
    shaper_fn = shaping_functions[args.reward_shaping]
    
    # 8. Print configuration
    print_config(env, agent, args)
    
    # 9. Start training
    train_loop(env, agent, logger, shaper_fn, args)
```

---

## Training Workflow

### Episode Structure

```
┌──────────────────────────────────────────┐
│          EPISODE LIFECYCLE               │
├──────────────────────────────────────────┤
│                                          │
│  START EPISODE                           │
│    ↓                                     │
│  1. Environment Reset                    │
│     obs, info = env.reset()              │
│     episode_reward = 0                   │
│     episode_steps = 0                    │
│     episode_lines = 0                    │
│                                          │
│  ┌────────────────────────────────┐     │
│  │   STEP LOOP (until done)       │     │
│  │                                 │     │
│  │  2. Select Action               │     │
│  │     action = agent.select_action()   │
│  │                                 │     │
│  │  3. Execute Action              │     │
│  │     next_obs, reward, done, info =   │
│  │       env.step(action)          │     │
│  │                                 │     │
│  │  4. Apply Reward Shaping        │     │
│  │     shaped_reward = shaper(...)  │     │
│  │                                 │     │
│  │  5. Store Experience            │     │
│  │     agent.remember(...)          │     │
│  │                                 │     │
│  │  6. Learn (if enough data)      │     │
│  │     if len(memory) >= batch:    │     │
│  │         agent.learn()           │     │
│  │                                 │     │
│  │  7. Update State                │     │
│  │     obs = next_obs              │     │
│  │     episode_reward += reward    │     │
│  │     episode_steps += 1          │     │
│  │                                 │     │
│  └────────────────────────────────┘     │
│                                          │
│  8. Extract Metrics                      │
│     episode_lines = extract_lines(info)  │
│                                          │
│  9. Update Exploration                   │
│     agent.update_epsilon(episode)        │
│                                          │
│  10. Log Episode                         │
│      logger.log_episode(...)             │
│                                          │
│  11. Save Checkpoint (periodic)          │
│      if episode % save_freq == 0:        │
│          agent.save_checkpoint()         │
│                                          │
│  12. Plot Progress (periodic)            │
│      if episode % plot_freq == 0:        │
│          logger.plot_progress()          │
│                                          │
│  END EPISODE                             │
│                                          │
└──────────────────────────────────────────┘
```

### Detailed Step Breakdown

#### Step 1: Environment Reset
```python
obs, info = env.reset(seed=seed)

# Returns:
# - obs: dict with 'board', 'holder', 'queue'
# - info: dict with metadata
```

#### Step 2: Action Selection
```python
action = agent.select_action(obs, training=True)

# Process:
# 1. Preprocess obs → tensor
# 2. If training and random() < epsilon:
#        action = random
#    else:
#        action = argmax(Q(obs))
# 3. Return action (int)
```

#### Step 3: Execute Action
```python
next_obs, reward, terminated, truncated, info = env.step(action)
done = terminated or truncated

# Returns:
# - next_obs: Next observation
# - reward: Environment reward (game score)
# - terminated: Episode ended naturally
# - truncated: Episode ended by time limit
# - info: Additional information
```

#### Step 4: Reward Shaping
```python
shaped_reward = shaper_fn(obs, action, reward, done, info)

# Process:
# 1. Extract board from obs
# 2. Compute metrics (holes, height, etc.)
# 3. Calculate bonuses/penalties
# 4. Return modified reward
```

#### Step 5: Store Experience
```python
agent.remember(obs, action, shaped_reward, next_obs, done)

# Stores tuple in replay buffer:
# (state, action, reward, next_state, done)
```

#### Step 6: Learn
```python
if len(agent.memory) >= agent.batch_size:
    loss_info = agent.learn()

# Process:
# 1. Sample batch from memory
# 2. Compute Q-values (policy network)
# 3. Compute targets (target network)
# 4. Calculate loss
# 5. Backpropagate
# 6. Update weights
# 7. Update target network (periodic)
```

---

## Component Interactions

### Inter-Component Communication

```
┌─────────────┐
│   train.py  │ (Orchestrator)
└─────┬───────┘
      │
      ├─→ ┌─────────────┐
      │   │   config.py │ ──→ Environment Creation
      │   └─────────────┘     Action Discovery
      │
      ├─→ ┌─────────────┐
      │   │  Agent      │
      │   └─────┬───────┘
      │         │
      │         ├─→ ┌─────────────┐
      │         │   │   Model     │ ──→ Q-network
      │         │   └─────────────┘     Target network
      │         │
      │         └─→ Memory (deque)
      │
      ├─→ ┌─────────────┐
      │   │  Reward     │ ──→ Shaping Functions
      │   │  Shaping    │     Helper Functions
      │   └─────────────┘
      │
      └─→ ┌─────────────┐
          │  Logger     │ ──→ Metrics Tracking
          └─────────────┘     Visualization
```

### Data Dependencies

```
Environment
    ↓ (observation)
Agent.select_action()
    ↓ (action)
Environment.step()
    ↓ (next_obs, reward, done, info)
Reward Shaping
    ↓ (shaped_reward)
Agent.remember()
    ↓ (stored in buffer)
Agent.learn()
    ↓ (network update)
Logger.log_episode()
    ↓ (saved to disk)
```

---

## Data Flow Diagrams

### Observation Processing Flow

```
Tetris Environment
       │
       │ observation (dict)
       ↓
┌─────────────────────┐
│  Observation Dict   │
│  - board: (20,10)   │
│  - holder: int      │
│  - queue: (5,)      │
└──────────┬──────────┘
           │
           │ extract_board_from_obs()
           ↓
┌─────────────────────┐
│   Board Array       │
│   shape: (20, 10)   │
│   values: [0, 1]    │
└──────────┬──────────┘
           │
           │ flatten (if needed)
           ↓
┌─────────────────────┐
│  Feature Vector     │
│  shape: (944,)      │
└──────────┬──────────┘
           │
           │ torch.FloatTensor
           ↓
┌─────────────────────┐
│  Network Input      │
│  shape: (1, 944)    │
│  or (1, 1, 20, 10)  │
└──────────┬──────────┘
           │
           │ forward()
           ↓
┌─────────────────────┐
│    Q-Values         │
│  shape: (1, 8)      │
└──────────┬──────────┘
           │
           │ argmax()
           ↓
       Action (int)
```

### Q-Learning Update Flow

```
Replay Buffer
     │
     │ sample_batch()
     ↓
┌──────────────────────────┐
│  Batch of Experiences    │
│  [(s, a, r, s', d), ...] │
└────────────┬─────────────┘
             │
             ├─────────────┐
             │             │
             ↓             ↓
    ┌─────────────┐  ┌─────────────┐
    │ Q-Network   │  │ Target Net  │
    │  Q(s, a)    │  │  Q_t(s')    │
    └──────┬──────┘  └──────┬──────┘
           │                │
           │                │ max()
           │                ↓
           │         ┌─────────────┐
           │         │ Q_target    │
           │         │ r + γ*max   │
           │         └──────┬──────┘
           │                │
           └────────┬───────┘
                    │
                    │ MSE Loss
                    ↓
            ┌───────────────┐
            │  Loss Value   │
            └───────┬───────┘
                    │
                    │ backward()
                    ↓
            ┌───────────────┐
            │  Gradients    │
            └───────┬───────┘
                    │
                    │ optimizer.step()
                    ↓
            ┌───────────────┐
            │ Updated Q-Net │
            └───────────────┘
```

### Episode Metrics Flow

```
Environment Step
     │
     ├─ reward
     ├─ steps
     ├─ done
     └─ info
     │
     ↓
Extract Metrics
     │
     ├─ lines_cleared
     ├─ episode_reward
     └─ episode_steps
     │
     ↓
Logger.log_episode()
     │
     ├─→ CSV File (append row)
     ├─→ Memory (store dict)
     └─→ JSON (periodic save)
     │
     ↓
Logger.plot_progress()
     │
     ├─ Read episode data
     ├─ Compute moving averages
     ├─ Generate plots
     └─ Save PNG file
```

---

## State Management

### Agent State Components

```python
Agent State = {
    # Networks
    'q_network': PyTorch Module,
    'target_network': PyTorch Module,
    
    # Optimizer
    'optimizer': Adam optimizer,
    
    # Memory
    'memory': deque of experiences,
    
    # Training Progress
    'steps_done': int,
    'episodes_done': int,
    'epsilon': float,
    
    # Hyperparameters
    'lr': float,
    'gamma': float,
    'batch_size': int,
    # ...
}
```

### Checkpoint Contents

```python
checkpoint = {
    'episode': 5000,
    'steps': 123456,
    'epsilon': 0.1234,
    
    'q_network_state': OrderedDict({
        'features.0.weight': tensor(...),
        'features.0.bias': tensor(...),
        # ... all layer parameters
    }),
    
    'target_network_state': OrderedDict({...}),
    
    'optimizer_state': {
        'state': {...},
        'param_groups': [...]
    },
    
    'hyperparameters': {
        'lr': 0.0001,
        'gamma': 0.99,
        # ...
    }
}
```

### Logger State

```python
Logger State = {
    # File paths
    'experiment_dir': str,
    'csv_file': str,
    'metrics_file': str,
    'plot_file': str,
    
    # Data
    'episode_data': [
        {
            'episode': int,
            'reward': float,
            'steps': int,
            'epsilon': float,
            'lines': int,
            'timestamp': str,
        },
        # ...
    ],
    
    # Configuration
    'config': dict
}
```

---

## Error Handling

### Common Error Scenarios

#### 1. Environment Creation Failure

```python
try:
    env = make_env()
except Exception as e:
    print(f"❌ Failed to create environment: {e}")
    print("Solutions:")
    print("  - Check tetris-gymnasium installation")
    print("  - Verify gymnasium version compatibility")
    print("  - Try: pip install --upgrade tetris-gymnasium")
    sys.exit(1)
```

#### 2. Checkpoint Loading Failure

```python
try:
    loaded = agent.load_checkpoint(latest=True)
    if not loaded:
        print("⚠️  No checkpoint found, starting fresh")
        start_episode = 0
except Exception as e:
    print(f"❌ Error loading checkpoint: {e}")
    print("Starting fresh training instead")
    start_episode = 0
```

#### 3. CUDA Out of Memory

```python
try:
    loss = agent.learn()
except RuntimeError as e:
    if "out of memory" in str(e):
        print("⚠️  GPU out of memory")
        print("Solutions:")
        print("  - Reduce batch size")
        print("  - Use CPU instead")
        print("  - Clear CUDA cache")
        torch.cuda.empty_cache()
    else:
        raise
```

#### 4. Invalid Observation Shape

```python
try:
    action = agent.select_action(obs)
except Exception as e:
    print(f"❌ Error in action selection: {e}")
    print(f"Observation type: {type(obs)}")
    print(f"Observation shape: {obs.shape if hasattr(obs, 'shape') else 'N/A'}")
    print("Check observation preprocessing")
    raise
```

### Graceful Shutdown

```python
try:
    # Training loop
    for episode in range(max_episodes):
        # ... training code ...
        pass

except KeyboardInterrupt:
    print("\n⚠️  Training interrupted by user")
    print("Saving checkpoint before exit...")
    agent.save_checkpoint(filename="interrupt.pth")
    logger.save_logs()
    print("✅ Checkpoint saved. Training can be resumed.")

except Exception as e:
    print(f"\n❌ Unexpected error: {e}")
    print("Attempting to save checkpoint...")
    try:
        agent.save_checkpoint(filename="error.pth")
        logger.save_logs()
        print("✅ Emergency checkpoint saved")
    except:
        print("❌ Failed to save checkpoint")
    raise

finally:
    # Always close environment
    env.close()
    print("Environment closed")
```

---

## Performance Optimization

### Training Speed Improvements

#### 1. GPU Utilization

```python
# Enable GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = Agent(..., device=device)

# Verify GPU usage
print(f"Using device: {device}")
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

#### 2. Disable Rendering During Training

```python
# Fast training (no visualization)
env = make_env(render_mode="rgb_array")  # or None

# Evaluation (with visualization)
eval_env = make_env(render_mode="human")
```

#### 3. Optimize Batch Size

```python
# Larger batch = faster training (if GPU available)
if torch.cuda.is_available():
    batch_size = 128  # Larger for GPU
else:
    batch_size = 32   # Smaller for CPU
```

#### 4. Efficient Memory Management

```python
# Clear old experiences periodically
if len(agent.memory) > agent.memory_size * 1.5:
    agent.memory = deque(list(agent.memory)[-agent.memory_size:], 
                         maxlen=agent.memory_size)

# Use torch.no_grad() for inference
with torch.no_grad():
    q_values = agent.q_network(state)
```

### Memory Optimization

```python
# Monitor memory usage
import psutil
process = psutil.Process()

def print_memory_usage():
    mem = process.memory_info().rss / 1024 ** 2
    print(f"Memory: {mem:.2f} MB")

# Check periodically
if episode % 100 == 0:
    print_memory_usage()
```

---

## Debugging Guide

### Debug Flags

```python
# Add debug mode to training
DEBUG = True

if DEBUG:
    # Print detailed information
    print(f"Observation: {obs}")
    print(f"Action: {action}")
    print(f"Reward: {reward}")
    print(f"Q-values: {agent.q_network(obs)}")
```

### Logging Levels

```python
# Verbose logging
VERBOSE = 2

if VERBOSE >= 1:
    # Basic info
    print(f"Episode {episode}/{max_episodes}")
    print(f"Reward: {episode_reward:.2f}")

if VERBOSE >= 2:
    # Detailed info
    print(f"Steps: {episode_steps}")
    print(f"Epsilon: {agent.epsilon:.4f}")
    print(f"Memory size: {len(agent.memory)}")

if VERBOSE >= 3:
    # Debug info
    print(f"Q-value range: [{q_min:.2f}, {q_max:.2f}]")
    print(f"Gradient norms: {grad_norms}")
```

### Diagnostic Functions

```python
def diagnose_training(agent, env, logger):
    """Run diagnostic tests"""
    
    print("\n" + "="*50)
    print("DIAGNOSTIC REPORT")
    print("="*50)
    
    # 1. Check agent
    print("\n1. Agent Status:")
    print(f"   Q-network device: {next(agent.q_network.parameters()).device}")
    print(f"   Memory size: {len(agent.memory)}/{agent.memory_size}")
    print(f"   Training steps: {agent.steps_done}")
    print(f"   Epsilon: {agent.epsilon:.4f}")
    
    # 2. Check environment
    print("\n2. Environment Status:")
    obs, _ = env.reset()
    print(f"   Observation type: {type(obs)}")
    if isinstance(obs, dict):
        for key, val in obs.items():
            print(f"   {key}: {val.shape if hasattr(val, 'shape') else type(val)}")
    
    # 3. Check logger
    print("\n3. Logger Status:")
    print(f"   Episodes logged: {len(logger.episode_data)}")
    print(f"   Experiment dir: {logger.experiment_dir}")
    
    # 4. Test forward pass
    print("\n4. Network Test:")
    try:
        with torch.no_grad():
            q_values = agent.q_network(agent._preprocess_state(obs))
        print(f"   ✅ Forward pass successful")
        print(f"   Q-values shape: {q_values.shape}")
        print(f"   Q-values range: [{q_values.min():.2f}, {q_values.max():.2f}]")
    except Exception as e:
        print(f"   ❌ Forward pass failed: {e}")
    
    print("\n" + "="*50)
```

### Performance Profiling

```python
import cProfile
import pstats

def profile_training():
    """Profile training loop to find bottlenecks"""
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run a few episodes
    for episode in range(10):
        # ... training code ...
        pass
    
    profiler.disable()
    
    # Print statistics
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions
```

---

## Summary

### Key Integration Points

1. **Config → Environment**
   - Creates and configures Tetris environment
   - Discovers action mappings dynamically
   - Provides environment to all components

2. **Config + Model → Agent**
   - Agent uses Model factory for networks
   - Observation space from Config
   - Action space from Config

3. **Agent → Memory**
   - Stores experiences in replay buffer
   - Samples for training
   - Manages capacity

4. **Agent + Model → Training**
   - Q-network for policy
   - Target network for stability
   - Optimizer for updates

5. **Reward Shaping → Learning**
   - Modifies rewards before storage
   - Guides exploration
   - Accelerates learning

6. **Logger → Monitoring**
   - Tracks all metrics
   - Creates visualizations
   - Saves checkpoints

### Critical Dependencies

```
train.py depends on:
├─ config.py (environment)
├─ agent.py (learning algorithm)
│  └─ model.py (neural networks)
├─ reward_shaping.py (reward modification)
├─ training_logger.py (metrics & logging)
└─ utils.py (helper functions)

evaluate.py depends on:
├─ config.py (environment)
├─ agent.py (trained policy)
│  └─ model.py (neural networks)
└─ utils.py (helper functions)
```

### Data Flow Summary

```
Environment
    ↓ observation
Agent
    ↓ action
Environment
    ↓ reward, next_obs, done, info
Reward Shaping
    ↓ shaped_reward
Agent Memory
    ↓ batch sample
Agent Learning
    ↓ updated networks
Repeat
```

---

*End of Integration and Workflow Documentation*
