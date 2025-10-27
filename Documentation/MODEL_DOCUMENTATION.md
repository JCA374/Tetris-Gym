# MODEL_DOCUMENTATION.md

# Neural Network Models - Detailed Documentation

## Overview

The Model module (`src/model.py`) provides neural network architectures for Q-value approximation in the DQN agent. It supports multiple architectures optimized for different input formats.

---

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Input Processing](#input-processing)
3. [Model Types](#model-types)
4. [Factory Pattern](#factory-pattern)
5. [Technical Details](#technical-details)
6. [Usage Examples](#usage-examples)
7. [Customization Guide](#customization-guide)

---

## Architecture Overview

### Purpose
Transform game observations into Q-value estimates for each possible action.

### Input/Output Flow
```
Observation
    ↓
Preprocessing
    ↓
Neural Network
    ↓
Q-Values (one per action)
```

### Supported Formats
1. **Spatial (2D/3D)**: Convolutional networks for board representations
2. **Vector (1D)**: Fully connected networks for flattened features

---

## Input Processing

### Shape Inference System

#### Function: `_infer_input_shape(obs_space)`

**Purpose**: Automatically detect and standardize input dimensions.

**Logic**:
```python
def _infer_input_shape(obs_space):
    """
    Convert various observation formats to (C, H, W).
    
    Common formats:
    - (H, W, C): Height-Width-Channels (OpenCV/Gym standard)
    - (C, H, W): Channels-Height-Width (PyTorch standard)
    - (H, W): Single channel (grayscale)
    - (features,): Flattened vector
    
    Returns: (C, H, W) tuple
    """
```

**Conversion Rules**:

1. **3D Input** → Detect channel position
```python
if len(shape) == 3:
    H, W, C = shape
    if C <= 16:  # Last dimension is channels
        return (C, H, W)
    else:  # First dimension is channels
        C, H, W = shape
        return (C, H, W)
```

2. **2D Input** → Add channel dimension
```python
if len(shape) == 2:
    H, W = shape
    return (1, H, W)  # Single channel
```

3. **1D Input** → Reshape for processing
```python
if len(shape) == 1:
    flat = int(np.prod(shape))
    return (1, flat, 1)  # Treat as 1D feature map
```

4. **Fallback** → Default Tetris dimensions
```python
return (1, 20, 10)  # Standard Tetris board
```

### Tensor Format Conversion

#### Function: `_to_nchw(x)`

**Purpose**: Ensure tensors match PyTorch's NCHW format.

```python
def _to_nchw(x):
    """
    Convert tensor to (Batch, Channels, Height, Width) format.
    
    Handles:
    - (B, H, W, C) → (B, C, H, W)
    - (B, H, W) → (B, 1, H, W)
    - Already correct → pass through
    
    Returns: Properly formatted tensor
    """
```

**Conversion Logic**:
```python
# Get shape
if len(x.shape) == 4:
    B, dim1, dim2, dim3 = x.shape
    
    # Check if already NCHW
    if dim1 <= 16:  # Channels first
        return x
    
    # Convert BHWC → BCHW
    return x.permute(0, 3, 1, 2)

elif len(x.shape) == 3:
    # Add channel dimension
    return x.unsqueeze(1)  # (B, H, W) → (B, 1, H, W)

else:
    # Pass through
    return x
```

---

## Model Types

### 1. ConvDQN - Convolutional Deep Q-Network

**Best For**: Spatial board representations (Tetris, chess, images)

#### Architecture

```
Input: (Batch, Channels, Height, Width)
Example: (32, 1, 20, 10) for Tetris

Features Block:
│
├─ Conv2d(in=C, out=32, kernel=3×3, padding=1)
│  │ Purpose: Extract low-level patterns
│  │ Output: (B, 32, H, W)
│  └─ ReLU activation
│
├─ Conv2d(in=32, out=64, kernel=3×3, padding=1)
│  │ Purpose: Build higher-level features
│  │ Output: (B, 64, H, W)
│  └─ ReLU activation
│
├─ AdaptiveAvgPool2d(output_size=(20, 10))
│  │ Purpose: Normalize spatial dimensions
│  │ Output: (B, 64, 20, 10)
│  │ Why: Handles varying input sizes
│
└─ Flatten()
   │ Output: (B, 12800)
   │ Calculation: 64 * 20 * 10 = 12800

Decision Block:
│
├─ Linear(in=12800, out=256)
│  └─ ReLU activation
│
└─ Linear(in=256, out=n_actions)
   │ Output: (B, n_actions)
   └─ Q-values for each action
```

#### Implementation

```python
class ConvDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        C, H, W = input_shape
        
        # Feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((20, 10)),  # Normalize size
            nn.Flatten(),
        )
        
        # Fixed size after pooling
        flat_features = 64 * 20 * 10  # 12,800
        
        # Decision head
        self.head = nn.Sequential(
            nn.Linear(flat_features, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )
    
    def forward(self, x):
        # Ensure proper format
        x = _to_nchw(x)
        
        # Extract features
        features = self.features(x)
        
        # Compute Q-values
        q_values = self.head(features)
        
        return q_values
```

#### Layer Details

**Convolutional Layers**:
- **Kernel Size**: 3×3 (captures local patterns)
- **Padding**: 1 (preserves spatial dimensions)
- **Channels**: 32 → 64 (increasing capacity)

**Why These Choices?**
```
3×3 kernel:
┌─┬─┬─┐
│ │ │ │  Detects patterns like:
├─┼─┼─┤  - Edges (filled vs empty)
│ │ │ │  - Holes (empty surrounded)
└─┴─┴─┘  - Shapes (piece outlines)

32 → 64 channels:
Layer 1: Basic patterns (lines, edges)
Layer 2: Complex patterns (shapes, holes, gaps)
```

**Adaptive Average Pooling**:
```python
AdaptiveAvgPool2d((20, 10))
```
- **Purpose**: Handle varying input sizes
- **Output**: Always 20×10 regardless of input
- **How**: Adjusts pooling window size automatically
- **Benefit**: Same network works for different boards

**Flattening**:
```python
Input:  (Batch, 64, 20, 10)
Flatten: 64 × 20 × 10 = 12,800 features
Output: (Batch, 12800)
```

**Fully Connected Layers**:
```
12800 → 256 → n_actions

Purpose:
- Aggregate spatial information
- Learn action-value mapping
- Compress to decision space
```

### 2. MLPDQN - Multi-Layer Perceptron Deep Q-Network

**Best For**: Pre-flattened features, tabular data, simple observations

#### Architecture

```
Input: (Batch, Features)
Example: (32, 944) for flattened Tetris obs

│
├─ Flatten() (ensure 2D)
│  │ Output: (B, in_dim)
│
├─ Linear(in=in_dim, out=512)
│  │ Purpose: First hidden layer
│  │ Output: (B, 512)
│  └─ ReLU activation
│
├─ Linear(in=512, out=256)
│  │ Purpose: Second hidden layer
│  │ Output: (B, 256)
│  └─ ReLU activation
│
└─ Linear(in=256, out=n_actions)
   │ Purpose: Output layer
   │ Output: (B, n_actions)
   └─ Q-values for each action
```

#### Implementation

```python
class MLPDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        C, H, W = input_shape
        in_dim = C * H * W  # Total features
        
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )
    
    def forward(self, x):
        return self.net(x)
```

#### When to Use?

**Advantages**:
- Simple and fast
- Works with any input shape
- Easy to debug
- Lower memory usage

**Disadvantages**:
- No spatial awareness
- Larger parameter count for images
- May miss local patterns

**Use Cases**:
- Flattened observations (already vector)
- Small state spaces
- Tabular features
- When Conv performance is poor

### Parameter Counts

**ConvDQN**:
```
Conv1: 3×3×1×32 + 32 = 320
Conv2: 3×3×32×64 + 64 = 18,496
FC1: 12800×256 + 256 = 3,277,056
FC2: 256×n_actions + n_actions

Total (8 actions): ~3,295,928 parameters
```

**MLPDQN** (944 input features):
```
FC1: 944×512 + 512 = 483,840
FC2: 512×256 + 256 = 131,328
FC3: 256×8 + 8 = 2,056

Total: ~617,224 parameters
```

**Analysis**:
- ConvDQN: More parameters, better spatial understanding
- MLPDQN: Fewer parameters, faster training, less expressive

---

## Factory Pattern

### Function: `create_model()`

**Purpose**: Unified interface for creating different model types.

```python
def create_model(obs_space, action_space, model_type="dqn"):
    """
    Factory function for creating Q-networks.
    
    Args:
        obs_space: Observation space (gym.spaces.Box) or shape tuple
        action_space: Action space (gym.spaces.Discrete) or int
        model_type: Architecture type
            - "dqn" or "conv": ConvDQN
            - "mlp" or "dense": MLPDQN
            - "dueling_dqn": Dueling DQN (future)
    
    Returns:
        model: PyTorch nn.Module
    """
```

### Implementation

```python
def create_model(obs_space, action_space, model_type="dqn"):
    # 1. Extract number of actions
    def _n_actions_from(action_space):
        if hasattr(action_space, "n"):
            return int(action_space.n)
        return int(action_space)
    
    n_actions = _n_actions_from(action_space)
    
    # 2. Infer input shape
    C, H, W = _infer_input_shape(obs_space)
    
    # 3. Create model based on type
    if model_type in ("mlp", "dense"):
        return MLPDQN((C, H, W), n_actions)
    else:  # Default to Conv
        return ConvDQN((C, H, W), n_actions)
```

### Usage Examples

```python
# From Gym environment
env = gym.make("TetrisGymnasium/Tetris")
model = create_model(env.observation_space, env.action_space)

# From manual specification
obs_shape = (1, 20, 10)  # Channels, Height, Width
n_actions = 8
model = create_model(obs_shape, n_actions, model_type="dqn")

# MLP for flattened input
flat_obs = (944,)
model = create_model(flat_obs, 8, model_type="mlp")
```

---

## Technical Details

### Activation Functions

#### ReLU (Rectified Linear Unit)
```python
ReLU(x) = max(0, x)
```

**Why ReLU?**
- Fast computation
- Helps with gradient flow
- Prevents vanishing gradients
- Introduces non-linearity

**Properties**:
```
Input:  [-2, -1, 0, 1, 2]
Output: [0, 0, 0, 1, 2]

Gradient:
  x > 0: gradient = 1
  x ≤ 0: gradient = 0
```

### Weight Initialization

**PyTorch Default**: Kaiming Uniform
```python
# Automatically applied
nn.Linear → kaiming_uniform_
nn.Conv2d → kaiming_uniform_
```

**Why Kaiming?**
- Designed for ReLU activations
- Prevents gradient explosion/vanishing
- Better than Xavier for ReLU

**Manual Initialization** (if needed):
```python
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
```

### Batch Normalization (Optional)

**Not Currently Used** - Can add if needed:

```python
class ConvDQNWithBN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        C, H, W = input_shape
        
        self.features = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),  # Normalize activations
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # ...
        )
```

**Trade-offs**:
- **Pros**: Faster training, better generalization
- **Cons**: Slower inference, more parameters, complexity

### Dropout (Optional)

**Not Currently Used** - Can add for regularization:

```python
self.head = nn.Sequential(
    nn.Linear(12800, 256),
    nn.ReLU(),
    nn.Dropout(0.5),  # Randomly drop 50% of neurons
    nn.Linear(256, n_actions),
)
```

**When to Use**:
- Overfitting on training data
- Large networks
- Limited training data

**Trade-offs**:
- **Pros**: Reduces overfitting
- **Cons**: Slower training, requires tuning

---

## Model Comparison

### ConvDQN vs MLPDQN

| Aspect | ConvDQN | MLPDQN |
|--------|---------|---------|
| **Best For** | Spatial data (boards, images) | Flattened features, vectors |
| **Parameters** | ~3.3M (Tetris) | ~617K (Tetris) |
| **Speed** | Slower (convolutions) | Faster (matrix mult only) |
| **Spatial Awareness** | Yes (local patterns) | No (global features only) |
| **Memory** | Higher | Lower |
| **Interpretability** | Filters can be visualized | Harder to interpret |
| **Tetris Performance** | Usually better | Can work if features good |

### Choosing the Right Model

**Use ConvDQN when**:
- Input is 2D (board, grid, image)
- Spatial relationships matter
- Local patterns are important
- Have sufficient compute resources

**Use MLPDQN when**:
- Input is pre-processed features
- Already flattened observations
- Limited compute resources
- Quick prototyping
- Conv performance is poor

---

## Usage Examples

### Creating a Model

```python
from src.model import create_model
import gymnasium as gym

# Method 1: From environment
env = gym.make("TetrisGymnasium/Tetris")
model = create_model(env.observation_space, env.action_space)

# Method 2: Manual specification
model = create_model(
    obs_space=(1, 20, 10),  # (C, H, W)
    action_space=8,
    model_type="dqn"
)

# Method 3: MLP
model = create_model(
    obs_space=(944,),  # Flattened features
    action_space=8,
    model_type="mlp"
)
```

### Forward Pass

```python
import torch

# Single observation
obs = torch.randn(1, 1, 20, 10)  # (Batch, C, H, W)
q_values = model(obs)
print(q_values.shape)  # (1, 8)

# Batch of observations
obs_batch = torch.randn(32, 1, 20, 10)
q_values_batch = model(obs_batch)
print(q_values_batch.shape)  # (32, 8)

# Select best action
best_action = q_values.argmax(dim=1)
print(best_action)  # tensor([3])
```

### Model Information

```python
# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Model structure
print(model)

# Layer details
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")
```

### Save/Load Model

```python
# Save
torch.save(model.state_dict(), 'model.pth')

# Load
model = create_model(obs_space, action_space)
model.load_state_dict(torch.load('model.pth'))
model.eval()  # Set to evaluation mode
```

---

## Customization Guide

### Adding a New Architecture

**Step 1**: Create model class

```python
class ResNetDQN(nn.Module):
    """DQN with Residual connections"""
    
    def __init__(self, input_shape, n_actions):
        super().__init__()
        C, H, W = input_shape
        
        # Your architecture here
        self.conv1 = nn.Conv2d(C, 64, 3, padding=1)
        # ... residual blocks ...
        self.fc = nn.Linear(flat_size, n_actions)
    
    def forward(self, x):
        # Your forward pass
        return q_values
```

**Step 2**: Add to factory

```python
def create_model(obs_space, action_space, model_type="dqn"):
    # ...
    if model_type == "resnet":
        return ResNetDQN((C, H, W), n_actions)
    # ...
```

**Step 3**: Use in training

```python
python train.py --model_type resnet
```

### Modifying Existing Architectures

#### Change Network Depth

```python
class DeepConvDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        C, H, W = input_shape
        
        self.features = nn.Sequential(
            nn.Conv2d(C, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),  # Added layer
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), # Added layer
            nn.AdaptiveAvgPool2d((20, 10)),
            nn.Flatten(),
        )
        
        self.head = nn.Sequential(
            nn.Linear(128 * 20 * 10, 512),  # More capacity
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )
```

#### Add Attention Mechanism

```python
class AttentionDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        # ... conv layers ...
        
        self.attention = nn.MultiheadAttention(
            embed_dim=64,
            num_heads=4
        )
        
        # ... rest of network ...
    
    def forward(self, x):
        # Extract features
        features = self.conv_features(x)
        
        # Apply attention
        attended_features, _ = self.attention(features, features, features)
        
        # Decision head
        q_values = self.head(attended_features)
        
        return q_values
```

---

## Advanced Topics

### Dueling DQN Architecture

**Concept**: Separate value and advantage streams

```python
class DuelingDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        C, H, W = input_shape
        
        # Shared feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(C, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((20, 10)),
            nn.Flatten(),
        )
        
        flat_size = 64 * 20 * 10
        
        # Value stream: V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(flat_size, 256), nn.ReLU(),
            nn.Linear(256, 1)  # Scalar value
        )
        
        # Advantage stream: A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(flat_size, 256), nn.ReLU(),
            nn.Linear(256, n_actions)  # Per-action advantage
        )
    
    def forward(self, x):
        # Shared features
        features = self.features(x)
        
        # Separate streams
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values
```

**Benefits**:
- Better learning efficiency
- More stable training
- Improved generalization

### Noisy Networks

**Concept**: Learnable noise for exploration

```python
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        
        self.reset_parameters()
    
    def forward(self, x):
        if self.training:
            # Add noise during training
            weight = self.weight_mu + self.weight_sigma * torch.randn_like(self.weight_sigma)
            bias = self.bias_mu + self.bias_sigma * torch.randn_like(self.bias_sigma)
        else:
            # No noise during evaluation
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)
```

---

## Performance Optimization

### Memory Efficiency

```python
# Use half precision (if GPU supports)
model = model.half()  # FP16 instead of FP32

# Gradient checkpointing (save memory)
from torch.utils.checkpoint import checkpoint
features = checkpoint(self.features, x)

# In-place operations
nn.ReLU(inplace=True)
```

### Speed Optimization

```python
# Compile model (PyTorch 2.0+)
model = torch.compile(model)

# Use channels_last memory format
model = model.to(memory_format=torch.channels_last)
x = x.to(memory_format=torch.channels_last)

# Mixed precision training
from torch.cuda.amp import autocast
with autocast():
    q_values = model(x)
```

---

## Testing

```python
def test_conv_dqn():
    model = ConvDQN((1, 20, 10), 8)
    x = torch.randn(16, 1, 20, 10)
    y = model(x)
    assert y.shape == (16, 8)

def test_mlp_dqn():
    model = MLPDQN((1, 20, 10), 8)
    x = torch.randn(16, 1, 20, 10)
    y = model(x)
    assert y.shape == (16, 8)

def test_create_model():
    model = create_model((1, 20, 10), 8, "dqn")
    assert isinstance(model, ConvDQN)
    
    model = create_model((944,), 8, "mlp")
    assert isinstance(model, MLPDQN)
```

---

## Summary

The Model module provides flexible, efficient neural network architectures for Q-value approximation:

**Key Features**:
- Automatic input shape handling
- Multiple architecture types
- Factory pattern for easy creation
- Optimized for Tetris and similar games

**Design Principles**:
- Flexibility: Support various input formats
- Simplicity: Easy to understand and modify
- Efficiency: Optimized implementations
- Extensibility: Easy to add new architectures

**Integration**:
- Used by Agent for policy and target networks
- Receives observations from environment
- Outputs Q-values for action selection
- Trained via backpropagation in Agent.learn()

---

*End of Model Documentation*
