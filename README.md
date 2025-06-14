# Tetris AI with Deep Reinforcement Learning

A modular Deep Q-Network (DQN) implementation for training AI agents to play Tetris using the modern **Tetris Gymnasium** environment.

## 🎯 Features

- **Modern Environment**: Uses [Tetris Gymnasium](https://github.com/Max-We/Tetris-Gymnasium) - the most up-to-date Tetris RL environment
- **Flexible Architecture**: Supports both standard DQN and Dueling DQN
- **Comprehensive Logging**: Detailed training metrics, plots, and TensorBoard support
- **Easy Evaluation**: Built-in evaluation scripts with video recording
- **Modular Design**: Clean, extensible codebase for research and experimentation

## 🚀 Quick Start

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd tetris_ai
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Test the setup**
   ```bash
   python test_setup.py
   ```

### Training

```bash
# Basic training
python train.py

# Advanced training with custom parameters
python train.py --episodes 1000 --lr 0.0001 --model_type dueling_dqn --experiment_name "my_experiment"
```

### Evaluation

```bash
# Evaluate trained model
python evaluate.py --episodes 10 --render

# Detailed evaluation with video recording
python evaluate.py --episodes 20 --render --save_video --detailed
```

## 📁 Project Structure

```
tetris_ai/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
├── config.py                # Environment configuration & hyperparameters
├── train.py                 # Training script
├── evaluate.py              # Evaluation script
├── test_setup.py           # Comprehensive test script
│
├── src/                     # Core library code
│   ├── __init__.py
│   ├── model.py            # Neural network architectures (DQN, Dueling DQN)
│   ├── agent.py            # RL agent implementation
│   ├── env.py              # Environment wrapper (simplified)
│   └── utils.py            # Utilities (logging, plotting, benchmarking)
│
├── models/                  # Saved model checkpoints (created during training)
├── logs/                   # Training logs and plots (created during training)
└── HW.txt                  # Hardware specifications
```

## 🔧 Configuration

Key settings in `config.py`:

```python
# Environment
ENV_NAME = "tetris_gymnasium/Tetris"

# Training hyperparameters
LR = 1e-4                    # Learning rate
GAMMA = 0.99                 # Discount factor
BATCH_SIZE = 32              # Batch size
MAX_EPISODES = 500           # Default training episodes

# Directories
MODEL_DIR = "models/"        # Model checkpoints
LOG_DIR = "logs/"           # Training logs
```

## 🎮 Environment Details

This project uses **Tetris Gymnasium**, which offers:

- **Modern API**: Built on Gymnasium (successor to OpenAI Gym)
- **Customizable**: Adjustable board size, gravity, reward functions
- **Feature-rich**: Comprehensive game statistics and info
- **Performance**: Both standard and JAX-based implementations
- **Documentation**: Excellent docs and examples

### Environment Configuration

You can customize the Tetris environment:

```python
# Different board sizes
env = make_env("tetris_gymnasium/Tetris", 
               board_height=15, board_width=8)

# Different preprocessing
env = make_env("tetris_gymnasium/Tetris", 
               preprocess=True, frame_stack=4)
```

## 🧠 Model Architectures

### Standard DQN
- Convolutional layers for image processing
- Fully connected layers for decision making
- Experience replay and target networks

### Dueling DQN
- Separate value and advantage streams
- Better value estimation for states
- Improved performance on Tetris

## 📊 Training Features

### Automatic Logging
- Episode rewards and statistics
- Training plots and progress visualization
- Model checkpoints and best model saving
- CSV logs for detailed analysis

### Monitoring
- Real-time training progress
- Periodic evaluation during training
- Performance metrics and benchmarking
- Optional TensorBoard integration

### Resumable Training
```bash
# Resume from latest checkpoint
python train.py --resume

# Custom model path
python evaluate.py --model_path models/best_model.pth
```

## 🎯 Training Tips

### Quick Testing (Fast feedback)
```bash
python train.py --episodes 100 --log_freq 5 --eval_freq 20
```

### Production Training (Best results)
```bash
python train.py --episodes 2000 --model_type dueling_dqn --lr 0.0001
```

### Custom Environment (Easier learning)
Modify `config.py` to use smaller board:
```python
def make_env(...):
    env = gym.make("tetris_gymnasium/Tetris", 
                   board_height=10, board_width=6)  # Smaller board
```

## 📈 Results and Analysis

Training outputs are saved to:
- `logs/<experiment_name>/` - Training logs and plots
- `models/` - Model checkpoints
- `models/evaluation_results/` - Evaluation summaries

View training progress:
```bash
# Generate plots from logs
python -c "from src.utils import TrainingLogger; logger = TrainingLogger('logs', 'your_experiment'); logger.plot_progress()"
```

## 🔍 Troubleshooting

### Common Issues

1. **Import errors**: Ensure all dependencies are installed
   ```bash
   pip install tetris-gymnasium gymnasium torch numpy
   ```

2. **CUDA issues**: Check PyTorch CUDA compatibility
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

3. **Environment errors**: Run the test script
   ```bash
   python test_setup.py
   ```

### Performance Optimization

- Use `render_mode=None` for training (faster)
- Reduce board size for quicker convergence
- Use GPU if available (automatic detection)
- Adjust `batch_size` based on your hardware

## 🔬 Research and Extensions

This codebase is designed for research and experimentation:

### Easy Modifications
- **New reward functions**: Modify the environment wrapper
- **Different architectures**: Add models to `src/model.py`
- **Alternative algorithms**: Extend `src/agent.py`
- **Custom environments**: Use different Tetris configurations

### Advanced Features
- **Multi-agent training**: Train multiple agents simultaneously
- **Hyperparameter tuning**: Use the modular config system
- **Transfer learning**: Load pre-trained models
- **Custom metrics**: Add domain-specific evaluation metrics

## 📚 References

- [Tetris Gymnasium Paper](https://easychair.org/publications/preprint/5sXq): "Piece by Piece: Assembling a Modular Reinforcement Learning Environment for Tetris"
- [Tetris Gymnasium GitHub](https://github.com/Max-We/Tetris-Gymnasium)
- [DQN Paper](https://arxiv.org/abs/1312.5602): "Playing Atari with Deep Reinforcement Learning"
- [Dueling DQN Paper](https://arxiv.org/abs/1511.06581): "Dueling Network Architectures for Deep Reinforcement Learning"

## 📄 License

This project is licensed under the MIT License. See the original Tetris Gymnasium repository for additional licensing information.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- New model architectures
- Better reward shaping
- Performance optimizations
- Additional evaluation metrics
- Documentation improvements

## 🎖️ Acknowledgments

- **Tetris Gymnasium**: Modern, modular Tetris environment
- **Gymnasium**: Standard RL environment API
- **PyTorch**: Deep learning framework
- **OpenAI**: Original Gym framework inspiration

---

**Happy training! 🎮🤖**