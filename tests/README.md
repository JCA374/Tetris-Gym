# Tetris-Gym Tests

This folder contains test and diagnostic scripts for the Tetris RL project.

## Running Tests

All test files can now be run directly from the `tests/` folder:

```bash
cd tests/
python test_actions_simple.py
python test_environment_rendering.py
python verify_training_actions.py
python diagnose_training.py
python diagnose_model.py
python test_reward_helpers.py
```

Or from the project root:

```bash
python tests/test_actions_simple.py
```

## Test Files

### Action Testing
- **test_actions_simple.py** - Tests that LEFT/RIGHT/HARD_DROP actions work correctly
  - Verifies horizontal movement
  - Checks piece placement across different columns
  - Validates action mappings

### Environment Testing
- **test_environment_rendering.py** - Tests environment rendering and action behavior
  - Tests pygame rendering
  - Validates action effects on the board
  - Checks column distribution

### Agent/Training Diagnostics
- **verify_training_actions.py** - Verifies agent's action selection during training
  - Tests exploration distribution
  - Checks epsilon-greedy behavior
  - Validates Q-network action selection

- **diagnose_training.py** - Comprehensive training diagnostic tool
  - Analyzes checkpoints and logs
  - Identifies training issues
  - Provides recommendations

- **diagnose_model.py** - Diagnoses Q-network behavior
  - Analyzes Q-values for all actions
  - Checks for common issues (all NOOP, collapsed values, etc.)
  - Tests network across multiple states

### Reward Function Testing
- **test_reward_helpers.py** - Tests reward shaping functions
  - Validates helper functions (holes, bumpiness, heights)
  - Tests reward calculations
  - Verifies reward shaping logic

## Verification

Run the import verification script to check all dependencies:

```bash
cd tests/
python verify_imports.py
```

This will check if all required modules can be imported correctly.

## Dependencies

Make sure you have all required packages installed:

```bash
pip install numpy torch gymnasium tetris-gymnasium
```

## Notes

- All test files have been updated to work from the `tests/` folder
- They automatically add the parent directory to Python's path
- Tests that require trained models will need checkpoint files in `../models/`
- Diagnostic tools will search for logs in `../logs/`
