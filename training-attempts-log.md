# Tetris AI Training Attempts Log

## Attempt 1: Initial Training with Scrambled Observations
**Date**: December 2024  
**Episodes**: 3500  
**Config**: Original config.py with flattened observations reshaped to square  
**Result**: FAILED - No lines cleared  

**What we tried**:
- Standard DQN with default hyperparameters (LR=1e-4, epsilon_decay=100k)
- Flattened dict observations reshaped into square (sqrt method)
- Simple reward shaping with height/hole penalties

**Why it failed**:
- Board spatial structure was destroyed by reshaping 944 elements into ~31x31 square
- Model couldn't learn spatial patterns (where to place pieces)
- Agent learned to survive (~100-200 steps) but never cleared lines
- Reward plateaued around 25 with 0 lines cleared for all 3500 episodes

**Key insight**: The model literally couldn't see the Tetris board properly - the 20x10 grid was scrambled into meaningless pixels.

---

## Attempt 2: Board Wrapper Implementation (Planned)
**Date**: December 2024  
**Config**: Updated config.py with TetrisBoardWrapper + curriculum learning  

**What we're changing**:
1. **TetrisBoardWrapper**: Preserves 20x10 board structure in 84x84 frame
2. **Hyperparameters**: LR=5e-4, epsilon_end=0.05, slower decay
3. **Reward shaping**: Added LINE_CLEAR_BONUS=10, TETRIS_BONUS=50
4. **Curriculum**: Start with 10x6 board → 15x8 → 20x10
5. **Training frequency**: Every 4 steps instead of when memory full

**Expected improvements**:
- Model can now see actual board layout
- Easier to discover line clearing on small board
- Stronger incentives for core objective (clearing lines)

---

## Lessons Learned

1. **Observation preprocessing matters**: Always visualize what the model sees
2. **Check for line clears early**: If no lines cleared by episode 500, something is fundamentally wrong
3. **Start small**: Curriculum learning with smaller boards helps discover core mechanics
4. **Reward shaping needs balance**: Too much survival bonus → agent just survives; too little → agent dies quickly