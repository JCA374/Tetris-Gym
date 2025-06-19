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


----
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

## Attempt 2: Fixed Environment + Initial Training
**Date**: December 2024  
**Episodes**: 2000  
**Config**: Fixed config.py with TetrisBoardWrapper + TetrisObservationWrapper + FrameStackWrapper  
**Result**: SUCCESS - Agent learning but needs optimization  

**What we implemented**:
1. **TetrisObservationWrapper**: Extracts clean board from dict observations
2. **TetrisBoardWrapper**: Preserves 20x10 board structure in 84x84 frame with proper aspect ratio
3. **FrameStackWrapper**: Stacks 4 frames for temporal information
4. **Fixed Agent**: Handles tensor shapes correctly, no more RuntimeErrors
5. **Basic reward shaping**: lines_cleared=10.0, tetris_bonus=5.0, holes=-2.0

**Results after 2000 episodes**:
- ✅ No more tensor shape errors - agent trains successfully
- ✅ Agent learns basic survival and piece placement
- ✅ Occasional line clearing discovered
- ⚠️ Epsilon collapsed too quickly (0.995 decay → ~0.01 by episode 500)
- ⚠️ Reward shaping may be too weak for aggressive line clearing

**Key insight**: Fixed environment works perfectly, but agent needs stronger incentives and longer exploration.

---

## Attempt 3: Reward Boost + Epsilon Fix
**Date**: December 2024 (current)  
**Episodes**: 2000+ (continuing from Attempt 2)  
**Config**: Same fixed environment + boosted rewards + slower epsilon decay  
**Result**: IN PROGRESS - Applying surgical fixes  

**What we're changing**:
1. **Reward shaping boost**: 
   - lines_cleared: 10.0 → 20.0 (2x stronger incentive)
   - tetris_bonus: 5.0 → 15.0 (3x stronger Tetris reward)
2. **Epsilon decay fix**:
   - epsilon_end: 0.01 → 0.05 (maintain 5% exploration)
   - epsilon_decay: 0.995 → 0.9995 (9x longer exploration period)

**Rationale**:
- **Reward analysis**: Shaping was working correctly but too weak
- **Exploration analysis**: Agent stopped exploring after ~500 episodes at 1% random actions
- **Combined effect**: Stronger line-clear incentives + longer exploration = more aggressive play

**Expected improvements**:
- Agent should discover line clearing is 2x more valuable than survival
- 9x longer exploration period to find optimal strategies  
- More aggressive piece placement and Tetris attempts
- Higher average lines per episode within 200 episodes

**Timeline**: Should see improvement within 100-200 episodes of applying fixes