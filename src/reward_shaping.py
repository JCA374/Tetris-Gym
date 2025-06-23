# balanced_reward_shaping.py
"""
Balanced reward shaping that encourages survival AND line clearing
Put this in your train.py or as a separate module
"""

import numpy as np


def balanced_tetris_reward(obs, action, base_reward, done, info):
    """
    Balanced reward shaping that doesn't create negative spiral
    
    Key principles:
    1. Small positive reward for survival (not negative!)
    2. Huge bonuses for line clearing
    3. Gentle height penalty (not harsh)
    4. Small death penalty (not overwhelming)
    """
    # Start with base game reward
    shaped_reward = base_reward

    # 1. SURVIVAL BONUS (always positive to encourage playing)
    if not done:
        shaped_reward += 1.0  # +1 per step survived

    # 2. LINE CLEARING - MASSIVE REWARDS
    lines_cleared = info.get('lines_cleared', 0)
    if lines_cleared > 0:
        # Exponential line bonuses
        line_rewards = {
            1: 40,    # Single
            2: 100,   # Double
            3: 300,   # Triple
            4: 1200   # Tetris!
        }

        bonus = line_rewards.get(lines_cleared, lines_cleared * 50)
        shaped_reward += bonus

        print(f"🎯 {lines_cleared} LINES! +{bonus} bonus!")

    # 3. HEIGHT PENALTY (gentle, not overwhelming)
    if len(obs.shape) == 3 and obs.shape[2] >= 1:
        board_channel = obs[:, :, 0]

        # Find the highest filled row
        filled_rows = np.any(board_channel > 0.01, axis=1)
        if np.any(filled_rows):
            first_filled = np.argmax(filled_rows)
            height = len(filled_rows) - first_filled

            # Only penalize if getting dangerously high
            if height > 15:  # 15 out of 24 rows
                height_penalty = (height - 15) * 0.5
                shaped_reward -= height_penalty

    # 4. DEATH PENALTY (small, not devastating)
    if done:
        shaped_reward -= 20  # Small penalty, not -100 or -500

        # But give partial credit for lasting longer
        steps = info.get('step_count', 0)
        if steps > 50:
            shaped_reward += steps * 0.1  # Partial credit for survival

    # 5. PIECE PLACEMENT BONUS (encourage active play)
    if 'piece_placed' in info:
        shaped_reward += 0.5

    return shaped_reward


def curriculum_tetris_training():
    """
    Curriculum learning approach - start easier, gradually increase difficulty
    """

    curriculum_stages = [
        {
            'episodes': (0, 100),
            'config': {
                'reward_shaping': balanced_tetris_reward,
                'survival_bonus': 2.0,  # Higher survival bonus initially
                'death_penalty': -10,   # Lower death penalty
                'line_multiplier': 2.0,  # Double line bonuses
            },
            'description': 'Stage 1: Learn to survive'
        },
        {
            'episodes': (100, 300),
            'config': {
                'reward_shaping': balanced_tetris_reward,
                'survival_bonus': 1.0,  # Normal survival
                'death_penalty': -20,   # Normal penalty
                'line_multiplier': 1.5,  # 1.5x line bonuses
            },
            'description': 'Stage 2: Discover line clearing'
        },
        {
            'episodes': (300, 1000),
            'config': {
                'reward_shaping': balanced_tetris_reward,
                'survival_bonus': 0.5,  # Lower survival bonus
                'death_penalty': -20,
                'line_multiplier': 1.0,  # Normal line bonuses
            },
            'description': 'Stage 3: Optimize line clearing'
        }
    ]

    return curriculum_stages


# Example integration with train.py:
"""
# In your training loop:

# Get current stage based on episode
current_stage = get_curriculum_stage(episode)

# Apply balanced reward shaping
shaped_reward = balanced_tetris_reward(obs, action, reward, done, info)

# Optional: Adjust based on curriculum
if current_stage:
    shaped_reward *= current_stage['config'].get('line_multiplier', 1.0)
"""

# Quick diagnostic function


def diagnose_reward_balance(episode_rewards, episode_steps, lines_cleared):
    """
    Diagnose if rewards are balanced properly
    """
    avg_reward = np.mean(episode_rewards)
    avg_steps = np.mean(episode_steps)
    total_lines = sum(lines_cleared)

    print("🔍 Reward Balance Diagnosis:")
    print(f"   Average reward: {avg_reward:.1f}")
    print(f"   Average steps: {avg_steps:.1f}")
    print(f"   Total lines: {total_lines}")

    if avg_reward < 0:
        print("   ❌ PROBLEM: Negative average rewards!")
        print("      → Death penalties too harsh")
        print("      → Need more survival bonus")

    if avg_steps < 50:
        print("   ❌ PROBLEM: Episodes too short!")
        print("      → Agent dying immediately")
        print("      → Reduce death penalty, increase survival bonus")

    if total_lines == 0 and len(episode_rewards) > 50:
        print("   ⚠️  WARNING: No lines cleared after 50 episodes")
        print("      → May need higher line bonuses")
        print("      → Check if epsilon is high enough")

    if avg_reward > 0 and avg_steps > 100:
        print("   ✅ Rewards appear balanced")
        print("      → Agent surviving and exploring")
