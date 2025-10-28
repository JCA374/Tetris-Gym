# verify_training_actions.py
"""Verify what actions are being selected during actual training"""

import torch
import numpy as np
from config import make_env, ACTION_MEANINGS
from src.agent import Agent
import random


def test_exploration_distribution():
    """Test what actions the agent selects during exploration"""
    
    print("="*80)
    print("🔍 VERIFYING TRAINING EXPLORATION")
    print("="*80)
    
    # Create environment and agent
    env = make_env(render_mode=None)
    agent = Agent(
        obs_space=env.observation_space,
        action_space=env.action_space,
        epsilon_start=0.95,  # Simulate episode 500 epsilon
        model_type='dqn'
    )
    
    agent.epsilon = 0.95  # Set to match your checkpoint
    agent.episodes_done = 500  # Match your checkpoint
    
    print(f"\n⚙️  Agent Configuration:")
    print(f"   Epsilon: {agent.epsilon:.4f}")
    print(f"   Episodes done: {agent.episodes_done}")
    print(f"   Device: {agent.device}")
    
    # Get a sample observation
    obs, _ = env.reset()
    
    # Test exploration (training=True)
    print("\n\n📊 TESTING EXPLORATION (training=True)")
    print("="*80)
    print("Selecting 1000 actions with epsilon=0.95 (training mode)...")
    
    action_counts = {i: 0 for i in range(8)}
    
    for _ in range(1000):
        action = agent.select_action(obs, training=True)
        action_counts[action] += 1
    
    print("\nAction distribution during TRAINING (should be diverse):")
    print("-"*80)
    
    total = sum(action_counts.values())
    action_names = ['NOOP', 'LEFT', 'RIGHT', 'DOWN', 'ROTATE_CW', 'ROTATE_CCW', 'HARD_DROP', 'SWAP']
    
    for i in range(8):
        count = action_counts[i]
        pct = (count / total) * 100
        bar = "█" * int(pct / 2)
        name = action_names[i] if i < len(action_names) else f"ACTION_{i}"
        print(f"   {i} {name:11s}: {bar:50s} {count:4d} ({pct:5.1f}%)")
    
    # Check if exploration is working
    print("\n\n🔍 DIAGNOSIS:")
    print("="*80)
    
    issues = []
    
    # Check 1: Is NOOP over-represented?
    if action_counts[0] > 900:
        issues.append("❌ CRITICAL: Agent selecting NOOP 90%+ of the time!")
        issues.append("   → Agent.select_action() is NOT using your fixed exploration logic")
        issues.append("   → Check if you saved the updated agent.py file")
    
    # Check 2: Are all actions being tried?
    zero_actions = [i for i, count in action_counts.items() if count == 0]
    if zero_actions:
        issues.append(f"❌ Actions never selected: {zero_actions}")
        issues.append("   → Exploration logic is incomplete")
    
    # Check 3: Is distribution roughly correct?
    # With epsilon=0.95 and fixed exploration, we expect:
    # - 35% horizontal (LEFT+RIGHT) ≈ 332 each
    # - 15% rotation (CW+CCW) ≈ 71 each
    # - 5% SWAP ≈ 47
    # - 20% DOWN ≈ 190
    # - 20% HARD_DROP ≈ 190
    # - 5% NOOP ≈ 47
    # Plus 5% exploitation (random from Q-network)
    
    expected = {
        0: 47,   # NOOP (5%)
        1: 166,  # LEFT (17.5% of 95%)
        2: 166,  # RIGHT (17.5% of 95%)
        3: 190,  # DOWN (20%)
        4: 71,   # ROTATE_CW (7.5%)
        5: 71,   # ROTATE_CCW (7.5%)
        6: 190,  # HARD_DROP (20%)
        7: 47,   # SWAP (5%)
    }
    
    print("\nExpected vs Actual (with epsilon=0.95):")
    print("-"*80)
    for i in range(8):
        exp = expected[i]
        act = action_counts[i]
        diff = act - exp
        status = "✅" if abs(diff) < 100 else "⚠️"
        name = action_names[i] if i < len(action_names) else f"ACTION_{i}"
        print(f"   {status} {i} {name:11s}: Expected ~{exp:3d}, Got {act:3d} (diff: {diff:+4d})")
    
    if action_counts[0] < 100:
        print("\n✅ Exploration looks good! NOOP is appropriately low.")
    else:
        issues.append("⚠️  NOOP usage higher than expected")
    
    # Test exploitation (training=False)
    print("\n\n📊 TESTING EXPLOITATION (training=False)")
    print("="*80)
    print("Selecting 100 actions with training=False (evaluation mode)...")
    
    eval_action_counts = {i: 0 for i in range(8)}
    
    for _ in range(100):
        action = agent.select_action(obs, training=False)
        eval_action_counts[action] += 1
    
    print("\nAction distribution during EVALUATION (Q-network only):")
    print("-"*80)
    
    for i in range(8):
        count = eval_action_counts[i]
        pct = (count / 100) * 100
        bar = "█" * count
        name = action_names[i] if i < len(action_names) else f"ACTION_{i}"
        print(f"   {i} {name:11s}: {bar:50s} {count:3d} ({pct:5.1f}%)")
    
    if eval_action_counts[0] == 100:
        issues.append("❌ EVALUATION: Always selecting NOOP!")
        issues.append("   → Q-network hasn't learned - this is expected at episode 500")
        issues.append("   → All Q-values are similar, argmax picks action 0")
    
    # Print issues
    if issues:
        print("\n\n🚨 ISSUES FOUND:")
        print("="*80)
        for issue in issues:
            print(issue)
        
        print("\n\n💡 SOLUTIONS:")
        print("="*80)
        
        if action_counts[0] > 900:
            print("1. CRITICAL: Your updated agent.py is NOT being used!")
            print("   Steps to fix:")
            print("   a. Verify src/agent.py has the complete select_action method with all 8 actions")
            print("   b. Check for syntax errors: python -m py_compile src/agent.py")
            print("   c. Restart your Python kernel/terminal")
            print("   d. Run this test again")
        
        if eval_action_counts[0] == 100:
            print("2. Q-network selecting only NOOP is EXPECTED at episode 500")
            print("   This is because the network hasn't learned yet.")
            print("   Continue training to episode 2000+ to see improvement.")
    else:
        print("\n\n✅ ALL CHECKS PASSED!")
        print("="*80)
        print("Your exploration is working correctly.")
        print("The 100% NOOP in watch_agent is because the Q-network hasn't learned yet.")
        print("This is normal at episode 500.")
    
    env.close()


if __name__ == "__main__":
    test_exploration_distribution()