# diagnose_model.py
"""Diagnose why the model only outputs NOOP"""

import torch
import numpy as np
from config import make_env
from src.agent import Agent


def diagnose_q_network(model_path):
    """Check what Q-values the network is producing"""
    
    print("="*80)
    print("🔍 DIAGNOSING Q-NETWORK")
    print("="*80)
    
    # Create environment and agent
    env = make_env(render_mode=None)
    agent = Agent(
        obs_space=env.observation_space,
        action_space=env.action_space,
        model_type='dqn'
    )
    
    # Load checkpoint
    agent.load_checkpoint(path=model_path)
    print(f"\n✅ Loaded checkpoint: {model_path}")
    print(f"   Episode: {agent.episodes_done}")
    print(f"   Epsilon: {agent.epsilon:.6f}")
    print(f"   Steps: {agent.steps_done}")
    
    # Get a sample observation
    obs, _ = env.reset()
    print(f"\n📊 Observation shape: {obs.shape}")
    print(f"   Observation range: [{obs.min():.3f}, {obs.max():.3f}]")
    
    # Check Q-values
    print("\n🧠 Q-NETWORK OUTPUT ANALYSIS")
    print("-"*80)
    
    agent.q_network.eval()
    with torch.no_grad():
        state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
        q_values = agent.q_network(state_tensor)
        q_values = q_values.cpu().numpy()[0]
    
    print(f"\nQ-values for all actions:")
    action_names = ['NOOP', 'LEFT', 'RIGHT', 'DOWN', 'ROTATE_CW', 'ROTATE_CCW', 'HARD_DROP', 'SWAP']
    
    for i, (name, q_val) in enumerate(zip(action_names, q_values)):
        marker = " ← SELECTED" if i == q_values.argmax() else ""
        print(f"   Action {i} ({name:11s}): Q = {q_val:8.4f}{marker}")
    
    # Statistics
    print(f"\n📈 Q-VALUE STATISTICS:")
    print(f"   Mean:    {q_values.mean():.4f}")
    print(f"   Std Dev: {q_values.std():.4f}")
    print(f"   Min:     {q_values.min():.4f}")
    print(f"   Max:     {q_values.max():.4f}")
    print(f"   Range:   {q_values.max() - q_values.min():.4f}")
    
    # Check for problems
    print("\n🔍 DIAGNOSTIC CHECKS:")
    print("-"*80)
    
    issues = []
    
    # Check 1: All Q-values near zero (not trained)
    if abs(q_values.mean()) < 0.1 and q_values.std() < 0.1:
        issues.append("❌ Q-values near zero - Network hasn't learned anything")
        issues.append("   → Need more training episodes (try 2000+)")
    
    # Check 2: Q-values all similar (no differentiation)
    if q_values.std() < 0.01:
        issues.append("❌ Q-values too similar - Network can't distinguish actions")
        issues.append("   → Check if network is getting gradients")
        issues.append("   → Verify reward shaping is providing learning signal")
    
    # Check 3: Q-values all very negative
    if q_values.max() < -10:
        issues.append("❌ All Q-values very negative - Agent expects failure")
        issues.append("   → Agent never experienced positive rewards")
        issues.append("   → Check reward shaping function")
    
    # Check 4: Only one action has high Q-value
    if q_values.argmax() == 0 and q_values[0] > q_values[1:].max() + 0.1:
        issues.append("❌ Only NOOP has high Q-value")
        issues.append("   → Agent learned to do nothing")
        issues.append("   → Check exploration is actually happening during training")
    
    # Check 5: Q-values exploded (too large)
    if abs(q_values.max()) > 1000:
        issues.append("❌ Q-values too large - Possible training instability")
        issues.append("   → Reduce learning rate")
        issues.append("   → Add gradient clipping")
    
    if issues:
        print("\n🚨 ISSUES FOUND:")
        for issue in issues:
            print(issue)
    else:
        print("✅ Q-values look reasonable")
    
    # Test on multiple states
    print("\n\n🎲 TESTING ON 10 RANDOM STATES")
    print("-"*80)
    
    action_selections = {i: 0 for i in range(len(action_names))}
    
    for i in range(10):
        obs, _ = env.reset(seed=i)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
            q_values = agent.q_network(state_tensor)
            action = q_values.argmax().item()
            action_selections[action] += 1
    
    print("\nAction selections across 10 different states:")
    for i, name in enumerate(action_names):
        count = action_selections[i]
        pct = (count / 10) * 100
        bar = "█" * count
        print(f"   {i} {name:11s}: {bar:10s} {count}/10 ({pct:.0f}%)")
    
    if action_selections[0] == 10:
        print("\n❌ ALWAYS SELECTING NOOP - This is the problem!")
        print("\n💡 POSSIBLE CAUSES:")
        print("   1. Network weights are still random/uninitialized")
        print("   2. Network learned that NOOP is safest (negative rewards for everything else)")
        print("   3. Training hasn't run long enough (only 500 episodes)")
        print("   4. Exploration not happening during training")
        print("\n💡 SOLUTIONS:")
        print("   1. Train for more episodes: python train.py --episodes 2000")
        print("   2. Check training logs for learning progress")
        print("   3. Verify exploration is using all actions during training")
        print("   4. Check reward shaping is providing positive feedback")
    
    env.close()
    
    return q_values, action_selections


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Diagnose Q-network behavior')
    parser.add_argument('--model', type=str, default='models/checkpoint_latest.pth',
                        help='Path to model checkpoint')
    args = parser.parse_args()
    
    diagnose_q_network(args.model)