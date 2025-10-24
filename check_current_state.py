#!/usr/bin/env python3
# check_current_state.py
"""
Quick script to check your agent's current state without editing train.py
Run this to see epsilon, learning rate, and Q-values at your current checkpoint
"""

import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from config import make_env
    from src.agent import Agent
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

def check_checkpoint():
    """Check the current state of your trained agent"""
    
    print("="*60)
    print("🔬 AGENT STATE DIAGNOSTIC")
    print("="*60)
    
    # Load latest checkpoint
    checkpoint_path = 'models/checkpoint_latest.pth'
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ No checkpoint found at {checkpoint_path}")
        print("Have you trained the model yet?")
        return
    
    print(f"\n📁 Loading checkpoint: {checkpoint_path}\n")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Print checkpoint info
        print(f"📊 CHECKPOINT INFO:")
        print(f"   Episode: {checkpoint.get('episode', 'Unknown')}")
        print(f"   Epsilon: {checkpoint.get('epsilon', 'Unknown'):.4f}")
        
        # Check if optimizer state is available
        if 'optimizer_state_dict' in checkpoint:
            # Get learning rate from optimizer
            try:
                lr = checkpoint['optimizer_state_dict']['param_groups'][0]['lr']
                print(f"   Learning rate: {lr:.6f}")
            except:
                print(f"   Learning rate: Unable to extract")
        
        print(f"   Reward shaping: {checkpoint.get('reward_shaping_type', 'Unknown')}")
        
        # Memory info
        if 'memory_size' in checkpoint:
            print(f"   Replay buffer size: {checkpoint['memory_size']}")
        
        print()
        
        # Create environment and agent
        print("🏗️  Recreating agent...")
        env = make_env(render_mode="rgb_array")
        
        agent = Agent(
            obs_space=env.observation_space,
            action_space=env.action_space,
            reward_shaping="aggressive"  # Match your training
        )
        
        # Load the checkpoint
        agent.load_checkpoint(latest=True, model_dir='models')
        
        print(f"✅ Agent loaded successfully")
        print(f"   Current epsilon: {agent.epsilon:.4f}")
        print()
        
        # Test Q-values
        print("🧠 TESTING Q-VALUES:")
        print("   Resetting environment and sampling Q-values...")
        
        obs, _ = env.reset(seed=42)
        
        with torch.no_grad():
            state = agent._preprocess_state(obs)
            q_vals = agent.q_network(state).cpu().numpy().flatten()
        
        print(f"\n   Q-value statistics:")
        print(f"      Min:    {q_vals.min():8.2f}")
        print(f"      Max:    {q_vals.max():8.2f}")
        print(f"      Mean:   {q_vals.mean():8.2f}")
        print(f"      Std:    {q_vals.std():8.2f}")
        print(f"      Range:  {q_vals.max() - q_vals.min():8.2f}")
        
        # Show all Q-values
        action_names = ['NOOP', 'RIGHT', 'LEFT', 'DOWN', 'ROT_CW', 'ROT_CCW', 'DROP', 'HOLD']
        print(f"\n   Q-values by action:")
        for i, (name, q) in enumerate(zip(action_names, q_vals)):
            print(f"      {i}. {name:8s}: {q:8.2f}")
        
        env.close()
        
        print()
        print("="*60)
        print("🎯 DIAGNOSIS")
        print("="*60)
        
        # Diagnose based on values
        epsilon = agent.epsilon
        q_std = q_vals.std()
        q_mean = q_vals.mean()
        
        issues = []
        recommendations = []
        
        # Check epsilon
        if epsilon > 0.15:
            issues.append("Epsilon too high (still exploring randomly)")
            recommendations.append("Force epsilon to 0.05 and continue training")
        elif epsilon < 0.02:
            issues.append("Epsilon too low (no exploration)")
            recommendations.append("Boost epsilon to 0.1 for more exploration")
        else:
            print("✅ Epsilon looks good (0.02-0.15 range)")
        
        # Check Q-values
        if q_std < 1.0:
            issues.append("Q-values collapsed (all identical)")
            recommendations.append("Model hasn't learned - restart with better rewards")
        elif q_std > 200:
            issues.append("Q-values unstable (huge variance)")
            recommendations.append("Learning rate too high - try LR=0.0001")
        elif abs(q_mean) < 1.0 and q_std < 5.0:
            issues.append("Q-values near zero (minimal learning)")
            recommendations.append("Reward function may be too weak")
        else:
            print("✅ Q-values show learning (varied and non-zero)")
        
        if not issues:
            print("✅ All metrics look healthy!")
            print("\n🤔 Since observations are correct and metrics look good,")
            print("   the issue is likely:")
            print("   1. Need MORE training (try 25k-50k episodes)")
            print("   2. Reward function doesn't incentivize line clearing enough")
            print("   3. Training got stuck in local minimum")
        else:
            print("\n❌ ISSUES FOUND:")
            for i, issue in enumerate(issues, 1):
                print(f"   {i}. {issue}")
            
            print("\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        
        print("\n" + "="*60)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_checkpoint()
