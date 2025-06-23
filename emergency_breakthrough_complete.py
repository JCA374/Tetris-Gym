#!/usr/bin/env python3
"""
emergency_breakthrough_complete.py

Emergency training using COMPLETE VISION to break the 62k plateau
This should achieve line clearing within 20-50 episodes!
"""

import numpy as np
import random
import time
import sys
import os

# CRITICAL: Import the FIXED config with complete vision
# Make sure you've created config_complete_vision.py first!
try:
    from config import make_env
    print("✅ Using complete vision config")
except ImportError:
    print("❌ ERROR: config_complete_vision.py not found!")
    print("Create it first using the artifact above!")
    sys.exit(1)

from src.agent import Agent

# Set seeds
random.seed(42)
np.random.seed(42)


def complete_vision_breakthrough():
    """Use complete vision to finally break the plateau"""
    
    print("🚀 COMPLETE VISION BREAKTHROUGH TRAINING")
    print("="*60)
    print("After 62,800 episodes of blindness, your agent can finally SEE!")
    print("Expected: Line clearing within 20-50 episodes")
    print()
    
    # Create environment with COMPLETE VISION
    env = make_env(use_complete_vision=True, use_cnn=True)
    
    print("✅ Environment created with 4-channel complete vision:")
    print("   Channel 0: Board (what you had)")
    print("   Channel 1: Active piece (WHAT WAS MISSING!)")
    print("   Channel 2: Holder")
    print("   Channel 3: Queue")
    print()
    
    # Create agent - fresh start recommended for clean learning
    agent = Agent(
        obs_space=env.observation_space,
        action_space=env.action_space,
        reward_shaping="none",  # We'll apply our own
        max_episodes=9000
    )
    
    # Try to load existing knowledge (optional)
    if agent.load_checkpoint(latest=True):
        print(f"📦 Loaded checkpoint from episode {agent.episodes_done}")
        print(f"   The agent knows survival but was blind to pieces")
        print(f"   Now it can finally see what it's placing!")
        # Boost exploration for new information
        agent.epsilon = 0.6
        agent.epsilon_decay = 0.999
        print(f"   Epsilon boosted to {agent.epsilon} for piece discovery")
    else:
        print("🆕 Starting fresh with complete vision")
        agent.epsilon = 0.8
        agent.epsilon_decay = 0.999
    
    # Enhanced reward shaping for piece-aware play
    def complete_vision_reward_shaping(obs, action, base_reward, done, info):
        shaped_reward = base_reward
        
        # Extract channels
        board_channel = obs[:, :, 0]    # Board state
        active_channel = obs[:, :, 1]   # Active piece (NEW!)
        
        # MASSIVE bonuses for line clearing now that we can see pieces
        lines = info.get('lines_cleared', 0)
        if lines > 0:
            shaped_reward += lines * 100  # Huge bonus
            if lines == 4:
                shaped_reward += 300  # Tetris jackpot
                print(f"💥 TETRIS! The agent can finally see to do this!")
        
        # Piece placement awareness bonus
        active_pixels = np.sum(active_channel > 0.01)
        if active_pixels > 0:
            # Find lowest row of active piece
            active_rows = np.any(active_channel > 0.01, axis=1)
            if np.any(active_rows):
                lowest_row = np.max(np.where(active_rows)[0])
                # Reward placing pieces low
                height_bonus = (obs.shape[0] - lowest_row) * 0.2
                shaped_reward += height_bonus
        
        # Survival bonus/penalty
        if done:
            shaped_reward -= 20
        else:
            shaped_reward += 0.1
            
        return shaped_reward
    
    # Training metrics
    lines_cleared_total = 0
    first_line_episode = None
    episodes_to_train = 9000  # Should be more than enough!
    start_episode = agent.episodes_done
    
    print(f"\n🎯 Training with complete vision...")
    print(f"Episodes {start_episode+1} to {start_episode+episodes_to_train}")
    print("="*60)
    
    for episode in range(start_episode, start_episode + episodes_to_train):
        obs, info = env.reset()
        episode_reward = 0
        episode_steps = 0
        lines_this_episode = 0
        done = False
        
        # Track what the agent sees
        piece_pixels_seen = 0
        
        while not done:
            # The agent can now see the piece it's placing!
            active_channel = obs[:, :, 1]
            piece_pixels = np.sum(active_channel > 0.01)
            piece_pixels_seen += piece_pixels
            
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Apply complete vision reward shaping
            shaped_reward = complete_vision_reward_shaping(obs, action, reward, done, info)
            
            # Store experience
            agent.remember(obs, action, shaped_reward, next_obs, done, info)
            
            # Learn frequently
            if episode_steps % 2 == 0 and len(agent.memory) >= agent.batch_size:
                agent.learn()
            
            # Track line clears
            lines = info.get('lines_cleared', 0)
            if lines > 0:
                lines_this_episode += lines
                lines_cleared_total += lines
                
                if first_line_episode is None:
                    first_line_episode = episode + 1
                    print(f"\n🎉 FIRST LINE CLEARED! Episode {first_line_episode}")
                    print(f"   Only took {first_line_episode - start_episode} episodes with vision!")
                    print(f"   (vs 62,800 episodes without seeing pieces)")
            
            episode_reward += shaped_reward
            episode_steps += 1
            obs = next_obs
        
        # Episode complete
        agent.end_episode(episode_reward, episode_steps, lines_this_episode)
        
        # Progress report
        avg_piece_pixels = piece_pixels_seen / max(1, episode_steps)
        
        if (episode + 1) % 5 == 0 or lines_this_episode > 0:
            print(f"Episode {episode+1:4d} | "
                  f"Lines: {lines_this_episode} (Total: {lines_cleared_total}) | "
                  f"Reward: {episode_reward:6.1f} | "
                  f"Steps: {episode_steps:3d} | "
                  f"Piece visibility: {avg_piece_pixels:.1f} pixels/step | "
                  f"ε: {agent.epsilon:.3f}")
        
        # Early success detection
        if lines_cleared_total >= 50:
            print(f"\n🏆 BREAKTHROUGH COMPLETE!")
            print(f"50+ lines cleared in just {episode - start_episode + 1} episodes!")
            break
            
        # Save periodically
        if (episode + 1) % 25 == 0:
            agent.save_checkpoint(episode + 1)
            print(f"   💾 Checkpoint saved")
    
    # Training complete
    env.close()
    
    # Final analysis
    episodes_trained = episode - start_episode + 1
    avg_lines = lines_cleared_total / episodes_trained if episodes_trained > 0 else 0
    
    print(f"\n" + "="*60)
    print(f"COMPLETE VISION BREAKTHROUGH RESULTS")
    print(f"="*60)
    print(f"Episodes trained: {episodes_trained}")
    print(f"Total lines cleared: {lines_cleared_total}")
    print(f"Average lines per episode: {avg_lines:.3f}")
    print(f"First line at episode: {first_line_episode or 'N/A'}")
    
    # Compare to blind training
    blind_avg = 0.03  # From 62,800 episodes
    improvement = avg_lines / blind_avg if blind_avg > 0 else float('inf')
    
    print(f"\n📊 COMPARISON:")
    print(f"   Without vision (62,800 eps): {blind_avg} lines/episode")
    print(f"   With complete vision: {avg_lines:.3f} lines/episode")
    print(f"   Improvement factor: {improvement:.0f}x")
    
    if lines_cleared_total >= 50:
        print(f"\n✅ MISSION ACCOMPLISHED!")
        print(f"The observation space fix worked perfectly!")
        print(f"\nNext steps:")
        print(f"1. Continue training: python train.py --episodes 5000 --use_complete_vision")
        print(f"2. Watch your agent master Tetris with full vision")
    elif lines_cleared_total > 0:
        print(f"\n⚠️  Breakthrough in progress!")
        print(f"Lines are being cleared - continue training!")
    else:
        print(f"\n❌ No lines cleared yet")
        print(f"Check that complete vision is working properly")
    
    # Save final checkpoint
    agent.save_checkpoint(episode + 1)
    
    return lines_cleared_total > 0


if __name__ == "__main__":
    print("Starting Complete Vision Breakthrough...")
    print("This fixes the 62,800 episode plateau by letting the agent SEE the pieces!")
    print()
    
    success = complete_vision_breakthrough()
    
    if success:
        print("\n🎉 THE VISION CRISIS IS SOLVED!")
        print("Your agent can finally see what it's doing!")
    else:
        print("\n🔧 Check that the complete vision wrapper is working")
        print("Run: python config_complete_vision.py to test")