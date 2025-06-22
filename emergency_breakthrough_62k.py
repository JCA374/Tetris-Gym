#!/usr/bin/env python3
"""
emergency_breakthrough_62k.py

EMERGENCY INTERVENTION for 62,800+ episode plateau
Specifically designed to break the "survival-only" local optimum
"""

import numpy as np
import random
import time
from config import make_env  # Use the fixed config
from src.agent import Agent
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


def emergency_breakthrough_training():
    """Emergency training designed specifically for 62k episode plateau"""
    
    print("🚨 EMERGENCY BREAKTHROUGH TRAINING")
    print("="*60)
    print("Designed for agents stuck after 60,000+ episodes")
    print("Target: Break survival-only local optimum")
    print("Strategy: MASSIVE line clear incentives + forced exploration")
    print()
    
    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Create environment
    env = make_env(use_cnn=True, frame_stack=1)
    
    # Create agent and load existing checkpoint
    agent = Agent(
        obs_space=env.observation_space,
        action_space=env.action_space,
        reward_shaping="simple",
        max_episodes=75000  # Match your current setup
    )
    
    # Load the existing 62k checkpoint
    if agent.load_checkpoint(latest=True):
        start_episode = agent.episodes_done
        print(f"✅ Loaded checkpoint from episode {start_episode}")
        print(f"Current epsilon: {agent.epsilon:.6f}")
        print(f"Total rewards history: {len(agent.total_rewards)} episodes")
    else:
        print("❌ No checkpoint found!")
        print("Make sure models/latest_checkpoint.pth exists")
        return
    
    # 🚨 EMERGENCY INTERVENTIONS
    print(f"\n🔥 APPLYING EMERGENCY INTERVENTIONS:")
    
    # 1. MASSIVE EPSILON BOOST (override adaptive schedule)
    original_epsilon = agent.epsilon
    agent.epsilon = 0.4  # Huge boost for stuck agent
    print(f"   Epsilon: {original_epsilon:.6f} → {agent.epsilon:.6f}")
    
    # 2. OVERRIDE REWARD SHAPING (we'll apply our own)
    agent.reward_shaping_type = "none"  # Disable agent's internal shaping
    print(f"   Disabled internal reward shaping - applying emergency shaping")
    
    # 🔥 EMERGENCY REWARD SHAPING
    MASSIVE_LINE_BONUS = 200        # Was probably ~10-20
    TETRIS_EXPLOSION = 500          # Was probably ~50
    SURVIVAL_PUNISHMENT = -5        # Discourage pure survival
    FIRST_LINE_CELEBRATION = 1000   # Massive first line bonus
    HEIGHT_PUNISHMENT = -10         # Strong height penalty
    
    print(f"   Emergency rewards:")
    print(f"     Line clear: +{MASSIVE_LINE_BONUS} (per line)")
    print(f"     Tetris: +{TETRIS_EXPLOSION}")
    print(f"     First line: +{FIRST_LINE_CELEBRATION}")
    print(f"     Survival: {SURVIVAL_PUNISHMENT}")
    print(f"     Height penalty: {HEIGHT_PUNISHMENT}")
    
    # 3. ACTION MASKING - Block passive actions
    USEFUL_ACTIONS = [1, 2, 3, 4, 5, 6]  # No NO-OP, no HOLD
    print(f"   Action masking: Only allow {USEFUL_ACTIONS}")
    
    # 4. FORCED EXPLORATION RATE
    FORCE_EXPLORATION_RATE = 0.3  # 30% random useful actions
    print(f"   Forced exploration: {FORCE_EXPLORATION_RATE*100}%")
    
    # Training parameters
    target_episodes = start_episode + 500  # Train 500 more episodes
    lines_cleared_total = 0
    breakthrough_threshold = 50  # Need 50+ total lines for breakthrough
    
    # Track emergency metrics
    lines_per_episode = []
    emergency_rewards = []
    exploration_actions = 0
    total_actions = 0
    
    print(f"\n🎯 EMERGENCY TRAINING:")
    print(f"Episodes {start_episode+1} to {target_episodes}")
    print(f"Goal: {breakthrough_threshold} total lines cleared")
    print("="*60)
    
    for episode in range(start_episode, target_episodes):
        obs, info = env.reset()
        episode_reward = 0
        original_reward = 0
        episode_steps = 0
        lines_this_episode = 0
        max_height = 0
        done = False
        
        while not done:
            # 🔥 FORCED ACTION SELECTION
            total_actions += 1
            
            if random.random() < FORCE_EXPLORATION_RATE:
                # Force random useful action
                action = random.choice(USEFUL_ACTIONS)
                exploration_actions += 1
            else:
                # Agent's choice, but mask useless actions
                action = agent.select_action(obs)
                if action not in USEFUL_ACTIONS:
                    action = random.choice(USEFUL_ACTIONS)
            
            # Environment step
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            original_reward += reward
            
            # 🔥 EMERGENCY REWARD SHAPING
            shaped_reward = reward
            
            # Line clear bonuses
            lines_cleared = info.get('lines_cleared', 0)
            if lines_cleared > 0:
                lines_this_episode += lines_cleared
                shaped_reward += lines_cleared * MASSIVE_LINE_BONUS
                
                # First line celebration
                if lines_cleared_total == 0:
                    shaped_reward += FIRST_LINE_CELEBRATION
                    print(f"\n🎉 FIRST LINE CLEARED! Episode {episode+1}")
                    print(f"   Reward bonus: +{FIRST_LINE_CELEBRATION}")
                
                # Tetris explosion
                if lines_cleared == 4:
                    shaped_reward += TETRIS_EXPLOSION
                    print(f"\n💥 TETRIS! Episode {episode+1}, 4 lines!")
                    print(f"   Reward bonus: +{TETRIS_EXPLOSION}")
            
            # Survival punishment (discourage pure survival)
            if not done:
                shaped_reward += SURVIVAL_PUNISHMENT
            
            # Height punishment (force aggressive play)
            try:
                # Extract board and calculate max height
                if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'board'):
                    board = env.unwrapped.board
                    if board is not None:
                        heights = []
                        for col in range(board.shape[1]):
                            height = 0
                            for row in range(board.shape[0]):
                                if board[row, col] != 0:
                                    height = board.shape[0] - row
                                    break
                            heights.append(height)
                        max_height = max(heights) if heights else 0
                        shaped_reward += max_height * HEIGHT_PUNISHMENT
            except:
                pass  # Skip if can't access board
            
            # Store experience with emergency shaping
            agent.remember(obs, action, shaped_reward, next_obs, done, info)
            
            # Learn aggressively (every step)
            if len(agent.memory) >= agent.batch_size:
                agent.learn()
            
            episode_reward += shaped_reward
            episode_steps += 1
            obs = next_obs
        
        # Episode completed
        lines_cleared_total += lines_this_episode
        lines_per_episode.append(lines_this_episode)
        emergency_rewards.append(episode_reward)
        
        # End episode with original reward (for proper tracking)
        agent.end_episode(original_reward, episode_steps, lines_this_episode)
        
        # Progress reporting
        if (episode + 1) % 5 == 0 or lines_this_episode > 0:
            exploration_rate = exploration_actions / max(1, total_actions)
            recent_lines = sum(lines_per_episode[-20:]) if len(lines_per_episode) >= 20 else sum(lines_per_episode)
            
            print(f"Episode {episode+1:4d} | "
                  f"Lines: {lines_this_episode} (Total: {lines_cleared_total:2d}) | "
                  f"Reward: {episode_reward:7.1f} | "
                  f"Height: {max_height:2.0f} | "
                  f"Explore: {exploration_rate:.2f} | "
                  f"Eps: {agent.epsilon:.3f}")
            
            if lines_this_episode > 0:
                print(f"         🎯 Line clear success! Running total: {lines_cleared_total}")
        
        # Check for breakthrough
        if lines_cleared_total >= breakthrough_threshold:
            print(f"\n🎉 BREAKTHROUGH ACHIEVED!")
            print(f"Total lines cleared: {lines_cleared_total}")
            print(f"Episodes for breakthrough: {episode - start_episode + 1}")
            agent.save_checkpoint(episode + 1)
            break
        
        # Save periodically
        if (episode + 1) % 50 == 0:
            agent.save_checkpoint(episode + 1)
            print(f"   💾 Checkpoint saved at episode {episode + 1}")
    
    env.close()
    
    # Final analysis
    episodes_trained = episode - start_episode + 1
    exploration_rate = exploration_actions / max(1, total_actions)
    avg_lines_per_episode = lines_cleared_total / episodes_trained
    
    print(f"\n" + "="*60)
    print(f"EMERGENCY TRAINING COMPLETE")
    print(f"="*60)
    print(f"Episodes trained: {episodes_trained}")
    print(f"Total lines cleared: {lines_cleared_total}")
    print(f"Average lines per episode: {avg_lines_per_episode:.3f}")
    print(f"Forced exploration rate: {exploration_rate:.1%}")
    print(f"Breakthrough achieved: {lines_cleared_total >= breakthrough_threshold}")
    
    if lines_cleared_total >= breakthrough_threshold:
        print(f"\n✅ SUCCESS! Agent broke through the plateau!")
        print(f"Next steps:")
        print(f"1. Continue training with normal settings:")
        print(f"   python train.py --episodes {episode + 2000} --resume")
        print(f"2. The agent now knows line clearing is valuable")
        print(f"3. Reduce reward shaping gradually as it improves")
        
    elif lines_cleared_total >= 10:
        print(f"\n⚠️  PARTIAL SUCCESS! Some progress made")
        print(f"Lines cleared: {lines_cleared_total}/50")
        print(f"Recommend:")
        print(f"1. Run emergency training again with higher bonuses")
        print(f"2. Or try curriculum learning approach")
        
    else:
        print(f"\n❌ PLATEAU PERSISTS")
        print(f"Only {lines_cleared_total} lines cleared")
        print(f"This indicates a fundamental issue:")
        print(f"1. Vision problem (run diagnostics again)")
        print(f"2. Action space issues")
        print(f"3. Network architecture unsuitable for Tetris")
        print(f"4. Environment reward structure incompatible")
    
    return lines_cleared_total >= breakthrough_threshold


if __name__ == "__main__":
    print("Starting Emergency Breakthrough Training...")
    print("This will attempt to break your 62k episode plateau")
    print()
    
    success = emergency_breakthrough_training()
    
    if success:
        print("\n🎉 MISSION ACCOMPLISHED!")
        print("Your agent has broken through the plateau!")
    else:
        print("\n🔧 Additional intervention needed")
        print("Consider checking vision system or trying different approaches")