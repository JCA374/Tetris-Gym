#!/usr/bin/env python3
"""
watch_agent.py - Text-based visualization of what your agent is doing
Shows the board state, actions taken, and statistics without needing GUI
"""

import os
import sys
import numpy as np
import torch
from config import make_env, MODEL_DIR
from src.agent import Agent
import time

def board_to_string(board):
    """Convert board array to readable string with column numbers"""
    if board is None:
        return "No board available"
    
    # Get board dimensions
    height, width = board.shape
    
    # Create string representation
    lines = []
    lines.append("   " + "".join([f"{i}" for i in range(width)]))  # Column numbers
    lines.append("   " + "-" * width)
    
    for row in range(height):
        row_str = f"{row:2d}|"
        for col in range(width):
            if board[row, col] > 0:
                row_str += "█"  # Filled cell
            else:
                row_str += "·"  # Empty cell
        lines.append(row_str)
    
    return "\n".join(lines)


def get_column_heights(board):
    """Calculate height of each column"""
    heights = []
    for col in range(board.shape[1]):
        column = board[:, col]
        filled = np.where(column > 0)[0]
        if len(filled) > 0:
            heights.append(board.shape[0] - filled[0])
        else:
            heights.append(0)
    return heights


def count_holes(board):
    """Count number of holes (empty cells with filled cells above)"""
    holes = 0
    for col in range(board.shape[1]):
        column = board[:, col]
        found_block = False
        for cell in column:
            if cell > 0:
                found_block = True
            elif found_block and cell == 0:
                holes += 1
    return holes


def analyze_board(board):
    """Analyze board state and return statistics"""
    heights = get_column_heights(board)
    holes = count_holes(board)
    max_height = max(heights) if heights else 0
    avg_height = sum(heights) / len(heights) if heights else 0
    bumpiness = sum(abs(heights[i] - heights[i+1]) for i in range(len(heights)-1))
    
    return {
        'heights': heights,
        'holes': holes,
        'max_height': max_height,
        'avg_height': avg_height,
        'bumpiness': bumpiness
    }


def get_action_name(action):
    """Convert action number to readable name"""
    action_names = {
        0: "NOOP    ",
        1: "LEFT    ",
        2: "RIGHT   ",
        3: "DOWN    ",
        4: "ROTATE_CW",
        5: "ROTATE_CCW",
        6: "HARD_DROP"
    }
    return action_names.get(action, f"UNKNOWN({action})")


def extract_board(obs):
    """Extract board from observation"""
    if isinstance(obs, dict):
        return obs.get('board', None)
    elif isinstance(obs, np.ndarray):
        # If it's a stacked observation (C, H, W), take the last channel
        if len(obs.shape) == 3:
            return obs[-1]  # Last channel
        return obs
    return None


def watch_agent(model_path, num_episodes=3, steps_per_episode=50):
    """Watch the agent play and show what it's doing"""
    
    print("=" * 80)
    print("👀 WATCHING TETRIS AGENT")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Episodes to watch: {num_episodes}")
    print(f"Max steps per episode: {steps_per_episode}")
    print()
    
    # Create environment
    env = make_env(render_mode="rgb_array")
    
    # Load agent
    agent = Agent(
        obs_space=env.observation_space,
        action_space=env.action_space,
        model_type='dqn'
    )
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    agent.load_checkpoint(path=model_path)
    print(f"✅ Agent loaded (epsilon: {agent.epsilon:.4f})")
    print()
    
    # Set to evaluation mode
    agent.q_network.eval()
    
    # Action counters
    action_counts = {i: 0 for i in range(env.action_space.n)}
    
    for episode in range(num_episodes):
        print("\n" + "=" * 80)
        print(f"🎮 EPISODE {episode + 1}/{num_episodes}")
        print("=" * 80)
        
        obs, info = env.reset()
        done = False
        step = 0
        total_reward = 0
        episode_actions = []
        
        while not done and step < steps_per_episode:
            # Get board state
            board = extract_board(obs)
            
            # Analyze board
            analysis = analyze_board(board)
            
            # Select action
            action = agent.select_action(obs, training=False)
            action_counts[action] += 1
            episode_actions.append(action)
            
            # Take action
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            # Show state every 10 steps or on important events
            if step % 10 == 0 or reward > 10 or done:
                print(f"\n--- Step {step} ---")
                print(f"Action: {get_action_name(action)}")
                print(f"Reward: {reward:.2f} (Total: {total_reward:.2f})")
                print(f"\nBoard State:")
                print(board_to_string(board))
                print(f"\nAnalysis:")
                print(f"  Column heights: {analysis['heights']}")
                print(f"  Max height: {analysis['max_height']}, Avg: {analysis['avg_height']:.1f}")
                print(f"  Holes: {analysis['holes']}, Bumpiness: {analysis['bumpiness']}")
                
                # Check for concerning patterns
                if analysis['holes'] > 5:
                    print(f"  ⚠️  WARNING: {analysis['holes']} holes detected!")
                if analysis['max_height'] > 15:
                    print(f"  ⚠️  WARNING: Board getting very high!")
                if max(analysis['heights']) - min(analysis['heights']) > 10:
                    print(f"  ⚠️  WARNING: Very uneven columns!")
                
                time.sleep(0.5)  # Pause for readability
            
            obs = next_obs
            step += 1
        
        # Episode summary
        print("\n" + "-" * 80)
        print(f"📊 EPISODE {episode + 1} SUMMARY")
        print("-" * 80)
        print(f"Total steps: {step}")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Avg reward/step: {total_reward/step:.3f}")
        
        # Action distribution for this episode
        print("\nActions taken this episode:")
        for action, name in [(0, "NOOP"), (1, "LEFT"), (2, "RIGHT"), 
                              (3, "DOWN"), (4, "ROTATE_CW"), (5, "ROTATE_CCW"), (6, "HARD_DROP")]:
            count = episode_actions.count(action)
            pct = 100 * count / len(episode_actions) if episode_actions else 0
            print(f"  {name:12s}: {count:3d} ({pct:5.1f}%)")
        
        # Analyze action patterns
        left_count = episode_actions.count(1)
        right_count = episode_actions.count(2)
        horizontal_total = left_count + right_count
        horizontal_pct = 100 * horizontal_total / len(episode_actions) if episode_actions else 0
        
        print(f"\n🎯 Pattern Analysis:")
        print(f"  Horizontal movement: {horizontal_pct:.1f}% of actions")
        if horizontal_pct < 10:
            print(f"  ❌ PROBLEM: Agent barely uses horizontal movement!")
            print(f"     This explains why it can't clear lines.")
        elif horizontal_pct < 30:
            print(f"  ⚠️  Low horizontal movement - agent needs more exploration")
        else:
            print(f"  ✅ Good horizontal movement")
        
        time.sleep(1)
    
    env.close()
    
    # Overall statistics
    print("\n" + "=" * 80)
    print("📈 OVERALL ACTION STATISTICS")
    print("=" * 80)
    total_actions = sum(action_counts.values())
    for action in range(env.action_space.n):
        count = action_counts[action]
        pct = 100 * count / total_actions if total_actions > 0 else 0
        print(f"{get_action_name(action)}: {count:4d} ({pct:5.1f}%)")
    
    # Diagnosis
    print("\n" + "=" * 80)
    print("🔍 DIAGNOSIS")
    print("=" * 80)
    
    left_pct = 100 * action_counts[1] / total_actions if total_actions > 0 else 0
    right_pct = 100 * action_counts[2] / total_actions if total_actions > 0 else 0
    horizontal_pct = left_pct + right_pct
    
    if horizontal_pct < 15:
        print("❌ CRITICAL ISSUE: Agent is NOT using horizontal movement!")
        print("   This is why it can't clear lines.")
        print("   The pieces are probably all stacking in columns 3-6.")
        print("\n💡 SOLUTION:")
        print("   1. Check that action discovery in config.py worked correctly")
        print("   2. Verify exploration in src/agent.py uses actions 1 and 2")
        print("   3. Increase early exploration weight for horizontal moves")
    elif horizontal_pct < 35:
        print("⚠️  Agent uses some horizontal movement, but not enough")
        print("   Needs more training or better exploration strategy")
    else:
        print("✅ Agent uses horizontal movement appropriately")
        print("   If still not clearing lines, check:")
        print("   - Rotation usage")
        print("   - Hole creation")
        print("   - Reward function")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Watch your Tetris agent in text mode')
    parser.add_argument('--model', type=str, default='models/checkpoint_latest.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--episodes', type=int, default=3,
                        help='Number of episodes to watch')
    parser.add_argument('--steps', type=int, default=50,
                        help='Max steps per episode')
    
    args = parser.parse_args()
    
    watch_agent(args.model, args.episodes, args.steps)