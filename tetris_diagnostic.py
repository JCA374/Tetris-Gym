#!/usr/bin/env python3
"""
Diagnostic tool to understand why the agent isn't clearing lines
"""

from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from src.agent import Agent
from config import make_env
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


def diagnose_agent():
    """Run diagnostic tests on the trained agent"""

    print("=== TETRIS AGENT DIAGNOSTIC ===")
    print("="*50)

    # Load environment and agent
    env = make_env(frame_stack=4)
    agent = Agent(
        obs_space=env.observation_space,
        action_space=env.action_space,
        reward_shaping="simple"
    )

    if not agent.load_checkpoint(latest=True):
        print("Error: No trained model found!")
        return

    print(f"Loaded agent with {agent.episodes_done} episodes trained")
    print(f"Current epsilon: {agent.epsilon}")

    # Test 1: Action Distribution Analysis
    print("\n1. ACTION DISTRIBUTION ANALYSIS")
    print("-" * 30)

    action_names = {
        0: "No-op",
        1: "Right",
        2: "Left",
        3: "Down",
        4: "Rotate CW",
        5: "Rotate CCW",
        6: "Hard Drop",
        7: "Hold"
    }

    # Collect action statistics
    action_counts = Counter()
    board_states = []
    max_heights = []

    for test_episode in range(10):
        obs, info = env.reset()
        done = False
        episode_actions = []

        while not done and len(episode_actions) < 200:
            # Get action from agent (eval mode)
            action = agent.select_action(obs, eval_mode=True)
            episode_actions.append(action)
            action_counts[action] += 1

            # Step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Analyze board state (if accessible)
            if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'board'):
                board = env.unwrapped.board
                if board is not None:
                    # Calculate max height
                    heights = []
                    for col in range(board.shape[1]):
                        col_height = 0
                        for row in range(board.shape[0]):
                            if board[row, col] != 0:
                                col_height = board.shape[0] - row
                                break
                        heights.append(col_height)
                    max_heights.append(max(heights))

        print(f"Episode {test_episode + 1}: {len(episode_actions)} actions")

    # Print action distribution
    total_actions = sum(action_counts.values())
    print("\nAction Distribution:")
    for action, count in sorted(action_counts.items()):
        percentage = (count / total_actions) * 100
        print(
            f"  {action} ({action_names.get(action, 'Unknown')}): {count} ({percentage:.1f}%)")

    # Test 2: Board Fill Pattern
    print("\n2. BOARD FILLING PATTERN")
    print("-" * 30)

    if max_heights:
        avg_max_height = np.mean(max_heights)
        print(f"Average maximum height: {avg_max_height:.1f}")
        print(f"Height range: {min(max_heights)} - {max(max_heights)}")

        # Check if agent is building straight up
        if avg_max_height > 15:
            print("⚠️ Agent tends to build very high!")
        elif avg_max_height < 5:
            print("⚠️ Agent keeps board very low - might be clearing pieces too early")

    # Test 3: Q-Value Analysis
    print("\n3. Q-VALUE ANALYSIS")
    print("-" * 30)

    # Get Q-values for a few states
    obs, info = env.reset()
    for step in range(5):
        with torch.no_grad():
            state_tensor = agent._preprocess_state(obs)
            q_values = agent.q_network(state_tensor)
            q_numpy = q_values.cpu().numpy().squeeze()

            print(f"\nStep {step + 1} Q-values:")
            sorted_actions = np.argsort(q_numpy)[::-1]  # Descending order
            for i, action in enumerate(sorted_actions[:3]):  # Top 3 actions
                print(
                    f"  {action} ({action_names.get(action, 'Unknown')}): {q_numpy[action]:.3f}")

        # Take best action and continue
        action = agent.select_action(obs, eval_mode=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    # Test 4: Line Clear Opportunity Detection
    print("\n4. MANUAL LINE CLEAR TEST")
    print("-" * 30)

    # Create a scenario where line clear is possible
    obs, info = env.reset()

    # Try to fill bottom row manually
    print("Attempting to create line clear opportunity...")

    forced_actions = [3, 6] * 20  # Down and hard drop repeatedly
    line_clear_achieved = False

    for action in forced_actions:
        obs, reward, terminated, truncated, info = env.step(action)
        if info.get('lines_cleared', 0) > 0:
            line_clear_achieved = True
            print(f"✅ Line cleared with forced actions! Reward: {reward}")
            break
        if terminated or truncated:
            break

    if not line_clear_achieved:
        print("❌ Even forced downward actions didn't create line clear")
        print("   The environment or reward structure might be the issue")

    env.close()

    # Final diagnosis
    print("\n" + "="*50)
    print("DIAGNOSIS SUMMARY:")
    print("="*50)

    # Check for common issues
    problems = []

    if action_counts.get(6, 0) / total_actions < 0.1:  # Less than 10% hard drops
        problems.append("- Agent rarely uses hard drop (fast piece placement)")

    if action_counts.get(3, 0) / total_actions < 0.2:  # Less than 20% down moves
        problems.append("- Agent doesn't actively move pieces down")

    if action_counts.get(0, 0) / total_actions > 0.3:  # More than 30% no-ops
        problems.append("- Agent does too many no-op actions")

    if avg_max_height < 5:
        problems.append("- Agent might be avoiding piece placement")
    elif avg_max_height > 15:
        problems.append("- Agent builds too high without clearing lines")

    if problems:
        print("Identified issues:")
        for problem in problems:
            print(problem)
    else:
        print("No obvious behavioral issues found")

    print("\nRECOMMENDATIONS:")
    print("1. Run break_plateau_train.py for focused intervention")
    print("2. Try curriculum wrapper with pre-filled rows")
    print("3. Consider imitation learning from hardcoded policy")
    print("4. Verify reward signals are being received correctly")


if __name__ == "__main__":
    import torch  # Import here to avoid issues
    diagnose_agent()
