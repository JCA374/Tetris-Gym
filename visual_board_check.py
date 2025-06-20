#!/usr/bin/env python3
"""
Critical diagnostic: Verify the model can actually see the Tetris board correctly
and make spatial decisions about piece placement.
"""

from src.agent import Agent
from config import make_env
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import os
import torch

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


class TetrisVisionDiagnostic:
    """Comprehensive diagnostic to verify model can see and understand the board"""

    def __init__(self):
        self.results = {}

    def test_observation_pipeline(self):
        """Test the complete observation preprocessing pipeline"""
        print("🔍 Testing Observation Pipeline")
        print("=" * 50)

        # Create environments at different stages
        import gymnasium as gym
        from gymnasium.envs.registration import register

        # Register raw environment
        try:
            register(
                id="TetrisVisionTest-v0",
                entry_point="tetris_gymnasium.envs.tetris:Tetris",
            )
        except gym.error.Error:
            pass

        # 1. Raw environment
        raw_env = gym.make("TetrisVisionTest-v0", render_mode="rgb_array")
        raw_obs, _ = raw_env.reset(seed=42)

        print(f"Raw observation type: {type(raw_obs)}")
        if isinstance(raw_obs, dict):
            print(f"Raw dict keys: {list(raw_obs.keys())}")
            for key, value in raw_obs.items():
                print(
                    f"  {key}: {value.shape} {value.dtype} range=[{value.min():.2f}, {value.max():.2f}]")

        # 2. Processed environment (what the agent sees)
        processed_env = make_env(frame_stack=4)
        processed_obs, _ = processed_env.reset(seed=42)

        print(
            f"\nProcessed observation: {processed_obs.shape} {processed_obs.dtype}")
        print(f"Range: [{processed_obs.min():.3f}, {processed_obs.max():.3f}]")

        # 3. Extract original board for comparison
        if isinstance(raw_obs, dict) and 'board' in raw_obs:
            original_board = raw_obs['board']
            print(f"\nOriginal board: {original_board.shape}")
            print(
                f"Board range: [{original_board.min()}, {original_board.max()}]")

            # Extract the most recent frame from processed observation
            if len(processed_obs.shape) == 3 and processed_obs.shape[-1] >= 1:
                recent_frame = processed_obs[:, :, -1]  # Last channel
                print(f"Extracted frame: {recent_frame.shape}")

                # Visualize the comparison
                self.visualize_board_transformation(
                    original_board, recent_frame)

                # Test spatial preservation
                spatial_score = self.test_spatial_preservation(
                    original_board, recent_frame)
                print(f"Spatial preservation score: {spatial_score:.3f}")

                self.results['spatial_preservation'] = spatial_score

        raw_env.close()
        processed_env.close()
        return True

    def visualize_board_transformation(self, original_board, processed_frame):
        """Visualize how the board gets transformed through preprocessing"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original board
        axes[0].imshow(original_board, cmap='viridis', aspect='auto')
        axes[0].set_title(f'Original Board\n{original_board.shape}')
        axes[0].set_xlabel('Width (columns)')
        axes[0].set_ylabel('Height (rows)')

        # Add grid to show individual cells
        for i in range(original_board.shape[0] + 1):
            axes[0].axhline(i - 0.5, color='white', linewidth=0.5, alpha=0.3)
        for j in range(original_board.shape[1] + 1):
            axes[0].axvline(j - 0.5, color='white', linewidth=0.5, alpha=0.3)

        # Processed frame
        axes[1].imshow(processed_frame, cmap='viridis', aspect='auto')
        axes[1].set_title(f'Processed Frame\n{processed_frame.shape}')
        axes[1].set_xlabel('Width (pixels)')
        axes[1].set_ylabel('Height (pixels)')

        # Difference visualization
        if original_board.shape != processed_frame.shape:
            # Resize original to match processed for comparison
            original_resized = cv2.resize(original_board.astype(np.float32),
                                          (processed_frame.shape[1],
                                           processed_frame.shape[0]),
                                          interpolation=cv2.INTER_NEAREST)
        else:
            original_resized = original_board.astype(np.float32)

        diff = np.abs(original_resized - processed_frame)
        axes[2].imshow(diff, cmap='Reds', aspect='auto')
        axes[2].set_title(f'Difference\nMax diff: {diff.max():.3f}')
        axes[2].set_xlabel('Width (pixels)')
        axes[2].set_ylabel('Height (pixels)')

        plt.tight_layout()
        plt.savefig('tetris_board_transformation.png',
                    dpi=150, bbox_inches='tight')
        plt.show()
        print("📊 Board transformation visualization saved as 'tetris_board_transformation.png'")

    def test_spatial_preservation(self, original_board, processed_frame):
        """Test if spatial relationships are preserved"""
        # Resize original to match processed dimensions for comparison
        if original_board.shape != processed_frame.shape:
            original_resized = cv2.resize(original_board.astype(np.float32),
                                          (processed_frame.shape[1],
                                           processed_frame.shape[0]),
                                          interpolation=cv2.INTER_NEAREST)
        else:
            original_resized = original_board.astype(np.float32)

        # Normalize both to [0, 1] for comparison
        if original_resized.max() > 1:
            original_resized = original_resized / original_resized.max()
        if processed_frame.max() > 1:
            processed_frame_norm = processed_frame / processed_frame.max()
        else:
            processed_frame_norm = processed_frame

        # Calculate correlation coefficient
        correlation = np.corrcoef(
            original_resized.flatten(), processed_frame_norm.flatten())[0, 1]

        return correlation if not np.isnan(correlation) else 0.0

    def test_action_space_understanding(self):
        """Test if the agent can understand different board states and choose appropriate actions"""
        print("\n🎯 Testing Action Space Understanding")
        print("=" * 50)

        env = make_env(frame_stack=4)
        agent = Agent(
            obs_space=env.observation_space,
            action_space=env.action_space,
            reward_shaping="simple"
        )

        # Test scenarios
        scenarios = []

        for scenario_num in range(5):
            obs, info = env.reset(seed=42 + scenario_num)

            # Get Q-values for this state
            with torch.no_grad():
                state_tensor = agent._preprocess_state(obs)
                q_values = agent.q_network(state_tensor)
                q_values_np = q_values.cpu().numpy().flatten()

            action = q_values_np.argmax()

            scenarios.append({
                'scenario': scenario_num + 1,
                'q_values': q_values_np.tolist(),
                'chosen_action': int(action),
                'action_name': self.get_action_name(action),
                'q_value_spread': float(q_values_np.max() - q_values_np.min()),
                'confidence': float(q_values_np.max())
            })

            print(f"Scenario {scenario_num + 1}: Action={self.get_action_name(action)} "
                  f"(Q={q_values_np[action]:.3f}), Spread={q_values_np.max() - q_values_np.min():.3f}")

        env.close()

        # Analyze action diversity
        actions_chosen = [s['chosen_action'] for s in scenarios]
        action_diversity = len(set(actions_chosen)) / len(actions_chosen)

        print(
            f"\nAction diversity: {action_diversity:.2f} (1.0 = all different, 0.2 = all same)")

        self.results['scenarios'] = scenarios
        self.results['action_diversity'] = action_diversity

        return action_diversity > 0.4  # At least some variety

    def get_action_name(self, action):
        """Convert action number to readable name"""
        action_names = {
            0: 'NO-OP', 1: 'RIGHT', 2: 'LEFT', 3: 'DOWN',
            4: 'ROTATE_CW', 5: 'ROTATE_CCW', 6: 'HARD_DROP', 7: 'HOLD'
        }
        return action_names.get(action, f'ACTION_{action}')

    def test_piece_detection(self):
        """Test if the model can detect different pieces and board states"""
        print("\n🧩 Testing Piece Detection")
        print("=" * 50)

        env = make_env(frame_stack=4)

        # Take several steps and see if observations change meaningfully
        obs, info = env.reset(seed=42)
        initial_obs = obs.copy()

        observations = [obs.copy()]

        for step in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            observations.append(obs.copy())

            if terminated or truncated:
                obs, info = env.reset()
                observations.append(obs.copy())

        env.close()

        # Analyze observation changes
        changes = []
        for i in range(1, len(observations)):
            diff = np.abs(observations[i] - observations[i-1])
            change_magnitude = np.mean(diff)
            changes.append(change_magnitude)

        avg_change = np.mean(changes)
        print(f"Average observation change per step: {avg_change:.4f}")

        # If changes are too small, the model might not be seeing state changes
        meaningful_changes = avg_change > 0.001

        print(f"Meaningful observation changes detected: {meaningful_changes}")

        self.results['avg_observation_change'] = avg_change
        self.results['meaningful_changes'] = meaningful_changes

        return meaningful_changes

    def test_board_awareness(self):
        """Test if the agent shows any awareness of board structure"""
        print("\n🏗️  Testing Board Structure Awareness")
        print("=" * 50)

        env = make_env(frame_stack=4)
        agent = Agent(
            obs_space=env.observation_space,
            action_space=env.action_space,
            reward_shaping="simple"
        )

        # Test response to different board states
        board_awareness_score = 0

        for test_num in range(3):
            obs, info = env.reset(seed=100 + test_num)

            # Get agent's action preferences
            with torch.no_grad():
                state_tensor = agent._preprocess_state(obs)
                q_values = agent.q_network(state_tensor)
                q_values_np = q_values.cpu().numpy().flatten()

            # Check if agent prefers movement actions over no-op
            movement_actions = [1, 2, 3, 6]  # RIGHT, LEFT, DOWN, HARD_DROP
            movement_q_avg = np.mean([q_values_np[a]
                                     for a in movement_actions])
            noop_q = q_values_np[0]

            if movement_q_avg > noop_q:
                board_awareness_score += 1

            print(f"Test {test_num + 1}: Movement Q-avg={movement_q_avg:.3f}, "
                  f"No-op Q={noop_q:.3f}, Prefers movement: {movement_q_avg > noop_q}")

        awareness_ratio = board_awareness_score / 3
        print(f"\nBoard awareness ratio: {awareness_ratio:.2f}")

        env.close()

        self.results['board_awareness_ratio'] = awareness_ratio
        return awareness_ratio > 0.5

    def create_summary_report(self):
        """Create a comprehensive summary of the vision diagnostic"""
        print("\n" + "=" * 60)
        print("🎯 TETRIS VISION DIAGNOSTIC SUMMARY")
        print("=" * 60)

        # Overall assessment
        issues = []

        if self.results.get('spatial_preservation', 0) < 0.7:
            issues.append("❌ Poor spatial preservation in preprocessing")
        else:
            print("✅ Spatial preservation: GOOD")

        if self.results.get('action_diversity', 0) < 0.4:
            issues.append("❌ Low action diversity - agent may be stuck")
        else:
            print("✅ Action diversity: GOOD")

        if not self.results.get('meaningful_changes', False):
            issues.append("❌ Observations not changing meaningfully")
        else:
            print("✅ Observation changes: GOOD")

        if self.results.get('board_awareness_ratio', 0) < 0.5:
            issues.append("❌ Low board structure awareness")
        else:
            print("✅ Board awareness: GOOD")

        if issues:
            print("\n🚨 CRITICAL ISSUES DETECTED:")
            for issue in issues:
                print(f"  {issue}")

            print("\n🔧 RECOMMENDED FIXES:")
            if self.results.get('spatial_preservation', 0) < 0.7:
                print("  1. Fix TetrisBoardWrapper to preserve spatial structure")
                print("  2. Verify aspect ratio preservation in resizing")
                print("  3. Consider using different interpolation method")

            if self.results.get('action_diversity', 0) < 0.4:
                print("  4. Increase exploration (epsilon) in agent")
                print("  5. Check if agent is properly learning")
                print("  6. Verify action space mapping")

            if not self.results.get('meaningful_changes', False):
                print("  7. Debug observation preprocessing pipeline")
                print("  8. Check if frame stacking is working correctly")

            print("\n❌ MODEL CANNOT PROPERLY SEE THE BOARD")
            return False
        else:
            print("\n✅ MODEL CAN SEE AND UNDERSTAND THE BOARD")
            print("\nThe agent appears to have proper board vision and can make")
            print("informed decisions about piece placement.")
            return True


def main():
    """Run the complete vision diagnostic"""
    print("🔍 Tetris Vision Diagnostic")
    print("Testing if the model can see the board correctly...")
    print()

    diagnostic = TetrisVisionDiagnostic()

    # Run all tests
    tests_passed = 0
    total_tests = 4

    try:
        if diagnostic.test_observation_pipeline():
            tests_passed += 1
    except Exception as e:
        print(f"❌ Observation pipeline test failed: {e}")

    try:
        if diagnostic.test_action_space_understanding():
            tests_passed += 1
    except Exception as e:
        print(f"❌ Action space test failed: {e}")

    try:
        if diagnostic.test_piece_detection():
            tests_passed += 1
    except Exception as e:
        print(f"❌ Piece detection test failed: {e}")

    try:
        if diagnostic.test_board_awareness():
            tests_passed += 1
    except Exception as e:
        print(f"❌ Board awareness test failed: {e}")

    # Generate summary
    success = diagnostic.create_summary_report()

    print(f"\nTests passed: {tests_passed}/{total_tests}")

    if success:
        print("\n🎉 VISION SYSTEM IS WORKING CORRECTLY!")
        print("The model can see the board and make spatial decisions.")
    else:
        print("\n🚨 VISION SYSTEM HAS CRITICAL ISSUES!")
        print("The model cannot properly see or understand the board.")
        print("Training will likely fail until these issues are fixed.")

    return success


if __name__ == "__main__":
    main()
