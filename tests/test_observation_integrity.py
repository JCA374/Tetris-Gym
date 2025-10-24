#!/usr/bin/env python3
# tests/test_observation_integrity.py
"""
Comprehensive diagnostic test to verify the model is properly seeing:
1. The game board
2. The active piece (current tetromino)
3. The piece rotation state
4. Additional game state (holder, queue)

This test will reveal if the poor performance (0.02 lines/episode after 13k episodes)
is due to missing critical observation information.
"""

import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import make_env
    from src.agent import Agent
    from src.model import create_model
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you run this from the project root directory")
    sys.exit(1)


class ObservationIntegrityTester:
    """Comprehensive observation integrity testing"""
    
    def __init__(self):
        self.results = {}
        self.test_passed = True
        
    def run_all_tests(self):
        """Run complete test suite"""
        print("="*80)
        print("🔬 TETRIS OBSERVATION INTEGRITY TEST SUITE")
        print("="*80)
        print("\nThis will diagnose why your model plateaued at 0.02 lines/episode")
        print("after 13,000+ episodes of training.\n")
        
        tests = [
            ("Environment Observation Space", self.test_env_observation_space),
            ("Raw Observation Content", self.test_raw_observation_content),
            ("Active Piece Visibility", self.test_active_piece_visibility),
            ("Observation Consistency", self.test_observation_consistency),
            ("Agent Preprocessing", self.test_agent_preprocessing),
            ("Model Input Shape", self.test_model_input_shape),
            ("Information Completeness", self.test_information_completeness),
            ("Dynamic State Changes", self.test_dynamic_changes),
        ]
        
        for test_name, test_func in tests:
            print(f"\n{'='*80}")
            print(f"🧪 TEST: {test_name}")
            print(f"{'='*80}")
            try:
                result = test_func()
                self.results[test_name] = result
                status = "✅ PASSED" if result else "❌ FAILED"
                print(f"\n{status}: {test_name}")
            except Exception as e:
                print(f"\n❌ ERROR in {test_name}: {e}")
                import traceback
                traceback.print_exc()
                self.results[test_name] = False
                self.test_passed = False
        
        self.print_summary()
        self.generate_diagnostic_report()
        
    def test_env_observation_space(self):
        """Test what the environment claims to provide"""
        print("\n📋 Checking environment observation space...")
        
        env = make_env(render_mode="rgb_array")
        obs_space = env.observation_space
        
        print(f"Observation Space Type: {type(obs_space)}")
        print(f"Observation Space: {obs_space}")
        
        if hasattr(obs_space, 'shape'):
            print(f"Shape: {obs_space.shape}")
            print(f"Dtype: {obs_space.dtype}")
            
            # Analyze dimensionality
            if len(obs_space.shape) == 1:
                print("⚠️  1D FEATURE VECTOR - Model sees flattened features, not spatial structure")
                print(f"   Feature count: {obs_space.shape[0]}")
            elif len(obs_space.shape) == 2:
                print("⚠️  2D SINGLE CHANNEL - Model sees board but may miss active piece")
                print(f"   Board size: {obs_space.shape}")
            elif len(obs_space.shape) == 3:
                channels = obs_space.shape[-1] if obs_space.shape[-1] < obs_space.shape[0] else obs_space.shape[0]
                print(f"✅ 3D MULTI-CHANNEL observation with {channels} channels")
                if channels == 1:
                    print("⚠️  WARNING: Only 1 channel - active piece likely NOT visible")
                elif channels >= 4:
                    print("✅ GOOD: Multiple channels can represent board, active piece, holder, queue")
        
        env.close()
        return True
        
    def test_raw_observation_content(self):
        """Test what's actually IN the observation"""
        print("\n🔍 Examining raw observation content...")
        
        env = make_env(render_mode="rgb_array")
        
        # Get initial observation
        obs, info = env.reset(seed=42)
        
        print(f"\nObservation type: {type(obs)}")
        print(f"Observation dtype: {obs.dtype}")
        print(f"Observation shape: {obs.shape}")
        print(f"Value range: [{obs.min():.4f}, {obs.max():.4f}]")
        print(f"Non-zero elements: {np.count_nonzero(obs)} / {obs.size} ({100*np.count_nonzero(obs)/obs.size:.1f}%)")
        
        # Check info dictionary for raw game state
        print(f"\nInfo dictionary keys: {list(info.keys())}")
        
        # Look for piece information in info
        piece_keys = ['active_tetromino', 'active_tetromino_mask', 'current_piece', 
                      'piece', 'holder', 'queue', 'tetromino']
        found_pieces = {key: key in info for key in piece_keys}
        
        print("\nPiece-related info available:")
        for key, available in found_pieces.items():
            status = "✅" if available else "❌"
            print(f"  {status} {key}")
            if available and key in info:
                val = info[key]
                if hasattr(val, 'shape'):
                    print(f"      Shape: {val.shape}")
        
        env.close()
        
        # Critical check: Is active piece visible?
        has_piece_info = any(found_pieces.values())
        if not has_piece_info:
            print("\n⚠️  CRITICAL: No active piece information found in observation or info!")
            print("   This means the model CANNOT see the piece it's supposed to place!")
            return False
        
        return True
        
    def test_active_piece_visibility(self):
        """Test if the active piece is visible in observations"""
        print("\n🎯 Testing active piece visibility...")
        
        env = make_env(render_mode="rgb_array")
        
        # Collect observations across multiple steps
        obs, info = env.reset(seed=42)
        initial_obs = obs.copy()
        
        print(f"Initial observation shape: {obs.shape}")
        
        # Take actions and look for piece movement
        observations = []
        piece_positions = []
        
        for step in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            observations.append(obs.copy())
            
            # Try to extract piece information
            if 'active_tetromino_mask' in info:
                piece_positions.append(np.sum(info['active_tetromino_mask']))
            
            if terminated or truncated:
                break
        
        env.close()
        
        # Analyze observation changes
        if len(observations) > 1:
            changes = []
            for i in range(1, len(observations)):
                diff = np.abs(observations[i] - observations[i-1])
                change_magnitude = np.mean(diff)
                max_change = np.max(diff)
                changes.append((change_magnitude, max_change))
            
            avg_change = np.mean([c[0] for c in changes])
            max_changes = [c[1] for c in changes]
            
            print(f"\nObservation change statistics:")
            print(f"  Average change per step: {avg_change:.6f}")
            print(f"  Max changes: min={min(max_changes):.3f}, max={max(max_changes):.3f}, mean={np.mean(max_changes):.3f}")
            
            if avg_change < 0.001:
                print("\n⚠️  PROBLEM: Very small observation changes!")
                print("   The observation may not be encoding piece movement properly.")
                print("   Model cannot learn without seeing state changes.")
                return False
            else:
                print("\n✅ Observations change meaningfully between steps")
        
        if piece_positions:
            print(f"\nPiece pixel counts across steps: {piece_positions}")
            if all(p == 0 for p in piece_positions):
                print("⚠️  WARNING: Active piece mask is always empty!")
                return False
        
        return True
        
    def test_observation_consistency(self):
        """Test if observations have consistent shape"""
        print("\n📐 Testing observation shape consistency...")
        
        env = make_env(render_mode="rgb_array")
        
        shapes = []
        obs, info = env.reset(seed=42)
        shapes.append(obs.shape)
        
        # Collect shapes across episode boundaries
        for episode in range(3):
            for step in range(20):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                shapes.append(obs.shape)
                
                if terminated or truncated:
                    obs, info = env.reset()
                    shapes.append(obs.shape)
                    break
        
        env.close()
        
        unique_shapes = set(shapes)
        print(f"Collected {len(shapes)} observations")
        print(f"Unique shapes found: {len(unique_shapes)}")
        for shape in unique_shapes:
            count = shapes.count(shape)
            print(f"  Shape {shape}: {count} times ({100*count/len(shapes):.1f}%)")
        
        if len(unique_shapes) > 1:
            print("\n⚠️  PROBLEM: Inconsistent observation shapes!")
            print("   This will cause training instability and failures.")
            return False
        
        print("\n✅ Observation shapes are consistent")
        return True
        
    def test_agent_preprocessing(self):
        """Test how the agent preprocesses observations"""
        print("\n🔧 Testing agent observation preprocessing...")
        
        env = make_env(render_mode="rgb_array")
        agent = Agent(
            obs_space=env.observation_space,
            action_space=env.action_space,
        )
        
        obs, info = env.reset(seed=42)
        
        print(f"Raw observation shape: {obs.shape}")
        print(f"Raw observation dtype: {obs.dtype}")
        print(f"Raw observation range: [{obs.min():.3f}, {obs.max():.3f}]")
        
        # Test preprocessing
        try:
            processed = agent._preprocess_state(obs)
            print(f"\nAfter agent preprocessing:")
            print(f"  Tensor shape: {processed.shape}")
            print(f"  Tensor dtype: {processed.dtype}")
            print(f"  Tensor device: {processed.device}")
            print(f"  Value range: [{processed.min().item():.3f}, {processed.max().item():.3f}]")
            
            # Check if preprocessing preserves information
            info_ratio = processed.numel() / obs.size
            print(f"  Information ratio (processed/raw): {info_ratio:.2f}")
            
            if info_ratio < 0.9:
                print(f"\n⚠️  WARNING: Preprocessing reduces information by {100*(1-info_ratio):.1f}%")
            
        except Exception as e:
            print(f"\n❌ Preprocessing failed: {e}")
            import traceback
            traceback.print_exc()
            env.close()
            return False
        
        env.close()
        print("\n✅ Agent preprocessing works")
        return True
        
    def test_model_input_shape(self):
        """Test if the model can process the preprocessed observations"""
        print("\n🧠 Testing model input processing...")
        
        env = make_env(render_mode="rgb_array")
        agent = Agent(
            obs_space=env.observation_space,
            action_space=env.action_space,
        )
        
        obs, info = env.reset(seed=42)
        
        # Process through agent
        state_tensor = agent._preprocess_state(obs)
        
        print(f"State tensor shape: {state_tensor.shape}")
        
        # Test forward pass
        try:
            with torch.no_grad():
                q_values = agent.q_network(state_tensor)
            
            print(f"Q-values shape: {q_values.shape}")
            print(f"Q-values: {q_values.cpu().numpy().flatten()}")
            print(f"Selected action: {q_values.argmax().item()}")
            
            # Check for degenerate Q-values
            q_np = q_values.cpu().numpy().flatten()
            q_std = np.std(q_np)
            q_range = np.max(q_np) - np.min(q_np)
            
            print(f"\nQ-value statistics:")
            print(f"  Mean: {np.mean(q_np):.3f}")
            print(f"  Std: {q_std:.3f}")
            print(f"  Range: {q_range:.3f}")
            
            if q_std < 0.001:
                print("\n⚠️  WARNING: Q-values are nearly identical!")
                print("   Model may not have learned anything meaningful.")
                print("   This could indicate the input doesn't contain useful information.")
                
        except Exception as e:
            print(f"\n❌ Model forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            env.close()
            return False
        
        env.close()
        print("\n✅ Model processes inputs correctly")
        return True
        
    def test_information_completeness(self):
        """Test what information is available vs what should be available"""
        print("\n📊 Testing information completeness...")
        
        env = make_env(render_mode="rgb_array")
        obs, info = env.reset(seed=42)
        
        # Create visual comparison
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Observation Information Analysis', fontsize=16, fontweight='bold')
        
        # Parse observation based on shape
        print(f"Analyzing observation shape: {obs.shape}")
        
        if len(obs.shape) == 3:  # Multi-channel
            channels = obs.shape[-1] if obs.shape[-1] < obs.shape[0] else obs.shape[0]
            print(f"Found {channels} channels")
            
            # Determine channel order
            if obs.shape[-1] < obs.shape[0]:  # HWC format
                channel_axis = -1
                for i in range(min(channels, 4)):
                    channel = obs[:, :, i]
                    ax = axes.flat[i]
                    im = ax.imshow(channel, cmap='viridis', aspect='auto')
                    ax.set_title(f'Channel {i}\n({np.sum(channel > 0.01)} active pixels)')
                    plt.colorbar(im, ax=ax)
            else:  # CHW format
                channel_axis = 0
                for i in range(min(channels, 4)):
                    channel = obs[i, :, :]
                    ax = axes.flat[i]
                    im = ax.imshow(channel, cmap='viridis', aspect='auto')
                    ax.set_title(f'Channel {i}\n({np.sum(channel > 0.01)} active pixels)')
                    plt.colorbar(im, ax=ax)
            
            # Analysis
            active_channels = 0
            for i in range(channels):
                if channel_axis == -1:
                    channel = obs[:, :, i]
                else:
                    channel = obs[i, :, :]
                if np.sum(channel > 0.01) > 0:
                    active_channels += 1
            
            print(f"Active channels: {active_channels}/{channels}")
            
            if channels < 3:
                print("\n⚠️  WARNING: Fewer than 3 channels!")
                print("   Likely missing: active piece, holder, or queue information")
                result = False
            else:
                result = True
                
        elif len(obs.shape) == 2:  # Single channel
            print("Single 2D observation (board only)")
            ax = axes[0, 0]
            im = ax.imshow(obs, cmap='viridis', aspect='auto')
            ax.set_title('Board State Only')
            plt.colorbar(im, ax=ax)
            
            print("\n⚠️  PROBLEM: Only board visible!")
            print("   Active piece information is MISSING")
            print("   Model cannot learn piece placement without seeing pieces!")
            result = False
            
        else:  # Feature vector
            print("1D feature vector")
            ax = axes[0, 0]
            ax.plot(obs)
            ax.set_title('Feature Vector')
            ax.set_xlabel('Feature Index')
            ax.set_ylabel('Value')
            
            print("\n⚠️  Feature vector format")
            print("   Cannot visually verify piece visibility")
            result = True  # Can't definitively say it's wrong
        
        # Info comparison
        ax = axes[1, 2]
        info_sources = ['board', 'active_piece', 'holder', 'queue']
        availability = [1, 0, 0, 0]  # Assume board is always there
        
        # Check what's actually available
        if len(obs.shape) == 3 and (obs.shape[-1] >= 4 or obs.shape[0] >= 4):
            availability = [1, 1, 1, 1]
        elif len(obs.shape) == 3 and (obs.shape[-1] >= 2 or obs.shape[0] >= 2):
            availability = [1, 1, 0, 0]
        
        colors = ['green' if a else 'red' for a in availability]
        ax.bar(info_sources, availability, color=colors)
        ax.set_title('Information Sources')
        ax.set_ylabel('Available')
        ax.set_ylim([0, 1.2])
        
        for i, (source, avail) in enumerate(zip(info_sources, availability)):
            ax.text(i, avail + 0.05, '✅' if avail else '❌', ha='center', fontsize=20)
        
        plt.tight_layout()
        plt.savefig('/home/claude/tests/observation_analysis.png', dpi=150, bbox_inches='tight')
        print(f"\n📊 Visual analysis saved to: tests/observation_analysis.png")
        
        env.close()
        return result
        
    def test_dynamic_changes(self):
        """Test if observations change dynamically with actions"""
        print("\n🎬 Testing dynamic observation changes...")
        
        env = make_env(render_mode="rgb_array")
        
        obs, info = env.reset(seed=42)
        observations = [obs.copy()]
        actions = []
        
        # Collect diverse actions
        action_sequence = [1, 1, 4, 3, 3, 6]  # Right, Right, Rotate, Down, Down, Hard Drop
        
        for action in action_sequence:
            actions.append(action)
            obs, reward, terminated, truncated, info = env.step(action)
            observations.append(obs.copy())
            
            if terminated or truncated:
                break
        
        # Analyze how observations change
        print(f"\nCollected {len(observations)} observations")
        
        changes = []
        for i in range(1, len(observations)):
            diff = observations[i] - observations[i-1]
            change = {
                'step': i,
                'action': actions[i-1] if i-1 < len(actions) else None,
                'mean_change': np.mean(np.abs(diff)),
                'max_change': np.max(np.abs(diff)),
                'changed_pixels': np.sum(np.abs(diff) > 0.01),
            }
            changes.append(change)
        
        print("\nChange analysis:")
        print(f"{'Step':<6} {'Action':<8} {'Mean Δ':<12} {'Max Δ':<12} {'Changed Pixels':<15}")
        print("-" * 60)
        for c in changes:
            action_name = ['NOOP', 'RIGHT', 'LEFT', 'DOWN', 'ROTATE_CW', 'ROTATE_CCW', 'DROP', 'HOLD'][c['action']]
            print(f"{c['step']:<6} {action_name:<8} {c['mean_change']:<12.6f} {c['max_change']:<12.3f} {c['changed_pixels']:<15}")
        
        # Statistics
        mean_changes = [c['mean_change'] for c in changes]
        avg_mean_change = np.mean(mean_changes)
        
        print(f"\nAverage observation change: {avg_mean_change:.6f}")
        
        if avg_mean_change < 0.0001:
            print("\n⚠️  CRITICAL PROBLEM: Observations barely change!")
            print("   The model cannot learn if observations don't reflect actions.")
            env.close()
            return False
        
        # Check if different actions cause different changes
        action_types = {}
        for c in changes:
            act = c['action']
            if act not in action_types:
                action_types[act] = []
            action_types[act].append(c['mean_change'])
        
        print("\nChange by action type:")
        for act, changes_list in action_types.items():
            action_name = ['NOOP', 'RIGHT', 'LEFT', 'DOWN', 'ROTATE_CW', 'ROTATE_CCW', 'DROP', 'HOLD'][act]
            print(f"  {action_name}: avg change = {np.mean(changes_list):.6f}")
        
        env.close()
        print("\n✅ Observations change dynamically")
        return True
        
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*80)
        print("📋 TEST SUMMARY")
        print("="*80)
        
        passed = sum(1 for v in self.results.values() if v)
        total = len(self.results)
        
        for test_name, result in self.results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status} {test_name}")
        
        print(f"\n{passed}/{total} tests passed")
        
        if passed < total:
            print("\n🔴 CRITICAL ISSUES DETECTED!")
            print("Your model's poor performance is likely due to observation problems.")
        else:
            print("\n✅ All tests passed!")
            print("Observations appear to be correct.")
            
    def generate_diagnostic_report(self):
        """Generate a detailed diagnostic report"""
        print("\n" + "="*80)
        print("📄 DIAGNOSTIC REPORT")
        print("="*80)
        
        report_lines = []
        report_lines.append("# Tetris AI Observation Diagnostic Report\n")
        report_lines.append(f"## Summary: {sum(self.results.values())}/{len(self.results)} tests passed\n")
        
        # Identify key issues
        failed_tests = [name for name, result in self.results.items() if not result]
        
        if failed_tests:
            report_lines.append("## 🔴 Critical Issues Detected\n")
            report_lines.append("Your model is NOT seeing the game state properly!\n")
            report_lines.append("### Failed Tests:\n")
            for test in failed_tests:
                report_lines.append(f"- {test}\n")
            
            report_lines.append("\n### Root Cause Analysis:\n")
            
            if "Active Piece Visibility" in failed_tests:
                report_lines.append("**PROBLEM 1: Active Piece Not Visible**\n")
                report_lines.append("- The model cannot see the piece it's supposed to place\n")
                report_lines.append("- This explains why it only clears 0.02 lines per episode\n")
                report_lines.append("- The model is essentially playing blind\n\n")
                
                report_lines.append("**SOLUTION:**\n")
                report_lines.append("1. Use the complete vision configuration:\n")
                report_lines.append("   ```python\n")
                report_lines.append("   env = make_env(use_complete_vision=True)\n")
                report_lines.append("   ```\n")
                report_lines.append("2. Ensure observations include 4 channels:\n")
                report_lines.append("   - Channel 0: Board state\n")
                report_lines.append("   - Channel 1: Active piece\n")
                report_lines.append("   - Channel 2: Holder\n")
                report_lines.append("   - Channel 3: Queue\n\n")
            
            if "Information Completeness" in failed_tests:
                report_lines.append("**PROBLEM 2: Incomplete Information**\n")
                report_lines.append("- Model is missing critical game state\n")
                report_lines.append("- Cannot make informed decisions\n\n")
            
            if "Dynamic State Changes" in failed_tests:
                report_lines.append("**PROBLEM 3: Static Observations**\n")
                report_lines.append("- Observations don't change with actions\n")
                report_lines.append("- Model cannot learn cause-effect relationships\n\n")
            
            report_lines.append("### Expected Performance After Fix:\n")
            report_lines.append("- Lines cleared should increase from 0.02 to 1-5 per episode\n")
            report_lines.append("- Training should show clear improvement within 1000 episodes\n")
            report_lines.append("- Eventually reaching 10-50+ lines per episode\n")
            
        else:
            report_lines.append("## ✅ Observations Look Correct\n")
            report_lines.append("If performance is still poor, consider:\n")
            report_lines.append("- Hyperparameter tuning (learning rate, epsilon decay)\n")
            report_lines.append("- Reward shaping improvements\n")
            report_lines.append("- Network architecture changes\n")
            report_lines.append("- Longer training duration\n")
        
        # Save report
        report_path = '/home/claude/tests/diagnostic_report.md'
        with open(report_path, 'w') as f:
            f.writelines(report_lines)
        
        print(f"\n📄 Full report saved to: {report_path}")


def main():
    """Run the comprehensive test suite"""
    tester = ObservationIntegrityTester()
    tester.run_all_tests()
    
    print("\n" + "="*80)
    print("🎯 NEXT STEPS")
    print("="*80)
    print("\n1. Review the generated files:")
    print("   - tests/observation_analysis.png (visual analysis)")
    print("   - tests/diagnostic_report.md (detailed report)")
    print("\n2. If tests failed, fix the observation configuration")
    print("\n3. Retrain your model with correct observations")
    print("\n4. Monitor lines cleared - should improve dramatically!")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
