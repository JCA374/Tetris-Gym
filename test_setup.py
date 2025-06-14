#!/usr/bin/env python3
"""
Comprehensive test script for Tetris Gymnasium setup
Tests all components before training
"""

import os
import sys
import traceback
import time
import numpy as np
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


def test_imports():
    """Test all required imports"""
    print("Testing imports...")

    required_packages = [
        ('torch', 'PyTorch'),
        ('numpy', 'NumPy'),
        ('matplotlib', 'Matplotlib'),
        ('gymnasium', 'Gymnasium'),
        ('tetris_gymnasium', 'Tetris Gymnasium'),
        ('cv2', 'OpenCV'),
        ('PIL', 'Pillow'),
        ('tqdm', 'tqdm'),
    ]

    results = {}

    for package, name in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {name}")
            results[package] = True
        except ImportError as e:
            print(f"  ❌ {name}: {e}")
            results[package] = False

    # Test optional packages
    optional_packages = [
        ('tensorboard', 'TensorBoard'),
        ('wandb', 'Weights & Biases'),
    ]

    print("\nOptional packages:")
    for package, name in optional_packages:
        try:
            __import__(package)
            print(f"  ✅ {name}")
            results[package] = True
        except ImportError:
            print(f"  ⚠️  {name} (optional)")
            results[package] = False

    return all(results[pkg] for pkg, _ in required_packages)


def test_environment():
    """Test Tetris Gymnasium environment creation"""
    print("\nTesting environment creation...")

    try:
        import gymnasium as gym
        from tetris_gymnasium.envs.tetris import Tetris

        # Test basic environment creation
        env = gym.make("tetris_gymnasium/Tetris", render_mode="rgb_array")
        print("  ✅ Environment created successfully")

        # Test environment properties
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")
        print(f"  Action space size: {env.action_space.n}")

        # Test reset
        obs, info = env.reset(seed=42)
        print(f"  ✅ Reset successful - Observation shape: {obs.shape}")
        print(f"  Info keys: {list(info.keys())}")

        # Test a few steps
        total_reward = 0
        for step in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if step == 0:
                print(f"  Step {step+1}: Action {action}, Reward {reward:.3f}, "
                      f"Terminated: {terminated}, Truncated: {truncated}")

            if terminated or truncated:
                obs, info = env.reset()
                break

        print(
            f"  ✅ Environment stepping works - Total reward: {total_reward:.3f}")

        env.close()
        return True

    except Exception as e:
        print(f"  ❌ Environment test failed: {e}")
        traceback.print_exc()
        return False


def test_environment_configs():
    """Test different environment configurations"""
    print("\nTesting environment configurations...")

    try:
        import gymnasium as gym

        configs = [
            {"board_height": 20, "board_width": 10},  # Standard
            {"board_height": 15, "board_width": 8},   # Smaller
            {"board_height": 10, "board_width": 6},   # Very small
        ]

        for i, config in enumerate(configs):
            env = gym.make("tetris_gymnasium/Tetris",
                           render_mode="rgb_array",
                           **config)
            obs, info = env.reset()
            print(
                f"  ✅ Config {i+1}: {config['board_height']}x{config['board_width']} - Shape: {obs.shape}")
            env.close()

        return True

    except Exception as e:
        print(f"  ❌ Configuration test failed: {e}")
        traceback.print_exc()
        return False


def test_config_module():
    """Test our config module"""
    print("\nTesting config module...")

    try:
        from config import make_env, test_environment, ENV_NAME

        print(f"  Environment name: {ENV_NAME}")

        # Test make_env function
        env = make_env(render_mode="rgb_array")
        print(
            f"  ✅ make_env works - Observation space: {env.observation_space}")

        # Test with preprocessing
        env_processed = make_env(
            render_mode="rgb_array", preprocess=True, frame_stack=4)
        obs, info = env_processed.reset()
        print(f"  ✅ Preprocessing works - Processed shape: {obs.shape}")

        env.close()
        env_processed.close()

        # Test built-in test function
        if test_environment():
            print("  ✅ Built-in test function works")
        else:
            print("  ❌ Built-in test function failed")
            return False

        return True

    except Exception as e:
        print(f"  ❌ Config module test failed: {e}")
        traceback.print_exc()
        return False


def test_model():
    """Test model creation and forward pass"""
    print("\nTesting model architectures...")

    try:
        from src.model import DQN, DuelingDQN, create_model
        import gymnasium as gym

        # Create dummy spaces
        obs_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 4), dtype=np.uint8)
        action_space = gym.spaces.Discrete(7)

        # Test regular DQN
        model = create_model(obs_space, action_space, "dqn")
        dummy_input = torch.randn(1, 4, 84, 84)  # Batch format
        output = model(dummy_input)
        print(f"  ✅ DQN: Input {dummy_input.shape} -> Output {output.shape}")

        # Test Dueling DQN
        dueling_model = create_model(obs_space, action_space, "dueling_dqn")
        output2 = dueling_model(dummy_input)
        print(
            f"  ✅ Dueling DQN: Input {dummy_input.shape} -> Output {output2.shape}")

        # Test with feature vector input
        feature_space = gym.spaces.Box(
            low=-1, high=1, shape=(200,), dtype=np.float32)
        feature_model = create_model(feature_space, action_space, "dqn")
        feature_input = torch.randn(1, 200)
        feature_output = feature_model(feature_input)
        print(
            f"  ✅ Feature DQN: Input {feature_input.shape} -> Output {feature_output.shape}")

        return True

    except Exception as e:
        print(f"  ❌ Model test failed: {e}")
        traceback.print_exc()
        return False


def test_agent():
    """Test agent creation and basic functionality"""
    print("\nTesting agent...")

    try:
        from src.agent import Agent
        from config import make_env

        env = make_env(render_mode="rgb_array")

        # Create agent
        agent = Agent(env.observation_space, env.action_space)
        print(f"  ✅ Agent created - Device: {agent.device}")

        # Test action selection
        obs, info = env.reset()
        action = agent.select_action(obs)
        print(f"  ✅ Action selection works - Action: {action}")

        # Test memory storage
        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.remember(obs, action, reward, next_obs, terminated or truncated)
        print(f"  ✅ Memory storage works - Memory size: {len(agent.memory)}")

        # Test learning (should not crash)
        for _ in range(100):  # Add more experiences
            obs = next_obs if not (terminated or truncated) else env.reset()[0]
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.remember(obs, action, reward, next_obs,
                           terminated or truncated)

        agent.learn()
        print(f"  ✅ Learning works - Steps: {agent.steps_done}")

        env.close()
        return True

    except Exception as e:
        print(f"  ❌ Agent test failed: {e}")
        traceback.print_exc()
        return False


def test_utilities():
    """Test utility functions"""
    print("\nTesting utilities...")

    try:
        from src.utils import make_dir, TrainingLogger, plot_rewards

        # Test directory creation
        test_dir = "test_outputs"
        make_dir(test_dir)
        print(f"  ✅ Directory creation works")

        # Test logger
        logger = TrainingLogger(test_dir, "test_experiment")
        logger.log_episode(1, 100.0, 50, 0.9)
        logger.save_logs()
        print(f"  ✅ Logger works")

        # Test plotting
        dummy_rewards = np.random.randn(50).cumsum() + 100
        plot_path = os.path.join(test_dir, "test_plot.png")
        plot_rewards(dummy_rewards, plot_path)
        print(f"  ✅ Plotting works")

        # Cleanup
        import shutil
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

        return True

    except Exception as e:
        print(f"  ❌ Utilities test failed: {e}")
        traceback.print_exc()
        return False


def test_performance():
    """Test performance and benchmark environment"""
    print("\nTesting performance...")

    try:
        from config import make_env
        from src.utils import benchmark_environment

        env = make_env(render_mode=None)  # No rendering for speed

        # Run benchmark
        results = benchmark_environment(env, n_steps=100)

        print(f"  ✅ Performance test completed:")
        print(f"    Steps per second: {results['steps_per_second']:.0f}")
        print(f"    Avg step time: {results['avg_step_time']*1000:.2f} ms")

        env.close()
        return True

    except Exception as e:
        print(f"  ❌ Performance test failed: {e}")
        traceback.print_exc()
        return False


def test_training_compatibility():
    """Test that everything works together for training"""
    print("\nTesting training compatibility...")

    try:
        from config import make_env
        from src.agent import Agent

        # Create environment and agent
        env = make_env(render_mode="rgb_array")
        agent = Agent(env.observation_space, env.action_space)

        print(f"  Environment-Agent compatibility: ✅")

        # Simulate a few training steps
        obs, info = env.reset()
        total_reward = 0

        for step in range(10):
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.remember(obs, action, reward, next_obs,
                           terminated or truncated)

            if len(agent.memory) >= agent.batch_size:
                agent.learn()

            total_reward += reward
            obs = next_obs

            if terminated or truncated:
                obs, info = env.reset()

        print(
            f"  ✅ Training simulation works - Total reward: {total_reward:.3f}")
        print(
            f"  Agent steps: {agent.steps_done}, Memory: {len(agent.memory)}")

        env.close()
        return True

    except Exception as e:
        print(f"  ❌ Training compatibility test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("Tetris Gymnasium Setup Test")
    print("=" * 60)

    tests = [
        ("Package Imports", test_imports),
        ("Environment Creation", test_environment),
        ("Environment Configurations", test_environment_configs),
        ("Config Module", test_config_module),
        ("Model Architecture", test_model),
        ("Agent Functionality", test_agent),
        ("Utilities", test_utilities),
        ("Performance", test_performance),
        ("Training Compatibility", test_training_compatibility),
    ]

    results = {}

    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = 0
    total = len(tests)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:25s}: {status}")
        if result:
            passed += 1

    print("=" * 60)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Ready to start training!")
        print("\nNext steps:")
        print("1. Run: python train.py --episodes 100")
        print("2. Monitor training in logs/ directory")
        print("3. Evaluate with: python evaluate.py --episodes 10 --render")
    else:
        print("❌ Some tests failed. Please fix the issues before training.")
        print("\nInstallation help:")
        print("pip install tetris-gymnasium gymnasium torch numpy matplotlib opencv-python pillow tqdm")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
