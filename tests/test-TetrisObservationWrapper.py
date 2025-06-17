#!/usr/bin/env python3
"""
Simple test script that doesn't read files
Save as test_fixed.py
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


def test_imports_and_basic():
    """Test basic imports and functionality"""
    print("🔧 Testing Fixed Environment and Agent")
    print("="*50)

    try:
        print("1. Testing imports...")

        # Test basic imports
        from config import make_env, test_environment
        print("   ✅ Config imported")

        from src.agent import Agent
        print("   ✅ Agent imported")

        print("\n2. Testing environment creation...")

        # Test environment creation
        env = make_env(frame_stack=4)
        print(f"   ✅ Environment created: {env.observation_space}")

        print("\n3. Testing observation consistency...")

        # Test observation consistency
        shapes = []
        obs, info = env.reset(seed=42)
        shapes.append(obs.shape)
        print(f"   Initial observation shape: {obs.shape}")

        for i in range(20):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            shapes.append(obs.shape)

            if terminated or truncated:
                obs, info = env.reset()
                shapes.append(obs.shape)

        unique_shapes = set(shapes)
        if len(unique_shapes) == 1:
            print(f"   ✅ All shapes consistent: {list(unique_shapes)[0]}")
        else:
            print(f"   ❌ Shape inconsistency: {unique_shapes}")
            return False

        env.close()

        print("\n4. Testing agent creation...")

        # Test agent creation
        env = make_env(frame_stack=4)
        agent = Agent(
            obs_space=env.observation_space,
            action_space=env.action_space,
            reward_shaping="simple"
        )
        print(f"   ✅ Agent created successfully")

        print("\n5. Testing agent-environment interaction...")

        # Test basic interaction
        obs, info = env.reset()

        for step in range(50):
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            # Store experience
            agent.remember(obs, action, reward, next_obs,
                           terminated or truncated, info)

            # Try learning every 10 steps
            if step > 0 and step % 10 == 0 and len(agent.memory) >= agent.batch_size:
                try:
                    metrics = agent.learn()
                    if metrics:
                        print(
                            f"   ✅ Learning successful at step {step}: loss={metrics['loss']:.4f}")
                        break
                    else:
                        print(f"   ⚠️  Learning returned None at step {step}")
                except Exception as e:
                    print(f"   ❌ Learning failed at step {step}: {e}")
                    return False

            obs = next_obs
            if terminated or truncated:
                obs, info = env.reset()

        env.close()

        print("\n6. Testing short training episodes...")

        # Test full episodes
        env = make_env(frame_stack=4)
        agent = Agent(
            obs_space=env.observation_space,
            action_space=env.action_space,
            reward_shaping="simple"
        )

        for episode in range(3):
            obs, info = env.reset()
            episode_reward = 0
            episode_steps = 0
            lines_cleared = 0

            done = False
            while not done and episode_steps < 500:  # Limit episode length
                action = agent.select_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(
                    action)
                done = terminated or truncated

                agent.remember(obs, action, reward, next_obs, done, info)

                # Learn every 4 steps
                if episode_steps > 0 and episode_steps % 4 == 0 and len(agent.memory) >= agent.batch_size:
                    agent.learn()

                episode_reward += reward
                episode_steps += 1
                lines_cleared += info.get('lines_cleared', 0)
                obs = next_obs

            agent.end_episode(episode_reward, episode_steps, lines_cleared)

            print(f"   Episode {episode+1}: reward={episode_reward:.1f}, "
                  f"steps={episode_steps}, lines={lines_cleared}, eps={agent.epsilon:.3f}")

        env.close()

        print(f"\n🎉 ALL TESTS PASSED!")
        print("="*50)
        print("✅ Environment produces consistent observations")
        print("✅ Agent handles observations correctly")
        print("✅ Learning works without tensor shape errors")
        print("✅ Training episodes run successfully")

        print(f"\nYour environment and agent are now fixed!")
        print("You can run training with:")
        print("  python train.py --episodes 100 --reward_shaping simple")

        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_imports_and_basic()

    if success:
        print("\n" + "="*50)
        print("🎯 READY FOR TRAINING!")
        print("="*50)
        print("Next steps:")
        print("1. Replace your config.py with the fixed version")
        print("2. Replace your src/agent.py with the fixed version")
        print("3. Run: python train.py --episodes 100 --reward_shaping simple")
        print("")
        print("The tensor shape issue should now be completely resolved!")
    else:
        print("❌ Please fix the issues above before training")
