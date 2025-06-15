#!/usr/bin/env python3
"""
Simple test to verify the working Tetris setup
"""


def test_working_setup():
    """Test that everything is working"""
    print("Testing Working Tetris Setup")
    print("=" * 40)

    try:
        # Test 1: Import and create environment
        print("1. Testing environment creation...")
        from config import make_env

        env = make_env()
        print(f"   ✅ Environment created")
        print(f"   Observation space: {env.observation_space}")
        print(f"   Action space: {env.action_space}")

        # Test 2: Reset environment
        print("\n2. Testing environment reset...")
        obs, info = env.reset()
        print(f"   ✅ Reset successful")
        print(f"   Observation shape: {obs.shape}")
        print(f"   Observation dtype: {obs.dtype}")
        print(f"   Info keys: {list(info.keys())}")

        # Test 3: Take some steps
        print("\n3. Testing environment steps...")
        total_reward = 0
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if i == 0:
                print(
                    f"   Step 1: action={action}, reward={reward}, done={terminated or truncated}")

            if terminated or truncated:
                print(f"   Episode ended at step {i+1}")
                obs, info = env.reset()

        print(f"   ✅ Environment stepping works")
        print(f"   Total reward: {total_reward}")

        env.close()

        # Test 4: Agent creation
        print("\n4. Testing agent creation...")
        from src.agent import Agent

        env = make_env()
        agent = Agent(env.observation_space, env.action_space)
        print(f"   ✅ Agent created successfully")
        print(f"   Device: {agent.device}")
        print(
            f"   Network parameters: {sum(p.numel() for p in agent.q_network.parameters())}")

        # Test 5: Agent action selection
        print("\n5. Testing agent functionality...")
        obs, info = env.reset()
        action = agent.select_action(obs)
        print(f"   ✅ Action selection works: {action}")

        # Test 6: Memory and learning
        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.remember(obs, action, reward, next_obs, terminated or truncated)
        print(f"   ✅ Memory works: {len(agent.memory)} experiences")

        # Add more experiences for learning test
        for _ in range(50):
            obs = next_obs if not (terminated or truncated) else env.reset()[0]
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.remember(obs, action, reward, next_obs,
                           terminated or truncated)

        # Test learning
        agent.learn()
        print(f"   ✅ Learning works: {agent.steps_done} steps completed")

        env.close()

        print("\n" + "=" * 40)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 40)
        print("Your setup is working perfectly!")
        print("\nNext steps:")
        print("1. Run: python train.py --episodes 50 --log_freq 5")
        print("2. Monitor progress in logs/ directory")
        print("3. Evaluate with: python evaluate.py --episodes 5 --render")

        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

        print("\nTroubleshooting:")
        print("1. Make sure you replaced config.py with the working version")
        print("2. Make sure you updated src/agent.py")
        print("3. Check that all imports are working")

        return False


if __name__ == "__main__":
    test_working_setup()
