import gymnasium as gym
import numpy as np


def manual_play():
    """
    Launches a modern Atari Tetris environment using Gymnasium.
    No ROM files needed - uses the built-in Atari environments.
    """
    print("Setting up Tetris environment...")

    try:
        # Create Atari Tetris environment using modern Gymnasium
        # This automatically handles ROM licensing
        import gymnasium as gym
        env = gym.make("ALE/Tetris-v5", render_mode="human")

        print("✅ Environment created successfully!")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        print(f"Available actions: {env.unwrapped.get_action_meanings()}")

    except Exception as e:
        print(f"❌ Error creating environment: {e}")
        print("Make sure you have gymnasium with Atari support installed:")
        print("pip install 'gymnasium[atari,accept-rom-license]'")
        return

    action_meanings = env.unwrapped.get_action_meanings()
    print(f"\nAvailable actions ({len(action_meanings)} total):")
    for i, action_name in enumerate(action_meanings):
        print(f"  {i}: {action_name}")
    print("\nControls:")
    print(f"  Type the action index each frame (0-{len(action_meanings)-1})")
    print("  Type 'q' to quit")
    print("  Type 'r' to reset the game")
    print("-" * 40)

    try:
        # Reset environment
        state = env.reset()
        if isinstance(state, tuple):  # Handle new gym API
            state = state[0]

        done = False
        step_count = 0
        total_reward = 0

        while not done:
            # Render the game
            env.render()

            # Get user input
            cmd = input(
                f"Step {step_count} | Total Reward: {total_reward:.1f} | Action [0-{env.action_space.n-1}], 'r' to reset, or 'q' to quit: ").strip()

            if cmd.lower() == 'q':
                print("Quitting...")
                break
            elif cmd.lower() == 'r':
                print("Resetting environment...")
                result = env.reset()
                if isinstance(result, tuple):
                    state = result[0]
                else:
                    state = result
                done = False
                step_count = 0
                total_reward = 0
                continue

            # Validate action input
            try:
                action = int(cmd)
                if not 0 <= action < env.action_space.n:
                    raise ValueError(
                        f"Action must be between 0 and {env.action_space.n-1}")
            except ValueError as e:
                print(f"  ❌ Invalid input: {e}")
                continue

            # Take the action
            try:
                result = env.step(action)

                # Handle both old and new gym API returns
                if len(result) == 4:
                    next_state, reward, done, info = result
                    truncated = False
                else:  # len(result) == 5, new gym API
                    next_state, reward, terminated, truncated, info = result
                    done = terminated or truncated

                state = next_state
                total_reward += reward
                step_count += 1

                action_name = env.unwrapped.get_action_meanings()[action]
                print(
                    f"  ▶ Action: {action_name} | Reward: {reward:.1f} | Done: {done}")

                # Print some game info if available
                if hasattr(env.unwrapped, 'ale') and hasattr(env.unwrapped.ale, 'lives'):
                    lives = env.unwrapped.ale.lives()
                    print(f"    Lives: {lives}")

            except Exception as e:
                print(f"  ❌ Error taking action: {e}")
                continue

    except KeyboardInterrupt:
        print("\n\nInterrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"❌ Unexpected error during gameplay: {e}")
    finally:
        try:
            env.close()
            print("Environment closed. Goodbye! 👋")
        except:
            pass


def check_installation():
    """Check if all required packages are properly installed"""
    print("Checking installation...")

    try:
        import gymnasium as gym
        print("✅ gymnasium imported successfully")
    except ImportError as e:
        print(f"❌ gymnasium import failed: {e}")
        return False

    try:
        import ale_py
        print("✅ ale_py imported successfully")
    except ImportError as e:
        print(f"❌ ale_py import failed: {e}")
        print("Try: pip install ale-py")
        return False

    try:
        # Test if ALE environments are available
        import gymnasium as gym
        env_id = "ALE/Tetris-v5"
        print(f"Testing environment: {env_id}")
        env = gym.make(env_id, render_mode="rgb_array")
        env.close()
        print("✅ ALE/Tetris-v5 environment created successfully")
        return True
    except Exception as e:
        print(f"❌ Environment creation failed: {e}")
        return False


def test_environment_only():
    """
    Just test if the environment can be created and reset without manual input.
    """
    print("Testing environment creation...")

    try:
        import gymnasium as gym
        env = gym.make("ALE/Tetris-v5", render_mode="rgb_array")

        print("✅ Environment created successfully!")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        print(f"Action meanings: {env.unwrapped.get_action_meanings()}")

        # Test reset
        result = env.reset()
        if isinstance(result, tuple):
            state = result[0]
        else:
            state = result
        print(f"✅ Environment reset successful! State shape: {state.shape}")

        # Test a few random actions
        print("Testing random actions...")
        for i in range(5):
            action = env.action_space.sample()
            result = env.step(action)

            if len(result) == 4:
                next_state, reward, done, info = result
            else:
                next_state, reward, terminated, truncated, info = result
                done = terminated or truncated

            action_name = env.unwrapped.get_action_meanings()[action]
            print(
                f"  Step {i+1}: Action {action} ({action_name}), Reward {reward:.1f}, Done {done}")

            if done:
                result = env.reset()
                if isinstance(result, tuple):
                    state = result[0]
                else:
                    state = result
                print("  Environment reset after done=True")
                break

        env.close()
        print("✅ Environment test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Tetris AI Manual Test")
    print("=" * 40)

    # First test if environment works at all
    if test_environment_only():
        print("\nEnvironment test passed! Starting manual play...")
        print("=" * 40)
        manual_play()
    else:
        print("\nEnvironment test failed. Please check your installation:")
        print("1. pip install gym-tetris nes-py")
        print("2. Make sure you have the ROM file in the right place")
        print("3. Try running: python -c 'import gym_tetris; print(\"Import successful\")'")
