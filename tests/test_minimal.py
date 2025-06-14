#!/usr/bin/env python3
"""
Minimal test to check if Atari environments work
"""


def test_imports():
    print("Testing imports...")
    try:
        import gymnasium
        print("✅ gymnasium imported")

        import ale_py
        print("✅ ale_py imported")

        # Check if ALE environments are registered
        import gymnasium.envs.registration
        env_ids = [env_id for env_id in gymnasium.envs.registration.registry.keys(
        ) if "ALE" in env_id]
        print(f"✅ Found {len(env_ids)} ALE environments")

        if "ALE/Tetris-v5" in env_ids:
            print("✅ ALE/Tetris-v5 is available")
        else:
            print("❌ ALE/Tetris-v5 not found")
            print("Available ALE environments:")
            tetris_envs = [env for env in env_ids if "tetris" in env.lower()]
            if tetris_envs:
                for env in tetris_envs:
                    print(f"  - {env}")
            else:
                print("  - No Tetris environments found")
                print("  - First few ALE environments:")
                for env in env_ids[:5]:
                    print(f"    - {env}")

        return True

    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def test_environment():
    print("\nTesting environment creation...")
    try:
        import gymnasium as gym

        # Try Tetris first
        try:
            env = gym.make("ALE/Tetris-v5", render_mode="rgb_array")
            print("✅ ALE/Tetris-v5 created successfully")
            env.close()
            return True
        except Exception as e:
            print(f"❌ ALE/Tetris-v5 failed: {e}")

        # Fallback to any working Atari game
        test_envs = ["ALE/Pong-v5", "ALE/Breakout-v5", "ALE/SpaceInvaders-v5"]
        for env_name in test_envs:
            try:
                env = gym.make(env_name, render_mode="rgb_array")
                print(f"✅ {env_name} works as fallback")
                env.close()
                return True
            except Exception as e:
                print(f"❌ {env_name} failed: {e}")

        return False

    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        return False


if __name__ == "__main__":
    print("Minimal ALE/Atari Test")
    print("=" * 30)

    if test_imports():
        test_environment()
    else:
        print("\nTry installing with:")
        print("pip install ale-py gymnasium")
        print("pip install 'gymnasium[atari,accept-rom-license]'")
