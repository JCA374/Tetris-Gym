import gymnasium as gym
import numpy as np
import ale_py
import time


def ensure_game_started(env, max_attempts=30):
    """
    Properly starts Atari Tetris by pressing FIRE multiple times
    and checking if pieces are actually falling.
    """
    print("Starting Tetris game...")

    # First, press FIRE several times to get through any menus
    for i in range(10):
        obs, reward, done, *_ = env.step(1)  # FIRE
        env.render()
        time.sleep(0.1)
        if reward > 0:  # Game might have started
            print(f"Game possibly started after {i+1} FIRE presses")
            break

    # Now alternate between FIRE and NOOP to ensure game starts
    print("Ensuring game is active...")
    for i in range(10):
        if i % 2 == 0:
            obs, reward, done, *_ = env.step(1)  # FIRE
        else:
            obs, reward, done, *_ = env.step(0)  # NOOP
        env.render()
        time.sleep(0.1)

    # Take a few NOOP actions to let the game settle
    for _ in range(5):
        env.step(0)
        env.render()
        time.sleep(0.1)

    print("Game should be started now!")
    return


def play_tetris_debug():
    """
    Play Tetris with debug information to see what's happening.
    """
    print("Starting Tetris with debug mode...")

    env = gym.make("ALE/Tetris-v5", render_mode="human")

    # Reset environment
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    # Ensure game starts properly
    ensure_game_started(env)

    print("\n🎮 CONTROLS 🎮")
    print("1 = FIRE/Rotate")
    print("2 = RIGHT")
    print("3 = LEFT")
    print("4 = DOWN (drop faster)")
    print("0 = NOOP (do nothing - let piece fall)")
    print("r = Reset game")
    print("q = Quit")
    print("\nTIP: If pieces aren't falling, try pressing 1 (FIRE) a few more times")
    print("-" * 50)

    done = False
    total_reward = 0
    step_count = 0
    last_reward_step = 0

    while not done:
        env.render()

        # Get action
        action_str = input(
            f"Step {step_count} | Score: {total_reward} | Action: ").strip().lower()

        if action_str == 'q':
            break
        elif action_str == 'r':
            print("Resetting...")
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            ensure_game_started(env)
            total_reward = 0
            step_count = 0
            continue

        # Convert to action
        try:
            if action_str == '':
                action = 0  # NOOP if just pressing enter
            else:
                action = int(action_str)
                if action not in range(5):
                    print(f"Invalid action! Use 0-4")
                    continue
        except ValueError:
            print(f"Invalid input! Use numbers 0-4")
            continue

        # Take action
        result = env.step(action)

        if len(result) == 4:
            obs, reward, done, info = result
        else:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated

        step_count += 1

        # Debug information
        if reward != 0:
            total_reward += reward
            print(f"  → Reward: +{reward} | Total: {total_reward}")
            last_reward_step = step_count

        # Check if game seems stuck (no rewards for many steps)
        if step_count - last_reward_step > 100 and total_reward == 0:
            print("  ⚠️  No rewards for 100 steps. Game might not be started properly.")
            print("     Try pressing 1 (FIRE) multiple times!")

    print(f"\nGame Over! Final Score: {total_reward}")
    env.close()


def test_auto_play():
    """
    Automatically play with a simple strategy to verify the game works.
    """
    print("Testing automatic play...")

    env = gym.make("ALE/Tetris-v5", render_mode="human")
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    # Start game
    ensure_game_started(env)

    print("Auto-playing with simple strategy...")
    print("(Pieces should be falling and moving)")

    done = False
    total_reward = 0
    step = 0

    while not done and step < 1000:  # Play for max 1000 steps
        # Simple strategy: mostly move left/right and drop
        if step % 10 < 3:
            action = 3  # LEFT
        elif step % 10 < 6:
            action = 2  # RIGHT
        elif step % 10 < 8:
            action = 4  # DOWN
        else:
            action = 0  # NOOP

        result = env.step(action)
        env.render()

        if len(result) == 4:
            obs, reward, done, info = result
        else:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated

        if reward > 0:
            total_reward += reward
            print(f"Step {step}: Score = {total_reward}")

        step += 1
        time.sleep(0.05)  # Slow down for visibility

    print(f"\nFinal Score: {total_reward}")
    print(f"Total Steps: {step}")

    if total_reward == 0:
        print("\n⚠️  No score achieved. The game might not have started properly.")
        print("    This can happen with Atari Tetris. Try running again.")

    env.close()


if __name__ == "__main__":
    print("Atari Tetris Starter")
    print("=" * 50)
    print("1. Play manually with debug info")
    print("2. Watch automatic play test")
    print("3. Quick test (just start and check)")

    choice = input("\nChoose option (1-3): ").strip()

    if choice == "1":
        play_tetris_debug()
    elif choice == "2":
        test_auto_play()
    elif choice == "3":
        # Quick test
        env = gym.make("ALE/Tetris-v5", render_mode="human")
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

        ensure_game_started(env)

        print("\nTaking 50 NOOP actions to see if pieces fall...")
        for i in range(50):
            env.step(0)  # NOOP
            env.render()
            time.sleep(0.1)

        env.close()
        print("Test complete!")
    else:
        print("Invalid choice!")
