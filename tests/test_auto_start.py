import gymnasium as gym
import numpy as np
import ale_py
import time


def auto_play_tetris():
    """
    Automatically starts Tetris and lets you play with simplified controls.
    """
    print("Starting Tetris with auto-start...")

    try:
        # Create environment
        env = gym.make("ALE/Tetris-v5", render_mode="human")
        print("✅ Environment created!")

        # Reset environment
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        print("Starting game automatically...")

        # Auto-start the game by pressing FIRE a few times
        for _ in range(5):
            result = env.step(1)  # FIRE action
            env.render()
            time.sleep(0.1)

        print("\n🎮 GAME STARTED! 🎮")
        print("\nSimplified Controls:")
        print("  A or ← : Move LEFT")
        print("  D or → : Move RIGHT")
        print("  S or ↓ : Drop faster")
        print("  W or ↑ : Rotate (FIRE)")
        print("  Space  : Do nothing")
        print("  R      : Reset game")
        print("  Q      : Quit")
        print("-" * 40)

        done = False
        total_reward = 0
        step_count = 0

        # Try to use keyboard input if available
        try:
            import msvcrt  # Windows only
            use_keyboard = True
            print("Using direct keyboard input (no need to press Enter)")
        except ImportError:
            use_keyboard = False
            print("Using standard input (press Enter after each command)")

        while not done:
            env.render()

            if use_keyboard and msvcrt.kbhit():
                key = msvcrt.getch().decode('utf-8').lower()
            else:
                key = input("Action: ").strip().lower()

            # Map keys to actions
            action = 0  # Default NOOP

            if key in ['a', '4']:  # LEFT
                action = 3
            elif key in ['d', '6']:  # RIGHT
                action = 2
            elif key in ['s', '2', '5']:  # DOWN
                action = 4
            elif key in ['w', '8']:  # ROTATE (FIRE)
                action = 1
            elif key == ' ':  # SPACE - NOOP
                action = 0
            elif key == 'r':  # RESET
                print("Resetting game...")
                state = env.reset()
                if isinstance(state, tuple):
                    state = state[0]
                # Auto-start again
                for _ in range(5):
                    env.step(1)
                    env.render()
                    time.sleep(0.1)
                total_reward = 0
                step_count = 0
                continue
            elif key == 'q':  # QUIT
                print("Quitting...")
                break
            else:
                continue  # Invalid input, skip

            # Take action
            result = env.step(action)

            if len(result) == 4:
                state, reward, done, info = result
            else:
                state, reward, terminated, truncated, info = result
                done = terminated or truncated

            total_reward += reward
            step_count += 1

            # Show score updates
            if reward > 0:
                print(f"  +{reward} points! Total: {total_reward}")

        print(f"\nGame Over! Final Score: {total_reward}")
        print(f"Total Steps: {step_count}")

    except KeyboardInterrupt:
        print("\nInterrupted!")
    finally:
        env.close()
        print("Environment closed.")


def test_random_agent():
    """
    Watch a random agent play Tetris (useful for testing).
    """
    print("Running random agent test...")

    env = gym.make("ALE/Tetris-v5", render_mode="human")
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]

    # Start the game
    for _ in range(5):
        env.step(1)  # FIRE
        env.render()
        time.sleep(0.1)

    print("Random agent playing...")
    done = False
    total_reward = 0

    while not done:
        # Random action, but prefer moving pieces
        if np.random.random() < 0.8:
            action = np.random.choice([2, 3, 4])  # RIGHT, LEFT, DOWN
        else:
            action = env.action_space.sample()

        result = env.step(action)
        env.render()

        if len(result) == 4:
            state, reward, done, info = result
        else:
            state, reward, terminated, truncated, info = result
            done = terminated or truncated

        total_reward += reward
        time.sleep(0.05)  # Slow down for visibility

        if reward > 0:
            print(f"Score: {total_reward}")

    print(f"Final Score: {total_reward}")
    env.close()


if __name__ == "__main__":
    print("Tetris Auto-Start Test")
    print("=" * 40)
    print("1. Play manually")
    print("2. Watch random agent")

    choice = input("Choose (1 or 2): ").strip()

    if choice == "1":
        auto_play_tetris()
    elif choice == "2":
        test_random_agent()
    else:
        print("Invalid choice!")
