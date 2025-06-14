import gym_tetris                     # ← must import before making any tetris envs!
from gym_tetris.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace


def manual_play():
    """
    Launches the Tetris environment from gym-tetris and lets
    you play by typing the index of the action you want.
    """
    # Use gym_tetris.make, not gym.make("ALE/Tetris-v5")
    env = JoypadSpace(gym_tetris.make("TetrisA-v0"), SIMPLE_MOVEMENT)

    print("Available actions (index → button combo):")
    for i, combo in enumerate(SIMPLE_MOVEMENT):
        print(f"  {i}: {combo}")
    print("Type the action index each frame (or 'q' to quit).")

    state, done = env.reset(), False
    try:
        while not done:
            env.render()
            cmd = input(
                f"Action [0–{len(SIMPLE_MOVEMENT)-1}] or 'q': ").strip()
            if cmd.lower() == 'q':
                break
            # validate
            try:
                act = int(cmd)
                if not 0 <= act < len(SIMPLE_MOVEMENT):
                    raise ValueError
            except ValueError:
                print("  ❌ Invalid index; please try again.")
                continue

            state, reward, done, info = env.step(act)
            print(f"  ▶ Reward: {reward:.1f}   Done: {done}")
    finally:
        env.close()
        print("Environment closed. Goodbye!")


if __name__ == "__main__":
    manual_play()
