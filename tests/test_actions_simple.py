# tests/test_actions_simple.py
import os, sys, random, numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from config import (
    make_env, discover_action_meanings,
    ACTION_LEFT, ACTION_RIGHT, ACTION_DOWN, ACTION_HARD_DROP
)

def extract_board(obs):
    if isinstance(obs, dict):
        b = obs.get("board") or obs.get("observation")
        if b is None:
            for v in obs.values():
                if isinstance(v, np.ndarray):
                    b = v; break
    else:
        b = obs
    b = np.asarray(b)
    if b.ndim == 3 and b.shape[-1] >= 1:
        b = b[..., 0]
    return (b > 0).astype(np.uint8)

def filled_columns(board_2d):
    return sorted(np.where(board_2d.sum(axis=0) > 0)[0].tolist())

def nudge_to_wall(env, mover, cycles):
    obs = None
    for _ in range(cycles):
        obs, _, term, trunc, _ = env.step(mover)
        if term or trunc:
            return obs, True
        obs, _, term, trunc, _ = env.step(ACTION_DOWN)
        if term or trunc:
            return obs, True
    return obs, False

def print_board(board_2d):
    H, W = board_2d.shape
    print("   " + "".join(str(i) for i in range(W)))
    print("   " + "-" * W)
    for r in range(H):
        print(f"{r:2d}|"+ "".join("█" if c else "." for c in board_2d[r]))

def test_horizontal_movement():
    print("\n TESTING HORIZONTAL MOVEMENT")
    print("=" * 70)

    np.random.seed(0); random.seed(0)
    env = make_env(render_mode="rgb_array", use_complete_vision=True, use_cnn=False)
    print(f"\n✅ Environment created: {env}")
    print(f"Action space: {env.action_space} (n={env.action_space.n})")
    print("   Using standard Tetris action mapping")

    discover_action_meanings(env)
    print("\n🎯 Action IDs:")
    print(f"  LEFT={ACTION_LEFT} RIGHT={ACTION_RIGHT} DOWN={ACTION_DOWN} HARD_DROP={ACTION_HARD_DROP}\n")

    BOARD_W = 10
    CYCLES = BOARD_W * 3

    # ---- LEFT to wall
    print("=" * 70); print("TEST 1: Move LEFT to wall then DROP"); print("=" * 70)
    obs, _ = env.reset(seed=1)
    obs, ended = nudge_to_wall(env, ACTION_LEFT, CYCLES)
    if not ended: obs, _, _, _, _ = env.step(ACTION_HARD_DROP)
    left_cols = filled_columns(extract_board(obs))
    print("Filled columns after LEFT:", left_cols)

    # ---- RIGHT to wall
    print("\n" + "=" * 70); print("TEST 2: Move RIGHT to wall then DROP"); print("=" * 70)
    obs, _ = env.reset(seed=2)
    obs, ended = nudge_to_wall(env, ACTION_RIGHT, CYCLES)
    if not ended: obs, _, _, _, _ = env.step(ACTION_HARD_DROP)
    right_cols = filled_columns(extract_board(obs))
    print("Filled columns after RIGHT:", right_cols)

    # ---- Alternate a few times
    print("\n" + "=" * 70); print("TEST 3: Alternate LEFT/RIGHT for 10 pieces"); print("=" * 70)
    obs, _ = env.reset(seed=3)
    for i in range(10):
        mover = ACTION_LEFT if i % 2 == 0 else ACTION_RIGHT
        obs, ended = nudge_to_wall(env, mover, cycles=3)
        if ended: break
        obs, _, term, trunc, _ = env.step(ACTION_HARD_DROP)
        if term or trunc: break
    final_board = extract_board(obs)
    final_cols = filled_columns(final_board)
    print(f"Final filled columns: {final_cols}")
    print(f"Number of columns used: {len(final_cols)}/10\n")
    print("Final Board State:"); print_board(final_board)

    print("\n" + "=" * 70); print("📊 VERDICT"); print("=" * 70)
    assert left_cols != right_cols, "LEFT and RIGHT produced identical columns."
    if left_cols:  assert min(left_cols) <= 2,  f"LEFT too centered:  {left_cols}"
    if right_cols: assert max(right_cols) >= 7, f"RIGHT too centered: {right_cols}"
    print("\n✅ ACTIONS LOOK GOOD")

if __name__ == "__main__":
    try:
        test_horizontal_movement()
    except AssertionError as e:
        print("\n❌ TEST FAILED:", e); sys.exit(1)
    except Exception as e:
        import traceback; print("\n❌ Crash:"); traceback.print_exc(); sys.exit(2)
    else:
        sys.exit(0)
