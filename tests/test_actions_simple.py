#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tests/test_actions_simple.py

Purpose:
- Verify that LEFT and RIGHT actually move pieces to different columns.
- Make the test deterministic (seeds).
- Avoid false positives by nudging more than board width before dropping.
- Assert on expected behavior (not just print).

Run:
    python tests/test_actions_simple.py
"""

import os
import sys
import random
import numpy as np

# Ensure project root is on path when running from tests/
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from config import (  # uses your config.py where env is created + action mapping is discovered
    make_env,
    discover_action_meanings,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_HARD_DROP,
)

# -----------------------------
# Helpers
# -----------------------------

def extract_board(obs):
    """
    Extract a (H,W) uint8 board from an observation.
    Works with:
      - dict observations (board/observation/first ndarray value)
      - array observations, possibly (H,W,1)
    """
    if isinstance(obs, dict):
        board = obs.get("board", None)
        if board is None:
            board = obs.get("observation", None)
        if board is None:
            # fallback: first ndarray-like value
            for v in obs.values():
                if isinstance(v, np.ndarray):
                    board = v
                    break
            if board is None:
                raise ValueError("No ndarray-like value found in dict observation")
    else:
        board = obs

    board = np.asarray(board)
    if board.ndim == 3 and board.shape[-1] >= 1:
        board = board[..., 0]  # (H,W,1) -> (H,W)

    # Ensure uint8 0/1 board; some envs may return 0..255
    board = (board > 0).astype(np.uint8)
    return board


def get_filled_columns(board_2d):
    """Return sorted list of column indices that contain any filled cell (>0)."""
    b = np.asarray(board_2d)
    return sorted(np.where(b.sum(axis=0) > 0)[0].tolist())


def print_board(board_2d):
    """Pretty-print a small 0/1 board."""
    H, W = board_2d.shape
    header = "".join(str(i) for i in range(W))
    print(f"   {header}")
    print("   " + "-" * W)
    for r in range(H):
        row = "".join("█" if c else "." for c in board_2d[r])
        print(f"{r:2d}|{row}")


# -----------------------------
# Tests
# -----------------------------

def test_horizontal_movement():
    print("\n TESTING HORIZONTAL MOVEMENT")
    print("=" * 70)

    # Determinism
    np.random.seed(0)
    random.seed(0)

    # Create env through your config (which also discovers action meanings)
    env = make_env(render_mode="rgb_array", use_complete_vision=True, use_cnn=False)
    print(f"\n✅ Environment created: {env}")
    print(f"Action space: {env.action_space} (n={env.action_space.n})")
    print("   Using standard Tetris action mapping")

    # Ensure mapping is up-to-date
    discover_action_meanings(env)

    print("\n🎯 Action Mappings:")
    print("   0: NOOP")
    print("   1: LEFT")
    print("   2: RIGHT")
    print("   3: DOWN")
    print("   4: ROTATE_CW")
    print("   5: ROTATE_CCW")
    print("   6: HARD_DROP")
    print("   7: SWAP")
    print("\nUsing actions from config.py:")
    print(f"  LEFT = {ACTION_LEFT}")
    print(f"  RIGHT = {ACTION_RIGHT}")
    print(f"  HARD_DROP = {ACTION_HARD_DROP}\n")

    # -----------------------------
    # TEST 1: Move LEFT to wall (fixed nudges) then DROP
    # -----------------------------
    print("=" * 70)
    print("TEST 1: Move LEFT to wall then DROP")
    print("=" * 70)

    obs, _ = env.reset(seed=1)
    for _ in range(12):  # > board width → guaranteed to reach wall
        obs, _, term, trunc, _ = env.step(ACTION_LEFT)
        if term or trunc:
            break
    obs, _, _, _, _ = env.step(ACTION_HARD_DROP)
    board_after_left = extract_board(obs)
    cols_left = get_filled_columns(board_after_left)
    print(f"Filled columns after LEFT: {cols_left}")

    # -----------------------------
    # TEST 2: Move RIGHT to wall (fixed nudges) then DROP
    # -----------------------------
    print("\n" + "=" * 70)
    print("TEST 2: Move RIGHT to wall then DROP")
    print("=" * 70)

    obs, _ = env.reset(seed=2)
    for _ in range(12):  # > board width → guaranteed to reach wall
        obs, _, term, trunc, _ = env.step(ACTION_RIGHT)
        if term or trunc:
            break
    obs, _, _, _, _ = env.step(ACTION_HARD_DROP)
    board_after_right = extract_board(obs)
    cols_right = get_filled_columns(board_after_right)
    print(f"Filled columns after RIGHT: {cols_right}")

    # -----------------------------
    # TEST 3: Alternate LEFT/RIGHT for 10 pieces
    # -----------------------------
    print("\n" + "=" * 70)
    print("TEST 3: Alternate LEFT/RIGHT for 10 pieces")
    print("=" * 70)

    obs, _ = env.reset(seed=3)
    for i in range(10):
        mover = ACTION_LEFT if i % 2 == 0 else ACTION_RIGHT
        # Nudge a few times for variety (not necessarily to wall)
        for _ in range(3):
            obs, _, term, trunc, _ = env.step(mover)
            if term or trunc:
                break
        obs, _, term, trunc, _ = env.step(ACTION_HARD_DROP)
        if term or trunc:
            print(f"  Episode ended after piece {i+1}")
            break

    final_board = extract_board(obs)
    final_cols = get_filled_columns(final_board)
    print(f"Final filled columns: {final_cols}")
    print(f"Number of columns used: {len(final_cols)}/10\n")

    print("Final Board State:")
    print_board(final_board)

    # -----------------------------
    # VERDICT & ASSERTS
    # -----------------------------
    print("\n" + "=" * 70)
    print("📊 VERDICT")
    print("=" * 70)

    left_right_different = cols_left != cols_right
    print(f"{'✅' if left_right_different else '❌'} LEFT and RIGHT create different patterns")

    # Minimal coverage target: we at least want a few columns used after alternation
    some_spread = len(final_cols) >= 3
    print(f"{'✅' if some_spread else '⚠️ '} Some columns used: {len(final_cols)}/10")

    # Hard assertions (fail fast on regressions)
    assert left_right_different, "LEFT and RIGHT produced the same filled columns (unexpected)."
    if cols_left:
        assert min(cols_left) <= 1, f"LEFT result is too centered: {cols_left}"
    if cols_right:
        assert max(cols_right) >= 8, f"RIGHT result is too centered: {cols_right}"

    print("\n✅ ACTIONS ARE WORKING CORRECTLY")
    print("   → If training still sticks to few columns, focus on exploration & reward shaping.")


# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":
    try:
        test_horizontal_movement()
    except AssertionError as e:
        print("\n❌ TEST FAILED:", e)
        sys.exit(1)
    except Exception as e:
        print("\n❌ Test crashed with error:")
        import traceback
        traceback.print_exc()
        sys.exit(2)
    else:
        sys.exit(0)
