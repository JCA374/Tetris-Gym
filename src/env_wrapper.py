# src/env_wrapper.py
import os
import numpy as np
import gymnasium as gym
import tetris_gymnasium  # registers the env


def _binarize(a: np.ndarray) -> np.ndarray:
    """Return binary board view (0/1) from any dtype."""
    a = np.asarray(a)
    # Treat >0 as filled; adjust here if your env uses a different encoding.
    return (a > 0).astype(np.uint8)


def _strip_walls(a: np.ndarray, row_thr: float = 0.98, col_thr: float = 0.98) -> np.ndarray:
    """
    Remove near-fully-filled border rows/cols (e.g., frame/borders).
    Keeps interior playfield without re-centering.
    """
    filled = _binarize(a)
    H, W = filled.shape

    row_ratio = filled.mean(axis=1) if H else np.array([])
    col_ratio = filled.mean(axis=0) if W else np.array([])

    keep_rows = [i for i in range(H) if row_ratio[i] <= row_thr]
    keep_cols = [j for j in range(W) if col_ratio[j] <= col_thr]

    # If stripping nuked everything (rare), fall back to original
    if not keep_rows:
        keep_rows = list(range(H))
    if not keep_cols:
        keep_cols = list(range(W))

    return a[np.ix_(keep_rows, keep_cols)]


def _crop_playfield_20x10(arr2d: np.ndarray, debug=False) -> np.ndarray:
    """
    Deterministically return a (20,10) playfield WITHOUT center bias.

    Rules:
      1) If already (20,10) → return as-is.
      2) Strip border-like rows/cols (>=98% filled).
      3) If larger than 20×10, slice TOP/LEFT-aligned (no center choice).
      4) If smaller, top-left pad with zeros.
    """
    a = np.asarray(arr2d)
    if a.ndim != 2:
        raise ValueError(f"Expected 2D board, got shape {a.shape}")

    H, W = a.shape
    if debug:
        print(f"[Wrapper] raw board shape: {H}x{W}")

    # (1) Short-circuit for exact shape
    if (H, W) == (20, 10):
        if debug:
            print("[Wrapper] exact 20x10 board, returning as-is.")
        return a

    # (2) Strip border-like rows/cols
    after = _strip_walls(a, row_thr=0.98, col_thr=0.98)
    h, w = after.shape
    if debug:
        print(f"[Wrapper] after strip walls: {h}x{w}")

    # (3) Enforce width then height without center-bias
    # Width → 10 (LEFT-aligned)
    if w > 10:
        if debug:
            print(f"[Wrapper] width {w} > 10 → using LEFT-aligned slice 0:10")
        after = after[:, 0:10]
        h, w = after.shape

    # Height → 20 (TOP-aligned)
    if h > 20:
        if debug:
            print(f"[Wrapper] height {h} > 20 → using TOP-aligned slice 0:20")
        after = after[0:20, :]
        h, w = after.shape

    # (4) If smaller, pad top-left
    if h < 20 or w < 10:
        if debug:
            print(f"[Wrapper] smaller than 20x10 → padding to 20x10 from top-left (have {h}x{w})")
        out = np.zeros((20, 10), dtype=after.dtype)
        out[:min(h, 20), :min(w, 10)] = after[:min(h, 20), :min(w, 10)]
        after = out

    if debug:
        hh, ww = after.shape
        print(f"[Wrapper] final cropped shape: {hh}x{ww}")
    return after


class CompleteVisionWrapper(gym.ObservationWrapper):
    """
    Converts env observation (dict/array) to a clean (20,10,1) uint8 board.

    IMPORTANT:
    - Never re-center an already (20,10) board.
    - Never center-crop; prefer left/top-aligned slicing.
    """
    _debug_once_budget = 10  # limit how many frames print (with env var)

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(20, 10, 1), dtype=np.uint8
        )
        print("🎯 CompleteVisionWrapper active → outputs (20,10,1) uint8")
        self._debug_enabled = os.getenv("TETRIS_WRAPPER_DEBUG", "0") == "1"

    def _extract_2d(self, obs):
        """Best-effort extraction of a 2D board array from various obs layouts."""
        if isinstance(obs, dict):
            board = obs.get("board")
            if board is None:
                board = obs.get("observation")
            if board is None:
                # Fall back to first ndarray value
                for v in obs.values():
                    if isinstance(v, np.ndarray):
                        board = v
                        break
            if board is None:
                board = np.zeros((20, 10), dtype=np.uint8)
        else:
            board = obs

        board = np.asarray(board)
        if board.ndim == 3:
            board2d = board[..., 0]
        elif board.ndim == 2:
            board2d = board
        else:
            # Unknown format → safe fallback
            board2d = np.zeros((20, 10), dtype=np.uint8)
        return board2d

    def observation(self, obs):
        board2d = self._extract_2d(obs)

        debug = False
        if self._debug_enabled and self._debug_once_budget > 0:
            debug = True
            self._debug_once_budget -= 1

        board2d = _crop_playfield_20x10(board2d, debug=debug)
        board2d = _binarize(board2d).astype(np.uint8)
        return np.expand_dims(board2d, axis=-1)
