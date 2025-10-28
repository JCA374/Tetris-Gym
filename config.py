# config.py
"""Configuration for Tetris RL Training - uses tetris_gymnasium.envs + robust action discovery and auto-calibration."""

import numpy as np
import gymnasium as gym

# Importing this module registers the env IDs (e.g., 'tetris_gymnasium/Tetris')
import tetris_gymnasium.envs

# =========================
# Training / general config
# =========================
ENV_NAME = "tetris_gymnasium/Tetris"
LR = 0.0001
MAX_EPISODES = 10000
MODEL_DIR = "models/"
LOG_DIR = "logs/"

EPISODES = 10000
BATCH_SIZE = 64
GAMMA = 0.99
LEARNING_RATE = 0.0001
MEMORY_SIZE = 100000
MIN_MEMORY_SIZE = 1000
TARGET_UPDATE_FREQUENCY = 1000

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.9999

HIDDEN_UNITS = [512, 256, 128]

ENV_CONFIG = {
    "height": 20,
    "width": 10,
}

REWARD_SHAPING_MODE = "balanced"

LOG_INTERVAL = 10
SAVE_INTERVAL = 500
MILESTONE_INTERVAL = 1000

# =========================
# Action IDs (populated at runtime)
# =========================
ACTION_NOOP       = 0
ACTION_LEFT       = 1
ACTION_RIGHT      = 2
ACTION_DOWN       = 3
ACTION_ROTATE_CW  = 4
ACTION_ROTATE_CCW = 5
ACTION_HARD_DROP  = 6
ACTION_SWAP       = 7

ACTION_MEANINGS = None  # filled by discover_action_meanings()


# ==========================================================
# Vision wrapper (kept from your version, with small cleanup)
# ==========================================================
class CompleteVisionWrapper(gym.ObservationWrapper):
    """
    Convert dict observations to a clean (20, 10, 1) uint8 board.
    Handles common tetris_gymnasium layouts:
      - (24,18) with walls → crop to [2:22, 4:14]
      - (20,10)           → pass-through
      - Otherwise         → best-effort (top-left 20x10 if available)
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(20, 10, 1), dtype=np.uint8
        )
        print("🎯 CompleteVisionWrapper initialized → outputs (20,10,1) uint8")

    def observation(self, obs):
        if isinstance(obs, dict) and "board" in obs:
            board = obs["board"]
        else:
            board = np.asarray(obs)

        board = np.asarray(board)
        if board.shape == (24, 18):
            # Standard tetris-gymnasium frame with walls
            playable = board[2:22, 4:14]
        elif board.shape == (20, 10):
            playable = board
        elif board.ndim == 2 and board.shape[0] >= 20 and board.shape[1] >= 10:
            playable = board[:20, :10]
        elif board.ndim == 3 and board.shape[-1] >= 1:
            # If already CHW/HWC with 1+ channels, take first channel then best-effort crop
            playable = board[..., 0]
            if playable.shape[0] >= 20 and playable.shape[1] >= 10:
                playable = playable[:20, :10]
        else:
            # Unexpected shape → zero pad/crop
            h, w = (board.shape + (0, 0))[:2]
            tmp = np.zeros((20, 10), dtype=np.uint8)
            tmp[: min(20, h), : min(10, w)] = board[: min(20, h), : min(10, w)]
            playable = tmp

        playable = (playable > 0).astype(np.uint8)
        return np.expand_dims(playable, axis=-1)  # (20,10,1)


# ============================================
# Helpers for calibration (board + wall nudges)
# ============================================
def _extract_board_simple(obs):
    """Return (H,W) 0/1 board from dict/array, supports (H,W,1)."""
    if isinstance(obs, dict):
        b = obs.get("board") or obs.get("observation")
        if b is None:
            for v in obs.values():
                if isinstance(v, np.ndarray):
                    b = v
                    break
            if b is None:
                return np.zeros((20, 10), dtype=np.uint8)
    else:
        b = obs
    b = np.asarray(b)
    if b.ndim == 3 and b.shape[-1] >= 1:
        b = b[..., 0]
    return (b > 0).astype(np.uint8)


def _filled_cols(board_2d):
    return np.where(np.asarray(board_2d).sum(axis=0) > 0)[0]


def _nudge_to_wall(env, mover, down, cycles=30):
    """Interleave horizontal moves with DOWN so the piece slides while falling."""
    obs = None
    for _ in range(cycles):
        obs, _, term, trunc, _ = env.step(mover)
        if term or trunc:
            return obs, True
        obs, _, term, trunc, _ = env.step(down)
        if term or trunc:
            return obs, True
    return obs, False


# ======================================
# Action discovery and runtime calibration
# ======================================
def discover_action_meanings(env, verbose=True):
    """
    Populate ACTION_* globals from the env if possible; otherwise fallback to standard mapping.
    """
    global ACTION_NOOP, ACTION_LEFT, ACTION_RIGHT, ACTION_DOWN
    global ACTION_ROTATE_CW, ACTION_ROTATE_CCW, ACTION_HARD_DROP, ACTION_SWAP
    global ACTION_MEANINGS

    meanings = None

    for obj in (env, getattr(env, "unwrapped", None)):
        if obj is None:
            continue
        if hasattr(obj, "get_action_meanings"):
            try:
                meanings = obj.get_action_meanings()
                break
            except Exception:
                pass
        md = getattr(obj, "metadata", {})
        if isinstance(md, dict) and "action_meanings" in md:
            meanings = md["action_meanings"]
            break

    if not meanings:
        n = getattr(env.action_space, "n", 8) or 8
        meanings = ["NOOP", "LEFT", "RIGHT", "DOWN", "ROTATE_CW", "ROTATE_CCW", "HARD_DROP", "SWAP"][:n]
        if verbose:
            print("⚠️  Could not read action meanings from env; using standard mapping.")

    norm = [str(m).upper() for m in meanings]

    def idx(label, default):
        try:
            return norm.index(label)
        except ValueError:
            return default

    ACTION_NOOP       = idx("NOOP", 0)
    ACTION_LEFT       = idx("LEFT", 1)
    ACTION_RIGHT      = idx("RIGHT", 2)
    ACTION_DOWN       = idx("DOWN", 3)
    ACTION_ROTATE_CW  = idx("ROTATE_CW", 4)
    ACTION_ROTATE_CCW = idx("ROTATE_CCW", 5)
    ACTION_HARD_DROP  = idx("HARD_DROP", 6)
    ACTION_SWAP       = idx("SWAP", 7)

    ACTION_MEANINGS = norm

    if verbose:
        print("🎯 Action Mappings (discovered):")
        for i, name in enumerate(norm):
            print(f"   {i}: {name}")
        print("\nUsing actions from config.py:")
        print(f"  LEFT = {ACTION_LEFT}")
        print(f"  RIGHT = {ACTION_RIGHT}")
        print(f"  DOWN = {ACTION_DOWN}")
        print(f"  HARD_DROP = {ACTION_HARD_DROP}")


def auto_calibrate_left_right(env, verbose=True):
    """
    Slide to each wall once; if mirrored/reversed, swap ACTION_LEFT and ACTION_RIGHT.
    """
    global ACTION_LEFT, ACTION_RIGHT, ACTION_DOWN, ACTION_HARD_DROP

    # Trial to the LEFT
    obs, _ = env.reset(seed=123)
    obs, ended = _nudge_to_wall(env, ACTION_LEFT, ACTION_DOWN, cycles=30)
    if not ended:
        obs, _, _, _, _ = env.step(ACTION_HARD_DROP)
    cols_left = _filled_cols(_extract_board_simple(obs))

    # Trial to the RIGHT
    obs, _ = env.reset(seed=456)
    obs, ended = _nudge_to_wall(env, ACTION_RIGHT, ACTION_DOWN, cycles=30)
    if not ended:
        obs, _, _, _, _ = env.step(ACTION_HARD_DROP)
    cols_right = _filled_cols(_extract_board_simple(obs))

    if len(cols_left) == 0 or len(cols_right) == 0:
        if verbose:
            print("⚠️  auto_calibrate_left_right: no filled columns detected; leaving mapping as-is.")
        return

    min_left  = int(np.min(cols_left))
    max_right = int(np.max(cols_right))

    # Expect (min_left <= 2) and (max_right >= 7) on a 10-wide board
    looks_reversed = (min_left >= 5) and (max_right <= 4)

    if looks_reversed:
        if verbose:
            print(f"🔁 Detected LEFT/RIGHT reversed (min_left={min_left}, max_right={max_right}). Swapping...")
        ACTION_LEFT, ACTION_RIGHT = ACTION_RIGHT, ACTION_LEFT
    else:
        if verbose:
            print(f"✅ LEFT/RIGHT look correct (min_left={min_left}, max_right={max_right}).")


# ---- Compatibility shim for older train.py (expects a dict) ----
def get_action_meanings(as_dict: bool = True):
    """
    Return action meanings. By default returns a dict {action_id: meaning}
    to match older train.py usage: for action_id, meaning in action_meanings.items(): ...
    Set as_dict=False to get the raw list.
    """
    default_list = [
        "NOOP", "LEFT", "RIGHT", "DOWN", "ROTATE_CW", "ROTATE_CCW", "HARD_DROP", "SWAP"
    ]
    lst = ACTION_MEANINGS or default_list
    if as_dict:
        return {i: name for i, name in enumerate(lst)}
    return lst

def get_action_ids():
    """
    Convenience dict of action name -> id.
    (Optional; handy if train.py wants explicit IDs.)
    """
    return {
        "NOOP": ACTION_NOOP,
        "LEFT": ACTION_LEFT,
        "RIGHT": ACTION_RIGHT,
        "DOWN": ACTION_DOWN,
        "ROTATE_CW": ACTION_ROTATE_CW,
        "ROTATE_CCW": ACTION_ROTATE_CCW,
        "HARD_DROP": ACTION_HARD_DROP,
        "SWAP": ACTION_SWAP,
    }




# =====================
# Env factory (public)
# =====================
def make_env(render_mode="rgb_array", use_complete_vision=True, use_cnn=False):
    """
    Create env, wrap with CompleteVisionWrapper (20x10x1), discover actions,
    and auto-calibrate LEFT/RIGHT.
    """
    env = gym.make(ENV_NAME, render_mode=render_mode, **ENV_CONFIG)
    print(f"✅ Environment created: {getattr(env.spec, 'id', ENV_NAME)}")
    print(f"   Action space: {env.action_space}")

    if use_complete_vision:
        env = CompleteVisionWrapper(env)
        print(f"   Observation space: {env.observation_space}")

    discover_action_meanings(env, verbose=True)
    auto_calibrate_left_right(env, verbose=True)

    return env


# ======================
# Quick manual self-test
# ======================
def test_environment():
    """Basic smoke test for the env + wrapper."""
    print("\n🧪 Testing environment...")

    env = make_env(render_mode="rgb_array", use_complete_vision=True)
    obs, info = env.reset(seed=42)
    print("✅ Reset OK")
    print(f"   Observation shape: {obs.shape} | dtype: {obs.dtype} | range: [{obs.min()}, {obs.max()}]")

    assert obs.shape == (20, 10, 1), f"Expected (20, 10, 1), got {obs.shape}"

    total_reward = 0.0
    for _ in range(10):
        a = env.action_space.sample()
        obs, r, term, trunc, _ = env.step(a)
        total_reward += r
        assert obs.shape == (20, 10, 1)
        if term or trunc:
            obs, info = env.reset()

    print(f"✅ Steps OK — Total reward: {total_reward:.2f}")
    env.close()
    return True


# Public exports
__all__ = [
    # env
    "ENV_NAME", "ENV_CONFIG", "make_env",
    # actions
    "ACTION_NOOP", "ACTION_LEFT", "ACTION_RIGHT", "ACTION_DOWN",
    "ACTION_ROTATE_CW", "ACTION_ROTATE_CCW", "ACTION_HARD_DROP", "ACTION_SWAP",
    "ACTION_MEANINGS",
    # discovery/calibration
    "discover_action_meanings", "auto_calibrate_left_right",
    # wrapper
    "CompleteVisionWrapper",
    "get_action_meanings", "get_action_ids",

]
