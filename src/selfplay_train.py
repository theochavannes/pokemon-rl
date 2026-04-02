"""
Phase C — Self-play training.

The agent trains against frozen snapshots of itself.
Every OPPONENT_UPDATE_FREQ steps the frozen opponent is updated to the latest best_model,
so the agent always faces a progressively stronger version of itself.

Usage (Showdown server must be running + phase B must be complete):
    C:/Users/theoc/miniconda3/envs/pokemon_rl/python.exe src/selfplay_train.py

The script looks for phase_B_final.zip (or best_model.zip as fallback) to seed both
the learning agent and the initial frozen opponent.
"""

import atexit
import ctypes
import os
import sys
from functools import partial
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(_PROJECT_ROOT)
sys.path.insert(0, str(_PROJECT_ROOT))

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from src.callbacks import WinRateCallback
from src.env.gen1_env import make_env

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BATTLE_FORMAT = "gen1randombattle"
TOTAL_TIMESTEPS = 500_000
MODEL_DIR = "models"
LOG_DIR = "logs"
REPLAY_DIR = "replays/selfplay"

N_ENVS = 1

# Swap the frozen opponent every N steps to keep training challenging.
# Too frequent = unstable (chasing a moving target); too rare = overfits to old self.
OPPONENT_UPDATE_FREQ = 50_000

PPO_KWARGS = dict(
    policy="MlpPolicy",
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    learning_rate=3e-4,
    verbose=0,
    tensorboard_log=LOG_DIR,
)

# Frozen opponent model path — this file is overwritten every OPPONENT_UPDATE_FREQ steps
_FROZEN_PATH = str(Path(MODEL_DIR) / "selfplay_frozen_opponent")


def _prevent_sleep() -> None:
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    atexit.register(
        lambda: ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    )
    print("Sleep prevention: ON")


class OpponentSwapCallback(BaseCallback):
    """
    Every OPPONENT_UPDATE_FREQ steps:
    1. Saves current best_model as the new frozen opponent checkpoint.
    2. Calls swap_model() on all FrozenPolicyPlayer instances in the env.
    """

    def __init__(self, env: DummyVecEnv, update_freq: int, verbose: int = 1):
        super().__init__(verbose)
        self.env = env
        self.update_freq = update_freq
        self._last_swap = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_swap >= self.update_freq:
            self._last_swap = self.num_timesteps
            self._swap_opponent()
        return True

    def _swap_opponent(self) -> None:
        # Save current model as the new frozen snapshot
        self.model.save(_FROZEN_PATH)

        # Reach into the DummyVecEnv to hot-swap each opponent's model
        # DummyVecEnv stores envs in self.envs (list of Monitor-wrapped envs)
        swapped = 0
        for env in self.env.envs:
            # Unwrap Monitor → SB3Wrapper → SingleAgentWrapper to reach the opponent
            try:
                opponent = env.env.env.opponent  # Monitor.env.SB3Wrapper.env.SingleAgentWrapper.opponent
                if hasattr(opponent, "swap_model"):
                    opponent.swap_model(_FROZEN_PATH)
                    swapped += 1
            except AttributeError:
                pass

        if self.verbose and swapped > 0:
            print(f"  [SelfPlay] Opponent updated at step {self.num_timesteps:,} ({swapped} env(s))")


def _seed_model_path() -> str:
    """Find the best starting checkpoint for self-play seeding."""
    for candidate in ["phase_B_final", "best_model"]:
        p = Path(MODEL_DIR) / f"{candidate}.zip"
        if p.exists():
            return str(Path(MODEL_DIR) / candidate)
    raise FileNotFoundError(
        f"No starting model found in {MODEL_DIR}/. "
        "Run src/train.py (phase A + B curriculum) first."
    )


def main() -> None:
    _prevent_sleep()
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(REPLAY_DIR, exist_ok=True)

    seed_path = _seed_model_path()
    print(f"\nSelf-play seeded from: {seed_path}.zip")

    # Save seed as the initial frozen opponent
    seed_model = MaskablePPO.load(seed_path)
    seed_model.save(_FROZEN_PATH)
    del seed_model

    print(f"Frozen opponent initialised: {_FROZEN_PATH}.zip")
    print(f"Opponent will update every {OPPONENT_UPDATE_FREQ:,} steps\n")

    # Build env — opponent is FrozenPolicyPlayer reading from _FROZEN_PATH
    env_fns = [
        partial(
            make_env,
            env_index=i,
            battle_format=BATTLE_FORMAT,
            save_replays=REPLAY_DIR,
            opponent_type="policy",
            opponent_model_path=_FROZEN_PATH,
        )
        for i in range(N_ENVS)
    ]
    env = DummyVecEnv(env_fns)

    # Load the learning model from seed
    model = MaskablePPO.load(
        seed_path,
        env=env,
        **{k: v for k, v in PPO_KWARGS.items() if k != "verbose"},
    )
    model.verbose = 0

    # Callbacks
    win_rate_cb = WinRateCallback(
        window=100,
        eval_freq=10_000,
        save_path=MODEL_DIR,
        replay_dir=REPLAY_DIR,
        notable_dir="replays/notable",
        verbose=1,
        phase_label="Self",
    )
    swap_cb = OpponentSwapCallback(env=env, update_freq=OPPONENT_UPDATE_FREQ, verbose=1)
    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=MODEL_DIR,
        name_prefix="selfplay",
        verbose=0,
    )

    print(f"Training for {TOTAL_TIMESTEPS:,} steps vs frozen self...\n")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[win_rate_cb, swap_cb, checkpoint_cb],
        reset_num_timesteps=True,
    )

    final_path = os.path.join(MODEL_DIR, "selfplay_final")
    model.save(final_path)
    print(f"\nSelf-play complete. Final model: {final_path}.zip")


if __name__ == "__main__":
    main()
