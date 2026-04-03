"""
Self-play training — agent trains against frozen snapshots of itself.

Every OPPONENT_UPDATE_FREQ steps the frozen opponent is updated to the latest best_model.
Outputs go to the same run as the curriculum (runs/run_NNN/) or a new run if standalone.

Usage:
    python src/selfplay_train.py                  # auto-detect latest curriculum run
    python src/selfplay_train.py --run run_001    # continue a specific run
    python src/selfplay_train.py --new-run        # fresh self-play run from best_model
"""

import argparse
import atexit
import ctypes
import os
import sys
from functools import partial
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(_PROJECT_ROOT)
sys.path.insert(0, str(_PROJECT_ROOT))

import logging

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from src.callbacks import WinRateCallback
from src.env.gen1_env import make_env
from src.logging_config import setup_logging
from src.obs_transfer import is_compatible, load_with_expanded_obs
from src.run_manager import RUNS_DIR, RunManager

log = logging.getLogger("pokemon_rl.selfplay")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BATTLE_FORMAT = "gen1randombattle"
TOTAL_TIMESTEPS = 500_000
N_ENVS = 1
OPPONENT_UPDATE_FREQ = 50_000

PPO_KWARGS = dict(
    policy="MlpPolicy",
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,  # Entropy bonus — prevents policy collapse to a single action
    learning_rate=3e-4,
    verbose=0,
)


def _prevent_sleep() -> None:
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    atexit.register(lambda: ctypes.windll.kernel32.SetThreadExecutionState(0x80000000))
    print("Sleep prevention: ON")


class OpponentSwapCallback(BaseCallback):
    def __init__(self, env: DummyVecEnv, frozen_path: str, update_freq: int, verbose: int = 1):
        super().__init__(verbose)
        self.env = env
        self.frozen_path = frozen_path
        self.update_freq = update_freq
        self._last_swap = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_swap >= self.update_freq:
            self._last_swap = self.num_timesteps
            self._swap_opponent()
        return True

    def _swap_opponent(self) -> None:
        self.model.save(self.frozen_path)
        swapped = 0
        for env in self.env.envs:
            opp = getattr(env, "_opponent", None)
            if opp and hasattr(opp, "swap_model"):
                opp.swap_model(self.frozen_path)
                swapped += 1
        if self.verbose and swapped > 0:
            print(f"  [SelfPlay] Opponent updated at step {self.num_timesteps:,}")


def _find_seed(run: RunManager) -> str:
    """Find the best model to seed self-play from (within this run or globally)."""
    # Check this run first: phase_C > phase_B > phase_A > best_model
    for candidate in ["phase_C_final", "phase_B_final", "phase_A_final", "best_model"]:
        p = Path(run.models_dir) / f"{candidate}.zip"
        if p.exists():
            return str(p.with_suffix(""))
    # Fallback: search all runs (newest first)
    for run_dir in sorted(RUNS_DIR.glob("run_*"), reverse=True):
        for candidate in ["phase_C_final", "phase_B_final", "phase_A_final", "best_model"]:
            p = run_dir / "models" / f"{candidate}.zip"
            if p.exists():
                return str(p.with_suffix(""))
    raise FileNotFoundError("No seed model found. Run src/train.py first.")


def main(run_id: str | None = None, new_run: bool = False) -> None:
    _prevent_sleep()

    if run_id:
        # Use a specific existing run
        run_dir = RUNS_DIR / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Run {run_id} not found in {RUNS_DIR}")
        run = RunManager(
            run_type="selfplay",
            config={"total_timesteps": TOTAL_TIMESTEPS, "opponent_update_freq": OPPONENT_UPDATE_FREQ, **PPO_KWARGS},
            new_run=False,
        )
        run.run_dir = run_dir
    else:
        run = RunManager(
            run_type="selfplay",
            config={"total_timesteps": TOTAL_TIMESTEPS, "opponent_update_freq": OPPONENT_UPDATE_FREQ, **PPO_KWARGS},
            new_run=new_run,
        )

    os.makedirs(run.models_dir, exist_ok=True)
    os.makedirs(run.logs_dir, exist_ok=True)

    setup_logging(log_dir=run.logs_dir)
    log.info("Starting %s (self-play training)", run.run_id)

    seed_path = _find_seed(run)
    frozen_path = str(Path(run.models_dir) / "selfplay_frozen_opponent")

    print(f"\n{run.run_id} — Self-play training")
    print(f"  Seed model:      {seed_path}.zip")
    print(f"  Frozen opponent: {frozen_path}.zip (updated every {OPPONENT_UPDATE_FREQ:,} steps)")

    # Save seed as initial frozen opponent
    seed_model = MaskablePPO.load(seed_path)
    seed_model.save(frozen_path)
    del seed_model

    replay_dir = run.replays_dir("selfplay")

    env_fns = [
        partial(
            make_env,
            env_index=i,
            battle_format=BATTLE_FORMAT,
            save_replays=replay_dir,
            opponent_type="policy",
            opponent_model_path=frozen_path,
        )
        for i in range(N_ENVS)
    ]
    env = DummyVecEnv(env_fns)

    current_obs_dim = env.observation_space.shape[0]
    if is_compatible(seed_path, current_obs_dim):
        model = MaskablePPO.load(
            seed_path,
            env=env,
            **{k: v for k, v in PPO_KWARGS.items() if k != "verbose"},
        )
    else:
        print(f"  Obs space changed — transferring weights from {seed_path}")
        model = load_with_expanded_obs(
            old_path=seed_path,
            new_obs_dim=current_obs_dim,
            env=env,
            ppo_kwargs={**PPO_KWARGS, "tensorboard_log": run.logs_dir},
        )
    model.tensorboard_log = run.logs_dir
    model.verbose = 0

    # Collect opponent refs for shaping decay
    opponents = []
    if hasattr(env, "envs"):
        for e in env.envs:
            if hasattr(e, "_opponent"):
                opponents.append(e._opponent)

    win_rate_cb = WinRateCallback(
        window=100,
        eval_freq=10_000,
        save_path=run.models_dir,
        replay_dir=replay_dir,
        notable_dir="replays/notable",
        verbose=1,
        phase_label="Self",
        training_log_path=run.training_log,
        run_id=run.run_id,
        opponents=opponents,
        run_manager=run,
        phase_name="selfplay",
        selfplay_path=frozen_path,
        env=env,
    )
    swap_cb = OpponentSwapCallback(env=env, frozen_path=frozen_path, update_freq=OPPONENT_UPDATE_FREQ)
    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=run.models_dir,
        name_prefix="selfplay",
        verbose=0,
    )

    print(f"\n  Training for {TOTAL_TIMESTEPS:,} steps...\n")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[win_rate_cb, swap_cb, checkpoint_cb],
        reset_num_timesteps=True,
    )

    final_path = str(Path(run.models_dir) / "selfplay_final")
    model.save(final_path)
    run.mark_complete()
    print(f"\nSelf-play complete → {final_path}.zip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None, help="Specific run ID to continue (e.g. run_001)")
    parser.add_argument("--new-run", action="store_true", help="Force a new run")
    args = parser.parse_args()
    main(run_id=args.run, new_run=args.new_run)
