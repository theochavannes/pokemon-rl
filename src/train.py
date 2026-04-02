"""
Curriculum training: RandomPlayer → MaxDamagePlayer.

Each phase auto-advances when the win-rate target is hit or the step cap is reached.
All outputs (models, logs, replays) go to runs/run_NNN/ — each run is isolated.
A new run is created automatically unless a resumable one exists.

Usage (Showdown server must be running first):
    python src/train.py           # auto-resume latest or create new run
    python src/train.py --new-run # force a new run

TensorBoard:
    tensorboard --logdir runs/run_NNN/logs/
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

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from src.callbacks import WinRateCallback
from src.env.gen1_env import make_env
from src.obs_transfer import load_with_expanded_obs, is_compatible
from src.run_manager import RunManager

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BATTLE_FORMAT = "gen1randombattle"
N_ENVS = 1

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
)

CURRICULUM = [
    dict(
        name="A",
        opponent_type="random",
        phase_label="Random",
        target_wr=0.75,
        max_steps=150_000,
    ),
    dict(
        name="B",
        opponent_type="maxdamage",
        phase_label="MaxDamage",
        target_wr=0.65,
        max_steps=200_000,
    ),
]


def _prevent_sleep() -> None:
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    atexit.register(
        lambda: ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    )
    print("Sleep prevention: ON")


def main(new_run: bool = False) -> None:
    _prevent_sleep()

    run = RunManager(
        run_type="curriculum",
        config={**PPO_KWARGS, "curriculum": CURRICULUM, "n_envs": N_ENVS},
        new_run=new_run,
    )

    os.makedirs(run.models_dir, exist_ok=True)
    os.makedirs(run.logs_dir, exist_ok=True)

    resume_path = run.latest_checkpoint()
    model = None

    for phase in CURRICULUM:
        replay_dir = run.replays_dir(f"phase_{phase['name'].lower()}")

        print(f"\n{'='*60}")
        print(f"  {run.run_id} — Phase {phase['name']}: vs {phase['phase_label']}")
        print(f"  Target: {phase['target_wr']*100:.0f}%  |  Cap: {phase['max_steps']:,} steps")
        print(f"  Replays → {replay_dir}")
        print(f"{'='*60}\n")

        env_fns = [
            partial(
                make_env,
                env_index=i,
                battle_format=BATTLE_FORMAT,
                save_replays=replay_dir,
                opponent_type=phase["opponent_type"],
            )
            for i in range(N_ENVS)
        ]
        env = DummyVecEnv(env_fns)

        if model is None:
            if resume_path:
                if is_compatible(resume_path, env.observation_space.shape[0]):
                    print(f"  Resuming from {resume_path}")
                    model = MaskablePPO.load(
                        resume_path, env=env,
                        **{k: v for k, v in PPO_KWARGS.items() if k != "verbose"},
                    )
                else:
                    print(f"  Obs space changed — transferring weights from {resume_path}")
                    model = load_with_expanded_obs(
                        old_path=resume_path,
                        new_obs_dim=env.observation_space.shape[0],
                        env=env,
                        ppo_kwargs={**PPO_KWARGS, "tensorboard_log": run.logs_dir},
                    )
                model.verbose = 0
            else:
                model = MaskablePPO(env=env, **{**PPO_KWARGS, "tensorboard_log": run.logs_dir})
        else:
            model.set_env(env)

        win_rate_cb = WinRateCallback(
            window=100,
            eval_freq=10_000,
            save_path=run.models_dir,
            replay_dir=replay_dir,
            notable_dir="replays/notable",
            verbose=1,
            stop_at_win_rate=phase["target_wr"],
            phase_label=phase["phase_label"],
            training_log_path=run.training_log,
        )
        checkpoint_cb = CheckpointCallback(
            save_freq=50_000,
            save_path=run.models_dir,
            name_prefix="ppo_pokemon",
            verbose=0,
        )

        model.learn(
            total_timesteps=phase["max_steps"],
            callback=[win_rate_cb, checkpoint_cb],
            reset_num_timesteps=(model is None and resume_path is None),
        )

        phase_path = str(Path(run.models_dir) / f"phase_{phase['name']}_final")
        model.save(phase_path)
        print(f"\n  Phase {phase['name']} complete → {phase_path}.zip")

    run.mark_complete()
    print(f"\nCurriculum complete. Run selfplay_train.py --run {run.run_id} to continue.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-run", action="store_true", help="Force a new run (don't resume)")
    args = parser.parse_args()
    main(new_run=args.new_run)
