"""
Curriculum training: RandomPlayer → MaxDamagePlayer → self-play trigger.

Each phase runs until the win-rate target is hit OR the step cap is reached.
After phase B, the best_model is handed off to selfplay_train.py automatically.

Usage (Showdown server must be running first):
    C:/Users/theoc/miniconda3/envs/pokemon_rl/python.exe src/train.py

TensorBoard:
    tensorboard --logdir logs/
"""

import atexit
import ctypes
import os
import re
import sys
from functools import partial
from pathlib import Path

# Ensure all relative paths resolve from project root regardless of launch directory.
_PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(_PROJECT_ROOT)
sys.path.insert(0, str(_PROJECT_ROOT))

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from src.callbacks import WinRateCallback
from src.env.gen1_env import make_env

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BATTLE_FORMAT = "gen1randombattle"
MODEL_DIR = "models"
LOG_DIR = "logs"

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
    tensorboard_log=LOG_DIR,
)

# Curriculum: agent advances to next phase when target_wr is reached or step cap hit.
# Phase A: vs RandomPlayer  — learns basic move selection and type advantage
# Phase B: vs MaxDamage     — forced to learn switching, prediction, not just spamming
# After phase B: selfplay_train.py takes over
CURRICULUM = [
    dict(
        name="A",
        opponent_type="random",
        phase_label="Random",
        replay_dir="replays/phase_a",
        target_wr=0.75,
        max_steps=150_000,
    ),
    dict(
        name="B",
        opponent_type="maxdamage",
        phase_label="MaxDamage",
        replay_dir="replays/phase_b",
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


def _find_latest_checkpoint(model_dir: str) -> str | None:
    """Return path to the highest-step checkpoint in model_dir, or None."""
    best_step, best_path = -1, None
    for p in Path(model_dir).glob("ppo_pokemon_*_steps.zip"):
        m = re.search(r"ppo_pokemon_(\d+)_steps", p.stem)
        if m and int(m.group(1)) > best_step:
            best_step = int(m.group(1))
            best_path = str(p)
    return best_path


def _build_env(phase: dict) -> DummyVecEnv:
    os.makedirs(phase["replay_dir"], exist_ok=True)
    env_fns = [
        partial(
            make_env,
            env_index=i,
            battle_format=BATTLE_FORMAT,
            save_replays=phase["replay_dir"],
            opponent_type=phase["opponent_type"],
        )
        for i in range(N_ENVS)
    ]
    return DummyVecEnv(env_fns)


def main() -> None:
    _prevent_sleep()
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    resume_path = _find_latest_checkpoint(MODEL_DIR)
    model = None

    for phase in CURRICULUM:
        print(f"\n{'='*60}")
        print(f"  Phase {phase['name']}: vs {phase['phase_label']}")
        print(f"  Target win rate: {phase['target_wr']*100:.0f}%  |  Cap: {phase['max_steps']:,} steps")
        print(f"{'='*60}\n")

        env = _build_env(phase)

        if model is None:
            if resume_path:
                print(f"  Resuming from {resume_path}")
                model = MaskablePPO.load(
                    resume_path,
                    env=env,
                    **{k: v for k, v in PPO_KWARGS.items() if k != "verbose"},
                )
                model.verbose = 0
            else:
                model = MaskablePPO(env=env, **PPO_KWARGS)
        else:
            model.set_env(env)

        win_rate_cb = WinRateCallback(
            window=100,
            eval_freq=10_000,
            save_path=MODEL_DIR,
            replay_dir=phase["replay_dir"],
            notable_dir="replays/notable",
            verbose=1,
            stop_at_win_rate=phase["target_wr"],
            phase_label=phase["phase_label"],
        )
        checkpoint_cb = CheckpointCallback(
            save_freq=50_000,
            save_path=MODEL_DIR,
            name_prefix="ppo_pokemon",
            verbose=0,
        )

        model.learn(
            total_timesteps=phase["max_steps"],
            callback=[win_rate_cb, checkpoint_cb],
            reset_num_timesteps=(model is None and resume_path is None),
        )

        # Save phase checkpoint so we can resume or inspect
        phase_path = os.path.join(MODEL_DIR, f"phase_{phase['name']}_final")
        model.save(phase_path)
        print(f"\n  Phase {phase['name']} complete — saved {phase_path}.zip")

    print("\nCurriculum complete. Run selfplay_train.py to continue with self-play.")


if __name__ == "__main__":
    main()
