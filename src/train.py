"""
Phase 4 — PPO training loop.

Trains MaskablePPO against RandomPlayer opponents.
Target: win rate >60% vs MaxDamagePlayer before moving to self-play (Phase 5).

Usage (Showdown server must be running first):
    C:/Users/theoc/miniconda3/envs/pokemon_rl/python.exe src/train.py

TensorBoard:
    tensorboard --logdir logs/
"""

import os
import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback
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

# N_ENVS=1 uses DummyVecEnv for easy debugging.
# To scale: set N_ENVS=4 and switch to SubprocVecEnv (see comment in main()).
# Each subprocess needs its own asyncio event loop — poke-env 0.13 supports this.
N_ENVS = 1

# PPO hyperparameters — validated by team (see content/rl_concepts/ppo_explained.md)
PPO_KWARGS = dict(
    policy="MlpPolicy",
    n_steps=2048,       # rollout length per env per update
    batch_size=64,      # must divide n_steps * N_ENVS evenly
    n_epochs=10,        # gradient steps per rollout
    gamma=0.99,         # discount — keeps signal over ~40-move battles
    gae_lambda=0.95,    # GAE λ — γλ=0.94 adequate for sparse terminal reward
    clip_range=0.2,     # PPO trust region
    learning_rate=3e-4,
    verbose=1,
    tensorboard_log=LOG_DIR,
)


def main() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # --- Build vectorised environment ---
    env_fns = [partial(make_env, env_index=i, battle_format=BATTLE_FORMAT) for i in range(N_ENVS)]

    # DummyVecEnv: single process, easy to debug.
    # SubprocVecEnv upgrade path (Phase 4 scale-up):
    #   from stable_baselines3.common.vec_env import SubprocVecEnv
    #   env = SubprocVecEnv(env_fns)
    env = DummyVecEnv(env_fns)

    # --- Build model ---
    model = MaskablePPO(env=env, **PPO_KWARGS)

    # --- Callbacks ---
    win_rate_cb = WinRateCallback(
        window=100,
        eval_freq=10_000,
        save_path=MODEL_DIR,
        verbose=1,
    )
    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=MODEL_DIR,
        name_prefix="ppo_pokemon",
        verbose=1,
    )

    # --- Train ---
    print(f"Training MaskablePPO for {TOTAL_TIMESTEPS:,} steps ({N_ENVS} env(s))...")
    print(f"TensorBoard: tensorboard --logdir {LOG_DIR}")
    print()

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[win_rate_cb, checkpoint_cb],
        reset_num_timesteps=True,
    )

    final_path = os.path.join(MODEL_DIR, "final_model")
    model.save(final_path)
    print(f"\nTraining complete. Final model saved to {final_path}.zip")


if __name__ == "__main__":
    main()
