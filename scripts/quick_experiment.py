"""
Quick experiment runner: test a single hyperparameter change for 50K steps.

Usage:
    python scripts/quick_experiment.py --gamma 0.95 --tag gamma095
    python scripts/quick_experiment.py --gae-lambda 1.0 --tag mc_returns
    python scripts/quick_experiment.py --gamma 0.95 --gae-lambda 1.0 --tag gamma095_mc
"""

import argparse
import os
import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
os.chdir(Path(__file__).parent.parent)

import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.callbacks import WinRateCallback
from src.env.gen1_env import make_env


def run_experiment(
    tag: str,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    total_steps: int = 50_000,
):
    print(f"\n{'=' * 60}")
    print(f"  Experiment: {tag}")
    print(f"  gamma={gamma}, gae_lambda={gae_lambda}, steps={total_steps:,}")
    print(f"{'=' * 60}\n")

    out_dir = Path(f"runs/experiment_{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = str(out_dir / "models")
    replay_dir = str(out_dir / "replays")
    os.makedirs(models_dir, exist_ok=True)

    selfplay_path = str(out_dir / "models" / "selfplay_frozen")

    # Copy BC as selfplay opponent
    bc_path = Path("models/bc_warmstart.zip")
    if bc_path.exists():
        MaskablePPO.load(str(bc_path.with_suffix(""))).save(selfplay_path)

    env_fns = [
        partial(
            make_env,
            env_index=i,
            battle_format="gen1randombattle",
            save_replays=replay_dir,
            opponent_type="mixed_league",
            shaping_factor=1.0,
            opponent_difficulty=2.0,
            selfplay_model_path=selfplay_path,
        )
        for i in range(4)
    ]
    env = DummyVecEnv(env_fns)

    # Load BC warm-start with experiment hyperparams
    model = MaskablePPO.load(
        str(bc_path.with_suffix("")),
        env=env,
        n_steps=2048,
        batch_size=128,
        n_epochs=3,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=0.1,
        max_grad_norm=0.5,
        ent_coef=0.01,
        learning_rate=1e-4,
    )

    # Halve switch bias
    with torch.no_grad():
        model.policy.action_net.bias.data[:6] *= 0.5

    opponents = []
    if hasattr(env, "envs"):
        for e in env.envs:
            if hasattr(e, "_opponent"):
                opponents.append(e._opponent)

    cb = WinRateCallback(
        window=100,
        eval_freq=10_000,
        save_path=models_dir,
        replay_dir=replay_dir,
        notable_dir=str(out_dir / "replays" / "notable"),
        verbose=1,
        stop_at_win_rate=None,
        phase_label=tag,
        training_log_path=str(out_dir / "training_log.md"),
        run_id=f"exp_{tag}",
        epsilon_schedule=(2.0, 0.1),
        opponents=opponents,
        phase_name="exp",
        selfplay_path=selfplay_path,
        env=env,
    )

    model.learn(total_timesteps=total_steps, callback=[cb], reset_num_timesteps=True)

    # Print final summary
    print(f"\n{'=' * 60}")
    print(f"  Experiment {tag} complete")
    print(f"  gamma={gamma}, gae_lambda={gae_lambda}")
    sb3_vals = getattr(model.logger, "name_to_value", {})
    ev = sb3_vals.get("train/explained_variance", float("nan"))
    print(f"  Final ExplVar: {ev:.3f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--steps", type=int, default=50_000)
    args = parser.parse_args()
    run_experiment(args.tag, args.gamma, args.gae_lambda, args.steps)
