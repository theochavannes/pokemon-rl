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

import logging

import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from src.callbacks import WinRateCallback
from src.env.gen1_env import make_env
from src.logging_config import setup_logging
from src.obs_transfer import is_compatible, load_with_expanded_obs
from src.run_manager import RunManager

log = logging.getLogger("pokemon_rl.train")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BATTLE_FORMAT = "gen1randombattle"
N_ENVS = 4


def _lr_schedule(progress_remaining: float) -> float:
    """Linear LR schedule: 3e-4 → 1e-4 over training."""
    return 1e-4 + (3e-4 - 1e-4) * progress_remaining


PPO_KWARGS = dict(
    policy="MlpPolicy",
    policy_kwargs=dict(net_arch=dict(pi=[256, 128], vf=[256, 128])),
    n_steps=2048,
    batch_size=128,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    max_grad_norm=0.5,
    ent_coef=0.01,
    learning_rate=_lr_schedule,  # 3e-4 → 1e-4 linear anneal
    verbose=1,
)

CURRICULUM = [
    dict(
        name="A",
        opponent_type="mixed_league",
        phase_label="League",
        target_wr=0.70,
        max_steps=2_000_000,
        shaping_factor=1.0,
        epsilon_start=2.0,  # SoftmaxDamagePlayer temperature: 2.0 = soft, anneals to 0.1
        epsilon_end=0.1,
        selfplay=True,
    ),
]


def _prevent_sleep() -> None:
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    atexit.register(lambda: ctypes.windll.kernel32.SetThreadExecutionState(0x80000000))
    print("Sleep prevention: ON")


def main(new_run: bool = False) -> None:
    _prevent_sleep()

    # Build JSON-safe config snapshot (exclude non-serializable callables/classes)
    config_snapshot = {k: v for k, v in PPO_KWARGS.items() if k not in ("policy_kwargs", "learning_rate")}
    config_snapshot["policy_kwargs"] = {"net_arch": {"pi": [256, 128], "vf": [256, 128]}}
    config_snapshot["learning_rate"] = "linear_schedule(3e-4 -> 1e-4)"

    run = RunManager(
        run_type="curriculum",
        config={**config_snapshot, "curriculum": CURRICULUM, "n_envs": N_ENVS},
        new_run=new_run,
    )

    global_episodes = 0  # cumulative battle count across all phases
    os.makedirs(run.models_dir, exist_ok=True)
    os.makedirs(run.logs_dir, exist_ok=True)

    setup_logging(log_dir=run.logs_dir)
    log.info("Starting %s (curriculum training)", run.run_id)

    resume_path = run.latest_checkpoint()
    progress = run.load_progress()
    model = None
    is_fresh_run = resume_path is None  # True only for brand-new runs with no checkpoints

    for phase in CURRICULUM:
        # Skip phases that were already completed in a previous session
        if progress.get("phase") and progress["phase"] > phase["name"]:
            print(f"  Phase {phase['name']} already completed — skipping")
            continue
        replay_dir = run.replays_dir(f"phase_{phase['name'].lower()}")

        print(f"\n{'=' * 60}")
        print(f"  {run.run_id} — Phase {phase['name']}: vs {phase['phase_label']}")
        shaping = phase.get("shaping_factor", 1.0)
        epsilon_start = phase.get("epsilon_start")

        # Restore epsilon from saved progress if resuming this phase
        if progress.get("phase") == phase["name"] and "epsilon" in progress:
            epsilon_start = progress["epsilon"]
            print(f"  Resuming with eps={epsilon_start:.2f} (from saved progress)")

        # Calculate remaining steps if resuming mid-phase
        remaining_steps = phase["max_steps"]
        if resume_path and progress.get("phase") == phase["name"] and "timesteps" in progress:
            done = progress["timesteps"]
            remaining_steps = max(phase["max_steps"] - done, 0)
            print(f"  Resuming from step {done:,} — {remaining_steps:,} steps remaining")

        print(
            f"  Target: {phase['target_wr'] * 100:.0f}%  |  Cap: {remaining_steps:,} steps (~{remaining_steps // 75:,} battles)"
        )
        print(f"  Reward shaping: {shaping:.0%}")
        if epsilon_start is not None:
            print(f"  Opponent eps: {epsilon_start} -> {phase.get('epsilon_end', 0.0)} (anneals on win rate)")
        selfplay_path = str(Path(run.models_dir) / "selfplay_frozen") if phase.get("selfplay") else None
        if selfplay_path:
            print("  Self-play: ON (frozen opponent updated every 200 battles)")
        print(f"  Replays -> {replay_dir}")
        print(f"{'=' * 60}\n")

        # Save current best as initial frozen opponent for self-play
        if selfplay_path:
            best = Path(run.models_dir) / "best_model.zip"
            if best.exists():
                from sb3_contrib import MaskablePPO as _PPO

                _PPO.load(str(best.with_suffix(""))).save(selfplay_path)
            elif model is not None:
                model.save(selfplay_path)
            else:
                # First phase, no model yet — use BC warm-start as initial frozen opponent
                bc_seed = Path("models/bc_warmstart.zip")
                if bc_seed.exists():
                    from sb3_contrib import MaskablePPO as _PPO

                    _PPO.load(str(bc_seed.with_suffix(""))).save(selfplay_path)
                    print("  Self-play: initialized frozen opponent from BC warm-start")
                else:
                    raise FileNotFoundError(
                        "Self-play requires a model but no best_model.zip or models/bc_warmstart.zip found"
                    )

        env_fns = [
            partial(
                make_env,
                env_index=i,
                battle_format=BATTLE_FORMAT,
                save_replays=replay_dir,
                opponent_type=phase["opponent_type"],
                shaping_factor=shaping,
                opponent_difficulty=epsilon_start or 0.8,
                selfplay_model_path=selfplay_path,
            )
            for i in range(N_ENVS)
        ]
        # DummyVecEnv when we need opponent access (epsilon annealing), SubprocVecEnv otherwise
        env = DummyVecEnv(env_fns) if epsilon_start is not None or N_ENVS == 1 else SubprocVecEnv(env_fns)

        if model is None:
            if resume_path:
                if is_compatible(resume_path, env.observation_space.shape[0]):
                    print(f"  Resuming from {resume_path}")
                    model = MaskablePPO.load(
                        resume_path,
                        env=env,
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
            else:
                # Try loading BC warm-start model (pre-trained on MaxDamagePlayer)
                bc_path = Path("models/bc_warmstart.zip")
                if bc_path.exists():
                    bc_model_path = str(bc_path.with_suffix(""))
                    if is_compatible(bc_model_path, env.observation_space.shape[0]):
                        print(f"  Loading behavioral cloning warm-start from {bc_path}")
                        model = MaskablePPO.load(
                            bc_model_path,
                            env=env,
                            **{k: v for k, v in PPO_KWARGS.items() if k != "verbose"},
                            tensorboard_log=run.logs_dir,
                        )
                        # Halve the BC-inherited anti-switch bias so PPO can learn switching
                        with torch.no_grad():
                            before = model.policy.action_net.bias.data[:6].mean().item()
                            model.policy.action_net.bias.data[:6] *= 0.5
                            after = model.policy.action_net.bias.data[:6].mean().item()
                            print(f"  BC warm-start loaded -- switch bias halved: {before:.2f} -> {after:.2f}")
                    else:
                        print("  BC warm-start obs space mismatch — transferring weights")
                        model = load_with_expanded_obs(
                            old_path=bc_model_path,
                            new_obs_dim=env.observation_space.shape[0],
                            env=env,
                            ppo_kwargs={**PPO_KWARGS, "tensorboard_log": run.logs_dir},
                        )
                else:
                    model = MaskablePPO(env=env, **{**PPO_KWARGS, "tensorboard_log": run.logs_dir})
                    # Bias policy toward moves (actions 6-9) over switches (actions 0-5).
                    # The bias is a learned parameter — PPO will adjust it naturally.
                    with torch.no_grad():
                        model.policy.action_net.bias.data[:6] -= 2.0  # penalize switches
                        model.policy.action_net.bias.data[6:] += 2.0  # boost moves
                        print(
                            f"  Logit bias: switches={model.policy.action_net.bias.data[:6].mean():.1f}, moves={model.policy.action_net.bias.data[6:].mean():.1f}"
                        )
        else:
            model.set_env(env)

        # Print model info
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n
        total_params = sum(p.numel() for p in model.policy.parameters())
        print(f"  Obs space : {obs_dim} dims")
        print(f"  Action sp : {act_dim} actions (4 moves + 6 switches)")
        print(f"  Network   : {total_params:,} parameters")
        print(f"  Rollout   : {PPO_KWARGS['n_steps']} steps × {N_ENVS} env(s)")
        print("  Eval every: 50 battles")
        print()

        # Collect opponent refs for epsilon annealing + selfplay swap (DummyVecEnv only)
        opponents = []
        if hasattr(env, "envs"):
            for e in env.envs:
                if hasattr(e, "_opponent"):
                    opponents.append(e._opponent)

        eps_sched = None
        if epsilon_start is not None:
            eps_sched = (epsilon_start, phase.get("epsilon_end", 0.0))

        win_rate_cb = WinRateCallback(
            window=100,
            eval_freq=10_000,
            save_path=run.models_dir,
            replay_dir=replay_dir,
            notable_dir=str(Path(run.run_dir) / "replays" / "notable"),
            verbose=1,
            stop_at_win_rate=phase["target_wr"],
            phase_label=phase["phase_label"],
            training_log_path=run.training_log,
            run_id=run.run_id,
            epsilon_schedule=eps_sched,
            opponents=opponents,
            run_manager=run,
            phase_name=phase["name"],
            selfplay_path=selfplay_path,
            shaping_decay_battles=5000,
            global_episodes_offset=global_episodes,
            env=env,
        )
        checkpoint_cb = CheckpointCallback(
            save_freq=50_000,
            save_path=run.models_dir,
            name_prefix="ppo_pokemon",
            verbose=0,
        )

        model.learn(
            total_timesteps=remaining_steps,
            callback=[win_rate_cb, checkpoint_cb],
            reset_num_timesteps=is_fresh_run,
        )
        is_fresh_run = False  # Only reset on the very first phase of a new run

        # Carry cumulative episode count to next phase for shaping decay
        global_episodes += win_rate_cb._total_episodes

        # Save phase completion + progress for resume
        current_epsilon = None
        for opp in opponents:
            if hasattr(opp, "temperature"):
                current_epsilon = opp.temperature
                break
            if hasattr(opp, "epsilon"):
                current_epsilon = opp.epsilon
                break
        run.save_progress(phase["name"], phase["max_steps"], current_epsilon)

        phase_path = str(Path(run.models_dir) / f"phase_{phase['name']}_final")
        model.save(phase_path)
        print(f"\n  Phase {phase['name']} complete -> {phase_path}.zip")

        # Clear resume state for next phase
        resume_path = None
        progress = {}

    run.mark_complete()
    print(f"\nCurriculum complete. Run selfplay_train.py --run {run.run_id} to continue.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-run", action="store_true", help="Force a new run (don't resume)")
    args = parser.parse_args()
    main(new_run=args.new_run)
