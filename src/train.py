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

PPO_KWARGS = dict(
    policy="MlpPolicy",  # [64, 64] MLP — tested [128,128] and [256,256], both worse
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,  # Effective horizon ~100 turns (avg game ~50 turns)
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,  # Entropy bonus — prevents policy collapse to a single action
    learning_rate=3e-4,
    verbose=1,
)

CURRICULUM = [
    dict(
        name="A",
        opponent_type="random",
        phase_label="Random",
        target_wr=0.90,
        max_steps=200_000,
        shaping_factor=1.0,
    ),
    dict(
        name="B",
        opponent_type="random_attacker",
        phase_label="RandAttacker",
        target_wr=0.95,
        max_steps=200_000,
        shaping_factor=1.0,
    ),
    dict(
        name="C",
        opponent_type="softmax_damage",
        phase_label="SoftmaxDmg",
        target_wr=0.70,
        max_steps=400_000,
        shaping_factor=1.0,
        epsilon_start=2.0,  # temperature: 2.0 = soft, anneals to 0.1 = near-argmax
        epsilon_end=0.1,
    ),
    dict(
        name="D",
        opponent_type="mixed",
        phase_label="Mixed+Self",
        target_wr=0.60,
        max_steps=500_000,
        shaping_factor=1.0,
        epsilon_start=0.95,
        epsilon_end=0.0,
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

    run = RunManager(
        run_type="curriculum",
        config={**PPO_KWARGS, "curriculum": CURRICULUM, "n_envs": N_ENVS},
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
            print(f"  Resuming with ε={epsilon_start:.2f} (from saved progress)")

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
            print(f"  Opponent ε: {epsilon_start} → {phase.get('epsilon_end', 0.0)} (anneals on win rate)")
        selfplay_path = str(Path(run.models_dir) / "selfplay_frozen") if phase.get("selfplay") else None
        if selfplay_path:
            print("  Self-play: ON (frozen opponent updated every 200 battles)")
        print(f"  Replays → {replay_dir}")
        print(f"{'=' * 60}\n")

        # Save current best as initial frozen opponent for self-play
        if selfplay_path:
            best = Path(run.models_dir) / "best_model.zip"
            if best.exists():
                from sb3_contrib import MaskablePPO as _PPO

                _PPO.load(str(best.with_suffix(""))).save(selfplay_path)
            elif model is not None:
                model.save(selfplay_path)

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
        if opponents:
            opp = opponents[0]
            if hasattr(opp, "temperature"):
                current_epsilon = opp.temperature
            elif hasattr(opp, "epsilon"):
                current_epsilon = opp.epsilon
        run.save_progress(phase["name"], phase["max_steps"], current_epsilon)

        phase_path = str(Path(run.models_dir) / f"phase_{phase['name']}_final")
        model.save(phase_path)
        print(f"\n  Phase {phase['name']} complete → {phase_path}.zip")

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
