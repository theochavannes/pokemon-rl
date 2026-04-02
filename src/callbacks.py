"""
Training callbacks for MaskablePPO.

WinRateCallback:
  - Tracks win rate, episode length, and action distribution
  - Logs to TensorBoard
  - Saves best model on win rate improvement
  - Snapshots replays to replays/notable/ at training milestones
  - Writes human-readable progress to content/training_log.md (for [MEDIA])
  - Appends milestone events to content/hooks.md

Win rate is derived from terminal episode rewards:
    +1.0 → win   |   -1.0 → loss   (no draws in gen1randombattle)
"""

import json
import os
import shutil
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

# Win rate thresholds worth flagging as video milestones.
# Start at 0.55 — random-vs-random baseline is ~50%, so anything below is noise.
_WIN_RATE_MILESTONES = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]

# Step milestones for replay snapshots
_REPLAY_MILESTONES = {10_000, 50_000, 100_000, 200_000, 350_000, 500_000}

# Action index labels (Gen 1 singles — 10 actions)
_ACTION_LABELS = [
    "switch_1", "switch_2", "switch_3", "switch_4", "switch_5", "switch_6",
    "move_1", "move_2", "move_3", "move_4",
]


class WinRateCallback(BaseCallback):
    """
    Full training monitor for MaskablePPO on Gen 1 Pokemon.

    Args:
        window:      Number of recent episodes to average over.
        eval_freq:   Log + snapshot frequency in timesteps.
        save_path:   Directory to save best model checkpoints.
        replay_dir:  Directory where training replays are saved (by make_env).
        notable_dir: Directory to copy milestone replay snapshots into.
        verbose:     0 = silent, 1 = print on improvement.
    """

    def __init__(
        self,
        window: int = 100,
        eval_freq: int = 10_000,
        eval_freq_episodes: int = 50,
        save_path: str = "models",
        replay_dir: str = "replays/phase4",
        notable_dir: str = "replays/notable",
        verbose: int = 1,
        stop_at_win_rate: float | None = None,
        phase_label: str = "",
        training_log_path: str | None = None,
        run_id: str = "",
        epsilon_schedule: tuple[float, float] | None = None,
        opponents: list | None = None,
    ):
        super().__init__(verbose)
        self.window = window
        self.eval_freq = eval_freq
        self.eval_freq_episodes = eval_freq_episodes
        self.save_path = save_path
        self.replay_dir = Path(replay_dir)
        self.notable_dir = Path(notable_dir)
        self.stop_at_win_rate = stop_at_win_rate
        self.phase_label = phase_label or ""
        self.run_id = run_id
        self.epsilon_schedule = epsilon_schedule  # (start, end) or None
        self.opponents = opponents or []          # opponent player objects
        # Per-run log if provided, otherwise fall back to legacy path
        self.content_log = Path(training_log_path) if training_log_path else Path("content/training_log.md")
        self.hooks_log = Path("content/hooks.md")

        self._episode_rewards: deque = deque(maxlen=window)
        self._episode_lengths: deque = deque(maxlen=window)
        self._action_counts = np.zeros(10, dtype=np.int64)
        self._best_win_rate: float = 0.0
        self._last_log_step: int = 0
        self._snapped_milestones: set = set()
        self._crossed_milestones: set = set()
        self._total_episodes: int = 0
        self._last_eval_episode: int = 0
        self._target_hit_once: bool = False
        self._last_heartbeat_step: int = 0
        self._heartbeat_freq: int = 2048  # print a heartbeat every rollout

    def _on_training_start(self) -> None:
        self.content_log.parent.mkdir(parents=True, exist_ok=True)
        self._write_content_log_header()
        if self.verbose:
            label = self.phase_label or "opponent"
            print(f"  Training started — collecting rollouts vs {label}...")
            print(f"  (first battles will print individually)\n")

    def _on_step(self) -> bool:
        # Track action distribution
        actions = self.locals.get("actions")
        if actions is not None:
            for a in np.array(actions).flatten():
                if 0 <= a < 10:
                    self._action_counts[a] += 1

        # Track episode outcomes
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._total_episodes += 1
                reward = info["episode"]["r"]
                length = info["episode"]["l"]
                self._episode_rewards.append(reward)
                self._episode_lengths.append(length)

                # Print first 10 episodes individually so user sees something early
                if self.verbose and self._total_episodes <= 10:
                    result = "WIN" if reward > 0 else "LOSS"
                    print(f"    battle #{self._total_episodes:>3}: {result}  ({length} turns)")

                    if self._total_episodes == 10:
                        wins = sum(1 for r in self._episode_rewards if r > 0)
                        print(f"    ... first 10 battles: {wins}W/{10-wins}L — continuing silently until eval\n")

        # Heartbeat: show step progress between evals
        if self.verbose and self.num_timesteps - self._last_heartbeat_step >= self._heartbeat_freq:
            self._last_heartbeat_step = self.num_timesteps
            eps = self._total_episodes
            # Don't print heartbeat if we're about to print a full eval line
            if self._total_episodes - self._last_eval_episode < self.eval_freq_episodes:
                recent_wins = sum(1 for r in self._episode_rewards if r > 0)
                recent_losses = len(self._episode_rewards) - recent_wins
                print(f"    ... step {self.num_timesteps:>6,}  |  {eps} battles (last {len(self._episode_rewards)}: {recent_wins}W/{recent_losses}L)")

        if (
            self._total_episodes - self._last_eval_episode >= self.eval_freq_episodes
            and len(self._episode_rewards) > 0
        ):
            self._last_eval_episode = self._total_episodes
            self._last_log_step = self.num_timesteps
            self._evaluate()
            # Early-stop this phase when target win rate is sustained
            if (
                self.stop_at_win_rate is not None
                and len(self._episode_rewards) >= self.window
            ):
                win_rate = float(np.mean([1.0 if r > 0 else 0.0 for r in self._episode_rewards]))
                if win_rate >= self.stop_at_win_rate:
                    if not self._target_hit_once:
                        # First time hitting target — flag it, require one more eval to confirm
                        self._target_hit_once = True
                        if self.verbose:
                            print(f"  [Target hit] win rate {win_rate:.3f} >= {self.stop_at_win_rate} — confirming on next eval...")
                    else:
                        # Sustained across 2 consecutive evals
                        if self.verbose:
                            print(f"  [Phase complete] win rate {win_rate:.3f} >= {self.stop_at_win_rate} sustained — advancing")
                        return False
                else:
                    self._target_hit_once = False

        return True

    def _evaluate(self) -> None:
        win_rate = float(np.mean([1.0 if r > 0 else 0.0 for r in self._episode_rewards]))
        avg_length = float(np.mean(self._episode_lengths)) if self._episode_lengths else 0.0
        n = len(self._episode_rewards)
        total_actions = self._action_counts.sum()
        action_pct = (self._action_counts / total_actions * 100) if total_actions > 0 else self._action_counts

        # TensorBoard only — exclude stdout/log/json/csv on each record() call
        _tb_only = ("stdout", "log", "json", "csv")
        self.logger.record("train/win_rate", win_rate, exclude=_tb_only)
        self.logger.record("train/avg_episode_length", avg_length, exclude=_tb_only)
        self.logger.record("train/episodes_in_window", n, exclude=_tb_only)
        self.logger.record("train/switch_rate_pct", float(action_pct[:6].sum()), exclude=_tb_only)
        self.logger.record("train/move_rate_pct", float(action_pct[6:].sum()), exclude=_tb_only)
        self.logger.dump(self.num_timesteps)

        # Anneal opponent epsilon based on win rate
        self._maybe_anneal_epsilon(win_rate)

        if self.verbose:
            bar = "█" * int(win_rate * 20) + "░" * (20 - int(win_rate * 20))
            label = self.phase_label or "opponent"
            eps_str = ""
            if self.opponents and hasattr(self.opponents[0], "epsilon"):
                eps_str = f"  opp_ε={self.opponents[0].epsilon:.2f}"
            print(
                f"  step {self.num_timesteps:>6,} │ "
                f"vs {label}: {win_rate*100:>5.1f}%  [{bar}]  "
                f"avg {avg_length:.0f} turns  "
                f"switch {action_pct[:6].sum():.0f}%  "
                f"({n} eps){eps_str}"
            )

        # Save best model
        if win_rate > self._best_win_rate and n >= self.window // 2:
            self._best_win_rate = win_rate
            os.makedirs(self.save_path, exist_ok=True)
            self.model.save(os.path.join(self.save_path, "best_model"))
            if self.verbose:
                print(f"[WinRate] New best {win_rate:.3f} — saved best_model")

        # Write checkpoint metadata so tournament scripts know each model's win rate
        self._write_checkpoint_metadata(win_rate, avg_length)

        # Content logging
        self._update_content_log(win_rate, avg_length, action_pct, n)
        self._check_win_rate_milestones(win_rate, avg_length, action_pct)
        self._maybe_snapshot_replays(win_rate)

    def _update_content_log(self, win_rate, avg_length, action_pct, n) -> None:
        """Append a row to content/training_log.md."""
        switch_pct = action_pct[:6].sum()
        move_pct = action_pct[6:].sum()
        top_move = int(np.argmax(self._action_counts[6:])) + 1  # 1-indexed
        top_move_pct = action_pct[6 + np.argmax(self._action_counts[6:])]

        row = (
            f"| {self.num_timesteps:>7} "
            f"| {win_rate:.3f} "
            f"| {avg_length:>5.1f} "
            f"| {switch_pct:>5.1f}% "
            f"| move_{top_move} ({top_move_pct:.1f}%) "
            f"| {n} |\n"
        )
        with open(self.content_log, "a", encoding="utf-8") as f:
            f.write(row)

    def _check_win_rate_milestones(self, win_rate, avg_length, action_pct) -> None:
        """Log to hooks.md when win rate crosses a threshold for the first time."""
        for threshold in _WIN_RATE_MILESTONES:
            if win_rate >= threshold and threshold not in self._crossed_milestones:
                self._crossed_milestones.add(threshold)
                label = f"{int(threshold * 100)}%"
                switch_pct = action_pct[:6].sum()
                note = (
                    f"\n### ✅ Win rate crossed {label} — step {self.num_timesteps} "
                    f"({datetime.now().strftime('%Y-%m-%d')})\n"
                    f"- Avg battle length: {avg_length:.1f} turns\n"
                    f"- Switch rate: {switch_pct:.1f}% of actions\n"
                    f"- Best move used: move_{int(np.argmax(self._action_counts[6:])) + 1} "
                    f"({action_pct[6 + np.argmax(self._action_counts[6:])]:.1f}% of move actions)\n"
                    f"- **Why it matters for the video:** The agent crossed {label} win rate "
                    f"vs RandomPlayer at step {self.num_timesteps:,}. "
                    f"{'It is switching more than attacking — may be learning defensive play.' if switch_pct > 40 else 'It heavily prefers attacking over switching — aggressive style.'}\n"
                )
                with open(self.hooks_log, "a", encoding="utf-8") as f:
                    f.write(note)
                if self.verbose:
                    print(f"[MEDIA]  Win rate milestone {label} reached at step {self.num_timesteps} → logged to hooks.md")

    def _maybe_snapshot_replays(self, win_rate: float) -> None:
        """Copy 3 most recent replays to notable/ at milestone steps."""
        milestone = self._nearest_milestone()
        if milestone is None or milestone in self._snapped_milestones:
            return

        replay_files = sorted(
            self.replay_dir.glob("*.html"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )[:3]

        if not replay_files:
            return

        self._snapped_milestones.add(milestone)
        self.notable_dir.mkdir(parents=True, exist_ok=True)
        step_label = f"{milestone // 1000}k"

        phase_tag = self.phase_label.lower().replace(" ", "_") if self.phase_label else "unknown"
        run_tag = self.run_id or "norun"
        for i, src in enumerate(replay_files, start=1):
            dst = self.notable_dir / f"{run_tag}_{phase_tag}_step{step_label}_wr{win_rate:.2f}_{i}.html"
            shutil.copy2(src, dst)

        if self.verbose:
            print(
                f"[MEDIA]  Snapped {len(replay_files)} replays → "
                f"replays/notable/ (step {step_label}, wr={win_rate:.3f})"
            )

    def _write_checkpoint_metadata(self, win_rate: float, avg_length: float) -> None:
        """
        Append a metadata entry to models/checkpoint_registry.json.
        Each entry maps a checkpoint filename → stats at the time of saving.
        Used by tournament and comparison scripts to select agents.
        """
        registry_path = Path(self.save_path) / "checkpoint_registry.json"
        registry = {}
        if registry_path.exists():
            with open(registry_path, "r") as f:
                registry = json.load(f)

        step = self.num_timesteps
        # Match SB3 CheckpointCallback naming convention
        checkpoint_name = f"ppo_pokemon_{step}_steps"
        registry[checkpoint_name] = {
            "step": step,
            "win_rate": round(win_rate, 4),
            "avg_battle_length": round(avg_length, 1),
            "timestamp": datetime.now().isoformat(),
            "is_best": win_rate >= self._best_win_rate,
        }
        # Always update best_model entry
        registry["best_model"] = {
            "step": step,
            "win_rate": round(self._best_win_rate, 4),
            "timestamp": datetime.now().isoformat(),
        }
        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=2)

    def _maybe_anneal_epsilon(self, win_rate: float) -> None:
        """Decrease opponent epsilon when agent is winning enough."""
        if not self.epsilon_schedule or not self.opponents:
            return
        eps_start, eps_end = self.epsilon_schedule
        if not hasattr(self.opponents[0], "epsilon"):
            return

        current_eps = self.opponents[0].epsilon
        if current_eps <= eps_end:
            return

        # When win rate > 60%, step epsilon down by 0.1
        # This is self-regulating: harder opponent → win rate drops → no more annealing
        if win_rate >= 0.60 and len(self._episode_rewards) >= self.window:
            new_eps = max(current_eps - 0.1, eps_end)
            for opp in self.opponents:
                opp.epsilon = new_eps
            if self.verbose:
                print(f"  [Curriculum] opponent epsilon: {current_eps:.2f} → {new_eps:.2f} (win rate {win_rate:.2f} >= 0.60)")

    def _nearest_milestone(self) -> int | None:
        for m in _REPLAY_MILESTONES:
            if self._last_log_step >= m and m not in self._snapped_milestones:
                return m
        return None

    def _write_content_log_header(self) -> None:
        """Write/append a phase header to training_log.md."""
        date = datetime.now().strftime("%Y-%m-%d %H:%M")
        label = self.phase_label or "Unknown"
        target = self.stop_at_win_rate or 0
        header = (
            f"\n## Phase vs {label}\n\n"
            f"Started: {date}  \n"
            f"Target: {target*100:.0f}% win rate\n\n"
            f"| Step    | Win Rate | Avg Turns | Switch% | Top Move         | Episodes |\n"
            f"|---------|----------|-----------|---------|------------------|----------|\n"
        )
        mode = "a" if self.content_log.exists() else "w"
        with open(self.content_log, mode, encoding="utf-8") as f:
            if mode == "w":
                f.write(f"# {self.run_id} Training Log\n")
            f.write(header)
