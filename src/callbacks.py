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

import os
import shutil
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

# Win rate thresholds worth flagging as video milestones
_WIN_RATE_MILESTONES = [0.10, 0.25, 0.40, 0.50, 0.60, 0.75]

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
        save_path: str = "models",
        replay_dir: str = "replays/phase4",
        notable_dir: str = "replays/notable",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.window = window
        self.eval_freq = eval_freq
        self.save_path = save_path
        self.replay_dir = Path(replay_dir)
        self.notable_dir = Path(notable_dir)
        self.content_log = Path("content/training_log.md")
        self.hooks_log = Path("content/hooks.md")

        self._episode_rewards: deque = deque(maxlen=window)
        self._episode_lengths: deque = deque(maxlen=window)
        self._action_counts = np.zeros(10, dtype=np.int64)
        self._best_win_rate: float = 0.0
        self._last_log_step: int = 0
        self._snapped_milestones: set = set()
        self._crossed_milestones: set = set()

    def _on_training_start(self) -> None:
        self.content_log.parent.mkdir(parents=True, exist_ok=True)
        self._write_content_log_header()

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
                self._episode_rewards.append(info["episode"]["r"])
                self._episode_lengths.append(info["episode"]["l"])

        if (
            self.num_timesteps - self._last_log_step >= self.eval_freq
            and len(self._episode_rewards) > 0
        ):
            self._last_log_step = self.num_timesteps
            self._evaluate()

        return True

    def _evaluate(self) -> None:
        win_rate = float(np.mean([1.0 if r > 0 else 0.0 for r in self._episode_rewards]))
        avg_length = float(np.mean(self._episode_lengths)) if self._episode_lengths else 0.0
        n = len(self._episode_rewards)
        total_actions = self._action_counts.sum()
        action_pct = (self._action_counts / total_actions * 100) if total_actions > 0 else self._action_counts

        # TensorBoard
        self.logger.record("train/win_rate", win_rate)
        self.logger.record("train/avg_episode_length", avg_length)
        self.logger.record("train/episodes_in_window", n)
        self.logger.record("train/switch_rate_pct", float(action_pct[:6].sum()))
        self.logger.record("train/move_rate_pct", float(action_pct[6:].sum()))
        self.logger.dump(self.num_timesteps)

        if self.verbose:
            print(
                f"[WinRate] step={self.num_timesteps:>7}  "
                f"win={win_rate:.3f}  avg_turns={avg_length:.1f}  "
                f"switch%={action_pct[:6].sum():.1f}  ({n} eps)"
            )

        # Save best model
        if win_rate > self._best_win_rate and n >= self.window // 2:
            self._best_win_rate = win_rate
            os.makedirs(self.save_path, exist_ok=True)
            self.model.save(os.path.join(self.save_path, "best_model"))
            if self.verbose:
                print(f"[WinRate] New best {win_rate:.3f} — saved best_model")

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

        for i, src in enumerate(replay_files, start=1):
            dst = self.notable_dir / f"phase4_step{step_label}_wr{win_rate:.2f}_{i}.html"
            shutil.copy2(src, dst)

        if self.verbose:
            print(
                f"[MEDIA]  Snapped {len(replay_files)} replays → "
                f"replays/notable/ (step {step_label}, wr={win_rate:.3f})"
            )

    def _nearest_milestone(self) -> int | None:
        for m in _REPLAY_MILESTONES:
            if self._last_log_step >= m and m not in self._snapped_milestones:
                return m
        return None

    def _write_content_log_header(self) -> None:
        """Write the markdown table header to training_log.md."""
        date = datetime.now().strftime("%Y-%m-%d %H:%M")
        header = (
            f"# Phase 4 Training Log\n\n"
            f"Started: {date}  \n"
            f"Format: gen1randombattle vs RandomPlayer  \n"
            f"Target: win rate >60% before self-play (Phase 5)\n\n"
            f"| Step    | Win Rate | Avg Turns | Switch% | Top Move         | Episodes |\n"
            f"|---------|----------|-----------|---------|------------------|----------|\n"
        )
        with open(self.content_log, "w", encoding="utf-8") as f:
            f.write(header)
