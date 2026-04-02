"""
Training callbacks for MaskablePPO.

WinRateCallback: tracks win rate from episode rewards during training and
logs to TensorBoard. Saves the best model whenever win rate improves.
At each eval interval, snapshots the 3 most recent battle replays to
replays/notable/ so training progress is visually documented.

Win rate is derived from terminal episode rewards:
    +1.0 → win   |   -1.0 → loss   (no draws in gen1randombattle)
"""

import os
import shutil
from collections import deque
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

# Milestone steps at which to snapshot replays for the video
_REPLAY_MILESTONES = {10_000, 50_000, 100_000, 200_000, 350_000, 500_000}


class WinRateCallback(BaseCallback):
    """
    Logs win rate over the last `window` completed episodes.
    Saves a checkpoint whenever win rate reaches a new high.
    Snapshots 3 recent replays to replays/notable/ at training milestones.

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
        self._episode_rewards: deque = deque(maxlen=window)
        self._best_win_rate: float = 0.0
        self._last_log_step: int = 0
        self._snapped_milestones: set = set()

    def _on_step(self) -> bool:
        # SB3 adds {"episode": {"r": total_reward, ...}} to infos on episode end
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._episode_rewards.append(info["episode"]["r"])

        if (
            self.num_timesteps - self._last_log_step >= self.eval_freq
            and len(self._episode_rewards) > 0
        ):
            self._last_log_step = self.num_timesteps
            win_rate = np.mean([1.0 if r > 0 else 0.0 for r in self._episode_rewards])
            n = len(self._episode_rewards)

            self.logger.record("train/win_rate", win_rate)
            self.logger.record("train/episodes_in_window", n)
            self.logger.dump(self.num_timesteps)

            if self.verbose:
                print(
                    f"[WinRate] step={self.num_timesteps:>7}  "
                    f"win_rate={win_rate:.3f}  ({n} episodes)"
                )

            if win_rate > self._best_win_rate and n >= self.window // 2:
                self._best_win_rate = win_rate
                os.makedirs(self.save_path, exist_ok=True)
                path = os.path.join(self.save_path, "best_model")
                self.model.save(path)
                if self.verbose:
                    print(f"[WinRate] New best {win_rate:.3f} — saved to {path}")

            self._maybe_snapshot_replays(win_rate)

        return True

    def _maybe_snapshot_replays(self, win_rate: float) -> None:
        """Copy the 3 most recent replays to notable/ at milestone steps."""
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
            dst = self.notable_dir / f"phase4_step{step_label}_winrate{win_rate:.2f}_{i}.html"
            shutil.copy2(src, dst)

        if self.verbose:
            print(
                f"[MEDIA]  Snapped {len(replay_files)} replays → "
                f"replays/notable/ (step {step_label}, win_rate={win_rate:.3f})"
            )

    def _nearest_milestone(self) -> int | None:
        """Return the nearest milestone if we just passed it, else None."""
        for m in _REPLAY_MILESTONES:
            if self._last_log_step >= m and m not in self._snapped_milestones:
                return m
        return None
