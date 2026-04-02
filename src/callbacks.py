"""
Training callbacks for MaskablePPO.

WinRateCallback: tracks win rate from episode rewards during training and
logs to TensorBoard. Saves the best model whenever win rate improves.

Win rate is derived from terminal episode rewards:
    +1.0 → win   |   -1.0 → loss   (no draws in gen1randombattle)
"""

import os
from collections import deque

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class WinRateCallback(BaseCallback):
    """
    Logs win rate over the last `window` completed episodes.
    Saves a checkpoint whenever win rate reaches a new high.

    Args:
        window:     Number of recent episodes to average over.
        eval_freq:  Log frequency in timesteps.
        save_path:  Directory to save best model checkpoints.
        verbose:    0 = silent, 1 = print on improvement.
    """

    def __init__(
        self,
        window: int = 100,
        eval_freq: int = 10_000,
        save_path: str = "models",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.window = window
        self.eval_freq = eval_freq
        self.save_path = save_path
        self._episode_rewards: deque = deque(maxlen=window)
        self._best_win_rate: float = 0.0
        self._last_log_step: int = 0

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
                    print(f"[WinRate] New best win rate {win_rate:.3f} — saved to {path}")

        return True
