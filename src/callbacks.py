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
import logging
import os
import shutil
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

log = logging.getLogger("pokemon_rl.callback")

# Win rate thresholds worth flagging as video milestones.
# Start at 0.55 — random-vs-random baseline is ~50%, so anything below is noise.
_WIN_RATE_MILESTONES = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]

# Step milestones for replay snapshots
_REPLAY_MILESTONES = {10_000, 50_000, 100_000, 200_000, 350_000, 500_000}

# Action index labels (Gen 1 singles — 10 actions)
_ACTION_LABELS = [
    "switch_1",
    "switch_2",
    "switch_3",
    "switch_4",
    "switch_5",
    "switch_6",
    "move_1",
    "move_2",
    "move_3",
    "move_4",
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
        run_manager=None,
        phase_name: str = "",
        selfplay_path: str | None = None,
        selfplay_update_freq: int = 200,
        shaping_decay_battles: int = 5000,
        global_episodes_offset: int = 0,
        env=None,
    ):
        super().__init__(verbose)
        self.selfplay_path = selfplay_path
        self.selfplay_update_freq = selfplay_update_freq
        self._last_selfplay_update: int = 0
        self.shaping_decay_battles = shaping_decay_battles
        self.global_episodes_offset = global_episodes_offset  # battles from previous phases
        self._env = env
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
        self.opponents = opponents or []  # opponent player objects
        self.run_manager = run_manager
        self.phase_name = phase_name
        # Per-run log if provided, otherwise fall back to legacy path
        self.content_log = Path(training_log_path) if training_log_path else Path("content/training_log.md")
        self.hooks_log = Path("content/hooks.md")

        self._episode_rewards: deque = deque(maxlen=window)
        self._episode_lengths: deque = deque(maxlen=window)
        self._epsilon_rewards: deque = deque(maxlen=500)  # long window for stable epsilon calc
        self._action_counts = np.zeros(10, dtype=np.int64)
        self._forced_switch_count: int = 0
        self._voluntary_switch_count: int = 0
        self._best_win_rate: float = 0.0
        self._last_log_step: int = 0
        self._snapped_milestones: set = set()
        self._crossed_milestones: set = set()
        self._total_episodes: int = 0
        self._last_eval_episode: int = 0
        self._target_hit_once: bool = False
        self._last_heartbeat_step: int = 0
        self._heartbeat_freq: int = 2048  # print a heartbeat every rollout

        # Move quality tracking
        self._best_move_picks: int = 0  # times agent picked highest-damage move
        self._move_decisions: int = 0  # total move decisions (excludes switches)
        self._damage_efficiency_sum: float = 0.0  # sum of (chosen_dmg / best_dmg)
        self._opp_fainted_total: int = 0  # total opponent Pokemon fainted
        self._own_fainted_total: int = 0  # total own Pokemon fainted
        self._fainted_episode_count: int = 0  # episodes counted for faint stats

        # Per-opponent tracking (for mixed_league: env0=SelfPlay, env1=MaxDmg, env2=TypeMatch, env3=SoftmaxDmg)
        self._per_env_wins: list[int] = [0, 0, 0, 0]
        self._per_env_total: list[int] = [0, 0, 0, 0]
        self._per_env_labels: list[str] = ["SelfPlay", "MaxDmg", "TypeMatch", "SoftmaxDmg"]

    def _on_training_start(self) -> None:
        self.content_log.parent.mkdir(parents=True, exist_ok=True)
        self._write_content_log_header()
        label = self.phase_label or "opponent"
        log.info("Training started — collecting rollouts vs %s", label)
        if self.verbose:
            print(f"  Training started — collecting rollouts vs {label}...")
            print("  (first battles will print individually)\n")

    @staticmethod
    def _move_damages_from_obs(obs: np.ndarray) -> np.ndarray:
        """Extract expected damage proxy for each of 4 moves from the observation.

        Own moves start at index 16, with 5 features per move:
          [base_power, type_index, pp_fraction, effectiveness, accuracy]
        Expected damage proxy = base_power * effectiveness * 4.0  (undo normalization)
        """
        damages = np.zeros(4)
        for i in range(4):
            base = 16 + i * 5
            bp = obs[base]  # base_power / 150
            eff = obs[base + 3]  # effectiveness / 4
            damages[i] = bp * eff * 4.0  # rough damage proxy
        return damages

    def _on_step(self) -> bool:
        # Track action distribution + forced vs voluntary switches
        actions = self.locals.get("actions")
        if actions is not None:
            # Use pre-step masks from MaskablePPO's rollout (the masks that were
            # active when the action was chosen), NOT the post-step env masks.
            action_masks = self.locals.get("action_masks")
            observations = self.locals.get("obs_tensor")
            # Convert obs to numpy if it's a tensor
            if observations is not None and hasattr(observations, "cpu"):
                obs_np = observations.cpu().numpy()
            else:
                obs_np = observations

            for idx, a in enumerate(np.array(actions).flatten()):
                if 0 <= a < 10:
                    self._action_counts[a] += 1
                    if a < 6:  # switch action
                        # Forced switch = no move actions available in the mask
                        if action_masks is not None and idx < len(action_masks):
                            moves_available = action_masks[idx][6:].any()
                            if moves_available:
                                self._voluntary_switch_count += 1
                            else:
                                self._forced_switch_count += 1
                        else:
                            self._voluntary_switch_count += 1  # assume voluntary if no mask info
                    elif a >= 6 and obs_np is not None and idx < len(obs_np):
                        # Track move quality for move actions (6-9)
                        damages = self._move_damages_from_obs(obs_np[idx])
                        move_idx = a - 6  # 0-3
                        chosen_dmg = damages[move_idx]
                        best_dmg = damages.max()
                        if best_dmg > 0:
                            self._move_decisions += 1
                            self._damage_efficiency_sum += chosen_dmg / best_dmg
                            if move_idx == int(np.argmax(damages)):
                                self._best_move_picks += 1

        # Track episode outcomes
        for env_idx, info in enumerate(self.locals.get("infos", [])):
            if "episode" in info:
                self._total_episodes += 1
                reward = info["episode"]["r"]
                length = info["episode"]["l"]
                self._episode_rewards.append(reward)
                self._episode_lengths.append(length)
                self._epsilon_rewards.append(reward)

                # Per-opponent tracking (DummyVecEnv: env_idx maps to opponent)
                if env_idx < len(self._per_env_total):
                    self._per_env_total[env_idx] += 1
                    if reward >= 0.5:
                        self._per_env_wins[env_idx] += 1

                # Log first 10 episodes individually so user sees something early
                if self._total_episodes <= 10:
                    result = "WIN" if reward > 0 else "LOSS"
                    log.info("battle #%3d: %s  (%d turns)", self._total_episodes, result, length)
                    if self.verbose:
                        print(f"    battle #{self._total_episodes:>3}: {result}  ({length} turns)")

                    if self._total_episodes == 10:
                        wins = sum(1 for r in self._episode_rewards if r >= 0.5)
                        log.info("First 10 battles: %dW/%dL", wins, 10 - wins)
                        if self.verbose:
                            print(f"    ... first 10 battles: {wins}W/{10 - wins}L — continuing silently until eval\n")

        # Heartbeat: show step progress between evals
        if self.num_timesteps - self._last_heartbeat_step >= self._heartbeat_freq:
            self._last_heartbeat_step = self.num_timesteps
            eps = self._total_episodes
            # Don't log heartbeat if we're about to log a full eval line
            if self._total_episodes - self._last_eval_episode < self.eval_freq_episodes:
                recent_wins = sum(1 for r in self._episode_rewards if r >= 0.5)
                recent_losses = len(self._episode_rewards) - recent_wins
                log.debug(
                    "step %6d | %d battles (last %d: %dW/%dL)",
                    self.num_timesteps,
                    eps,
                    len(self._episode_rewards),
                    recent_wins,
                    recent_losses,
                )
                if self.verbose:
                    print(
                        f"    ... step {self.num_timesteps:>6,}  |  {eps} battles (last {len(self._episode_rewards)}: {recent_wins}W/{recent_losses}L)"
                    )

        if self._total_episodes - self._last_eval_episode >= self.eval_freq_episodes and len(self._episode_rewards) > 0:
            self._last_eval_episode = self._total_episodes
            self._last_log_step = self.num_timesteps
            self._evaluate()
            # Early-stop: only graduate when BOTH conditions are met:
            # 1. Win rate target is sustained (2 consecutive evals)
            # 2. If epsilon annealing is active, epsilon must have reached its end value
            if self.stop_at_win_rate is not None and len(self._episode_rewards) >= self.window:
                # Check if difficulty has fully annealed (epsilon or temperature)
                epsilon_done = True
                if self.epsilon_schedule and self.opponents:
                    _, end_val = self.epsilon_schedule
                    temp_opps = [o for o in self.opponents if hasattr(o, "temperature")]
                    eps_opps = [o for o in self.opponents if hasattr(o, "epsilon")]
                    if temp_opps:
                        cur = temp_opps[0].temperature
                        epsilon_done = cur <= end_val + 0.05
                        if not epsilon_done:
                            log.debug("Curriculum: temperature=%.2f, need <=%.2f to graduate", cur, end_val)
                            if self.verbose and not hasattr(self, "_eps_warned"):
                                print(
                                    f"  [Curriculum] temperature still at {cur:.2f} — phase won't end until {end_val:.2f}"
                                )
                                self._eps_warned = True
                    elif eps_opps:
                        cur = eps_opps[0].epsilon
                        epsilon_done = cur <= end_val + 0.01
                        if not epsilon_done:
                            log.debug("Curriculum: epsilon=%.2f, need <=%.2f to graduate", cur, end_val)
                            if self.verbose and not hasattr(self, "_eps_warned"):
                                print(
                                    f"  [Curriculum] epsilon still at {cur:.2f} — phase won't end until {end_val:.2f}"
                                )
                                self._eps_warned = True

                win_rate = float(np.mean([1.0 if r >= 0.5 else 0.0 for r in self._episode_rewards]))
                if win_rate >= self.stop_at_win_rate and epsilon_done:
                    if not self._target_hit_once:
                        self._target_hit_once = True
                        log.info(
                            "Target hit: win_rate=%.3f >= %.3f — confirming next eval", win_rate, self.stop_at_win_rate
                        )
                        if self.verbose:
                            print(
                                f"  [Target hit] win rate {win_rate:.3f} >= {self.stop_at_win_rate} at epsilon 0.0 — confirming..."
                            )
                    else:
                        log.info(
                            "Phase complete: win_rate=%.3f >= %.3f at full difficulty — advancing",
                            win_rate,
                            self.stop_at_win_rate,
                        )
                        if self.verbose:
                            print(
                                f"  [Phase complete] win rate {win_rate:.3f} >= {self.stop_at_win_rate} at full difficulty — advancing"
                            )
                        return False
                else:
                    self._target_hit_once = False

        return True

    def _evaluate(self) -> None:
        win_rate = float(np.mean([1.0 if r >= 0.5 else 0.0 for r in self._episode_rewards]))
        avg_length = float(np.mean(self._episode_lengths)) if self._episode_lengths else 0.0
        n = len(self._episode_rewards)
        total_actions = self._action_counts.sum()
        action_pct = (self._action_counts / total_actions * 100) if total_actions > 0 else self._action_counts

        # Compute voluntary vs forced switch rates
        vol_switch_pct = (self._voluntary_switch_count / total_actions * 100) if total_actions > 0 else 0.0
        forced_switch_pct = (self._forced_switch_count / total_actions * 100) if total_actions > 0 else 0.0

        # Move quality metrics
        best_move_rate = (self._best_move_picks / self._move_decisions * 100) if self._move_decisions > 0 else 0.0
        dmg_efficiency = (self._damage_efficiency_sum / self._move_decisions) if self._move_decisions > 0 else 0.0

        # Mean episode return and variance
        mean_return = float(np.mean(list(self._episode_rewards))) if self._episode_rewards else 0.0
        return_std = float(np.std(list(self._episode_rewards))) if len(self._episode_rewards) > 1 else 0.0

        # Grab SB3's PPO training metrics from the logger
        sb3_vals = getattr(self.model.logger, "name_to_value", {})
        expl_var = sb3_vals.get("train/explained_variance", float("nan"))
        entropy = sb3_vals.get("train/entropy_loss", float("nan"))
        approx_kl = sb3_vals.get("train/approx_kl", float("nan"))
        pg_loss = sb3_vals.get("train/policy_gradient_loss", float("nan"))
        value_loss = sb3_vals.get("train/value_loss", float("nan"))
        clip_frac = sb3_vals.get("train/clip_fraction", float("nan"))

        # TensorBoard only — exclude stdout/log/json/csv on each record() call
        _tb_only = ("stdout", "log", "json", "csv")
        self.logger.record("train/win_rate", win_rate, exclude=_tb_only)
        self.logger.record("train/avg_episode_length", avg_length, exclude=_tb_only)
        self.logger.record("train/episodes_in_window", n, exclude=_tb_only)
        self.logger.record("train/voluntary_switch_pct", vol_switch_pct, exclude=_tb_only)
        self.logger.record("train/forced_switch_pct", forced_switch_pct, exclude=_tb_only)
        self.logger.record("train/move_rate_pct", float(action_pct[6:].sum()), exclude=_tb_only)
        self.logger.record("train/best_move_rate", best_move_rate, exclude=_tb_only)
        self.logger.record("train/damage_efficiency", dmg_efficiency, exclude=_tb_only)
        self.logger.record("train/mean_episode_return", mean_return, exclude=_tb_only)
        self.logger.record("train/return_std", return_std, exclude=_tb_only)
        # Per-opponent win rates (mixed_league)
        for idx, lbl in enumerate(self._per_env_labels):
            if self._per_env_total[idx] > 0:
                opp_wr = self._per_env_wins[idx] / self._per_env_total[idx]
                self.logger.record(f"train/wr_{lbl}", opp_wr, exclude=_tb_only)
        self.logger.dump(self.num_timesteps)

        # Decay reward shaping based on total battles
        self._decay_shaping()

        # Anneal opponent epsilon based on win rate
        self._maybe_anneal_epsilon(win_rate)

        # Dynamic entropy: prevent death spiral by forcing exploration when stuck
        self._maybe_adjust_entropy(win_rate)

        # Always log eval to file (structured for post-run analysis)
        label = self.phase_label or "opponent"
        eps_str = ""
        for opp in self.opponents:
            if hasattr(opp, "temperature"):
                eps_str = f" temp={opp.temperature:.2f}"
                break
            if hasattr(opp, "epsilon"):
                eps_str = f" opp_eps={opp.epsilon:.2f}"
                break
        log.info(
            "EVAL step=%d vs=%s win_rate=%.3f avg_turns=%.1f best_move=%.1f%% dmg_eff=%.2f"
            " vol_switch=%.1f%% forced_switch=%.1f%% expl_var=%.3f episodes=%d%s",
            self.num_timesteps,
            label,
            win_rate,
            avg_length,
            best_move_rate,
            dmg_efficiency,
            vol_switch_pct,
            forced_switch_pct,
            expl_var,
            n,
            eps_str,
        )
        # Verbose PPO internals to log file only
        if not np.isnan(entropy):
            log.debug(
                "PPO internals: entropy=%.3f approx_kl=%.4f pg_loss=%.4f value_loss=%.3f clip_frac=%.3f",
                entropy,
                approx_kl,
                pg_loss,
                value_loss,
                clip_frac,
            )

        if self.verbose:
            bar = "█" * int(win_rate * 20) + "░" * (20 - int(win_rate * 20))
            ev_str = f"  ev={expl_var:.2f}" if not np.isnan(expl_var) else ""
            # Per-opponent breakdown
            opp_parts = []
            for idx, lbl in enumerate(self._per_env_labels):
                if self._per_env_total[idx] > 0:
                    opp_wr = self._per_env_wins[idx] / self._per_env_total[idx] * 100
                    opp_parts.append(f"{lbl}={opp_wr:.0f}%")
            opp_str = f"  [{', '.join(opp_parts)}]" if opp_parts else ""
            print(
                f"  step {self.num_timesteps:>6,} │ "
                f"vs {label}: {win_rate * 100:>5.1f}%  [{bar}]  "
                f"avg {avg_length:.0f} turns  "
                f"bestmv {best_move_rate:.0f}%  dmgeff {dmg_efficiency:.2f}  "
                f"vol.sw {vol_switch_pct:.0f}%  forced {forced_switch_pct:.0f}%{ev_str}  "
                f"({n} eps){eps_str}{opp_str}"
            )

        # Update frozen self-play opponent every 50K steps
        if self.selfplay_path and self.num_timesteps - self._last_selfplay_update >= 50_000:
            self._last_selfplay_update = self.num_timesteps
            self.model.save(self.selfplay_path)
            for opp in self.opponents:
                if hasattr(opp, "swap_model"):
                    opp.swap_model(self.selfplay_path)
            log.info("SelfPlay: frozen opponent updated (step %d)", self.num_timesteps)
            if self.verbose:
                print(f"  [SelfPlay] Frozen opponent updated (step {self.num_timesteps})")
            # Save numbered snapshot for future fictitious self-play
            league_dir = Path(self.save_path) / "league"
            league_dir.mkdir(exist_ok=True)
            snapshot_num = self.num_timesteps // 1000
            snapshot_path = str(league_dir / f"snapshot_{snapshot_num:04d}")
            self.model.save(snapshot_path)
            log.info("SelfPlay: saved league/snapshot_%04d", snapshot_num)
            if self.verbose:
                print(f"  [SelfPlay] Saved league/snapshot_{snapshot_num:04d}")

        # Save best model -- keep timestamped copy + symlink-style "best_model"
        if win_rate > self._best_win_rate and n >= self.window // 2:
            self._best_win_rate = win_rate
            os.makedirs(self.save_path, exist_ok=True)
            # Timestamped snapshot (never overwritten)
            wr_tag = f"{win_rate:.3f}".replace(".", "")
            snapshot_name = f"best_wr{wr_tag}_step{self.num_timesteps}"
            self.model.save(os.path.join(self.save_path, snapshot_name))
            # Latest best (always points to the current best, for easy loading)
            self.model.save(os.path.join(self.save_path, "best_model"))
            log.info("New best win_rate=%.3f — saved %s", win_rate, snapshot_name)
            if self.verbose:
                print(f"[WinRate] New best {win_rate:.3f} — saved {snapshot_name}")

        # Save progress for resume
        if self.run_manager and self.phase_name:
            eps = None
            for opp in self.opponents:
                if hasattr(opp, "temperature"):
                    eps = opp.temperature
                    break
                if hasattr(opp, "epsilon"):
                    eps = opp.epsilon
                    break
            self.run_manager.save_progress(self.phase_name, self.num_timesteps, eps)

        # Write checkpoint metadata so tournament scripts know each model's win rate
        self._write_checkpoint_metadata(win_rate, avg_length)

        # Content logging
        self._update_content_log(
            win_rate,
            avg_length,
            action_pct,
            n,
            vol_switch_pct,
            forced_switch_pct,
            best_move_rate,
            dmg_efficiency,
            expl_var,
        )
        self._check_win_rate_milestones(win_rate, avg_length, action_pct)
        self._maybe_snapshot_replays(win_rate)

    def _update_content_log(
        self,
        win_rate,
        avg_length,
        action_pct,
        n,
        vol_switch_pct,
        forced_switch_pct,
        best_move_rate,
        dmg_efficiency,
        expl_var,
    ) -> None:
        """Append a row to content/training_log.md."""
        top_move = int(np.argmax(self._action_counts[6:])) + 1  # 1-indexed
        top_move_pct = action_pct[6 + np.argmax(self._action_counts[6:])]
        ev_str = f"{expl_var:>6.3f}" if not np.isnan(expl_var) else "   N/A"

        row = (
            f"| {self.num_timesteps:>7} "
            f"| {win_rate:.3f} "
            f"| {avg_length:>5.1f} "
            f"| {best_move_rate:>5.1f}% "
            f"| {dmg_efficiency:>5.2f} "
            f"| {ev_str} "
            f"| {vol_switch_pct:>5.1f}% "
            f"| {forced_switch_pct:>5.1f}% "
            f"| move_{top_move} ({top_move_pct:.1f}%) "
            f"| {n} |\n"
        )
        with open(self.content_log, "a", encoding="utf-8") as f:
            f.write(row)

    def _check_win_rate_milestones(self, win_rate, avg_length, action_pct) -> None:
        """Print milestone to console (hooks.md is maintained manually by [MEDIA])."""
        for threshold in _WIN_RATE_MILESTONES:
            if win_rate >= threshold and threshold not in self._crossed_milestones:
                self._crossed_milestones.add(threshold)
                label_pct = f"{int(threshold * 100)}%"
                log.info("MILESTONE: win rate crossed %s at step %d", label_pct, self.num_timesteps)
                if self.verbose:
                    print(f"  [Milestone] Win rate crossed {label_pct} at step {self.num_timesteps:,}")

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

        log.info("Snapped %d replays -> replays/notable/ (step %s, wr=%.3f)", len(replay_files), step_label, win_rate)
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
            with open(registry_path) as f:
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

    def _decay_shaping(self) -> None:
        """Linearly decay reward shaping from 1.0 to 0.0 over shaping_decay_battles (global)."""
        if not self._env or not self.shaping_decay_battles:
            return
        global_battles = self.global_episodes_offset + self._total_episodes
        progress = min(global_battles / self.shaping_decay_battles, 1.0)
        new_factor = max(1.0 - progress, 0.0)

        # Update shaping_factor on all env instances
        if hasattr(self._env, "envs"):
            for e in self._env.envs:
                # Navigate wrapper chain: Monitor -> SB3Wrapper -> SingleAgentWrapper -> Gen1Env
                inner = e
                while hasattr(inner, "env"):
                    inner = inner.env
                if hasattr(inner, "shaping_factor"):
                    inner.shaping_factor = new_factor
        if self._total_episodes % 200 == 0:
            log.info(
                "Shaping: factor=%.2f (battle %d/%d)", new_factor, self._total_episodes, self.shaping_decay_battles
            )
            if self.verbose:
                print(
                    f"  [Shaping] factor={new_factor:.2f} (battle {self._total_episodes}/{self.shaping_decay_battles})"
                )

    def _maybe_adjust_entropy(self, win_rate: float) -> None:
        """Increase entropy when stuck at low win rate, decrease when winning.

        Prevents the death spiral: when the agent always loses, PPO advantages
        are ~0 and the policy stops exploring. High entropy forces exploration.
        When the agent starts winning, lower entropy lets the policy sharpen.
        """
        if not hasattr(self.model, "ent_coef"):
            return
        ent = self.model.ent_coef
        if win_rate < 0.10 and ent < 0.1:
            new_ent = min(ent * 1.5, 0.1)
            self.model.ent_coef = new_ent
            log.info("Entropy: %.3f -> %.3f (win_rate=%.2f, stuck — forcing exploration)", ent, new_ent, win_rate)
            if self.verbose:
                print(f"  [Entropy] Stuck at {win_rate:.0%} — increased ent_coef to {new_ent:.3f}")
        elif win_rate > 0.50 and ent > 0.01:
            new_ent = max(ent * 0.9, 0.01)
            self.model.ent_coef = new_ent
            log.info("Entropy: %.3f -> %.3f (win_rate=%.2f, winning — sharpening)", ent, new_ent, win_rate)
            if self.verbose:
                print(f"  [Entropy] Winning at {win_rate:.0%} — decreased ent_coef to {new_ent:.3f}")

    def _maybe_anneal_epsilon(self, win_rate: float) -> None:
        """Set opponent epsilon/temperature as a continuous function of win rate.

        Searches all opponents for annealable attributes (temperature or epsilon)
        rather than assuming opponents[0] is the target. This supports mixed_league
        where opponents[0] is FrozenPolicyPlayer (no temperature/epsilon).
        """
        if not self.epsilon_schedule or not self.opponents:
            return
        eps_start, eps_end = self.epsilon_schedule
        if len(self._epsilon_rewards) < 500:
            return

        # Find annealable opponents by type
        temp_opps = [o for o in self.opponents if hasattr(o, "temperature")]
        eps_opps = [o for o in self.opponents if hasattr(o, "epsilon")]
        if not temp_opps and not eps_opps:
            return

        eps_win_rate = float(np.mean([1.0 if r >= 0.5 else 0.0 for r in self._epsilon_rewards]))

        if temp_opps:
            # Softmax: temperature = 2.0 at wr=0%, 0.1 at wr=100%
            current_temp = temp_opps[0].temperature
            target_temp = 2.0 * (1.0 - eps_win_rate) + 0.1 * eps_win_rate
            # Rate limit: max change 0.1 per eval
            new_temp = current_temp + max(min(target_temp - current_temp, 0.1), -0.1)
            if abs(new_temp - current_temp) > 0.01:
                for opp in temp_opps:
                    opp.temperature = new_temp
                log.info("Curriculum: temperature %.2f -> %.2f (wr500=%.2f)", current_temp, new_temp, eps_win_rate)
                if self.verbose:
                    print(f"  [Curriculum] temperature: {current_temp:.2f} → {new_temp:.2f} (wr500={eps_win_rate:.2f})")
        elif eps_opps:
            current_eps = eps_opps[0].epsilon
            target_eps = max(min(1.0 - eps_win_rate, eps_start), eps_end)
            # Clamp: max drop 0.03, max rise 0.05
            new_eps = max(target_eps, current_eps - 0.03)
            new_eps = min(new_eps, max(target_eps, current_eps + 0.05))
            if abs(new_eps - current_eps) > 0.005:
                for opp in eps_opps:
                    opp.epsilon = new_eps
                log.info("Curriculum: epsilon %.2f -> %.2f (wr500=%.2f)", current_eps, new_eps, eps_win_rate)
                if self.verbose:
                    print(f"  [Curriculum] epsilon: {current_eps:.2f} → {new_eps:.2f} (wr500={eps_win_rate:.2f})")

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
            f"Target: {target * 100:.0f}% win rate\n\n"
            f"| Step    | Win Rate | Avg Turns | BestMv% | DmgEff | ExplVar | Vol.Sw% | Forced% | Top Move         | Episodes |\n"
            f"|---------|----------|-----------|---------|--------|---------|---------|---------|------------------|----------|\n"
        )
        mode = "a" if self.content_log.exists() else "w"
        with open(self.content_log, mode, encoding="utf-8") as f:
            if mode == "w":
                f.write(f"# {self.run_id} Training Log\n")
            f.write(header)
