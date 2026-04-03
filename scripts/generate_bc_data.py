"""Generate behavioral cloning data from a heuristic teacher.

Runs the teacher agent against one or more opponents, recording
(observation, action, action_mask) at every turn. Saves batches to disk
incrementally so progress survives crashes. Supports resuming.

Teachers:
  maxdamage    — MaxDamagePlayer (original, never switches)
  smart        — SmartHeuristicPlayer (MaxDamage moves + competitive switching)

Opponents: random, maxdamage, typematchup, aggressive_switcher, smart, mixed (all)

Usage:
    python scripts/generate_bc_data.py                         # smart vs mixed (default)
    python scripts/generate_bc_data.py --teacher maxdamage     # original BC
    python scripts/generate_bc_data.py --n-battles 10000
    python scripts/generate_bc_data.py --resume                # resume interrupted run
    python scripts/generate_bc_data.py --clean                 # delete batches, start fresh
"""

import argparse
import asyncio
import contextlib
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import time as _time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

log = logging.getLogger(__name__)

import numpy as np
from poke_env.environment.singles_env import SinglesEnv
from poke_env.player.baselines import RandomPlayer
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import ServerConfiguration

# Project root
_ROOT = Path(__file__).parent.parent
os.chdir(_ROOT)
sys.path.insert(0, str(_ROOT))

from src.agents.heuristic_agent import (
    AggressiveSwitcher,
    MaxDamagePlayer,
    SmartHeuristicPlayer,
    TypeMatchupPlayer,
)
from src.env.gen1_env import embed_battle

BATTLE_FORMAT = "gen1randombattle"
_BASE_PORT = 8000
_BATCH_DIR = Path("data/bc_batches")
_BATCH_SIZE = 200  # save to disk every N battles

_TEACHER_CLS = {
    "maxdamage": MaxDamagePlayer,
    "smart": SmartHeuristicPlayer,
}

_OPPONENT_CLS = {
    "random": RandomPlayer,
    "maxdamage": MaxDamagePlayer,
    "typematchup": TypeMatchupPlayer,
    "aggressive_switcher": AggressiveSwitcher,
    "smart": SmartHeuristicPlayer,
}


# ---------------------------------------------------------------------------
# Batch I/O
# ---------------------------------------------------------------------------


def _batch_path(opponent_name: str, batch_idx: int) -> Path:
    return _BATCH_DIR / f"{opponent_name}_{batch_idx:04d}.npz"


def _count_existing_battles(opponent_name: str) -> tuple[int, int]:
    """Return (n_battles_completed, next_batch_idx) from saved batches."""
    total_battles = 0
    max_idx = -1
    for f in sorted(_BATCH_DIR.glob(f"{opponent_name}_*.npz")):
        try:
            meta = np.load(f)
            total_battles += int(meta.get("n_battles", len(meta["actions"])))
            idx = int(f.stem.split("_")[-1])
            max_idx = max(max_idx, idx)
        except Exception:  # noqa: S112
            continue
    return total_battles, max_idx + 1


def _save_batch(opponent_name: str, batch_idx: int, obs: list, actions: list, masks: list, n_battles: int):
    """Save one batch of transitions to disk."""
    _BATCH_DIR.mkdir(parents=True, exist_ok=True)
    path = _batch_path(opponent_name, batch_idx)
    np.savez_compressed(
        path,
        observations=np.array(obs),
        actions=np.array(actions, dtype=np.int64),
        masks=np.array(masks),
        n_battles=np.array(n_battles),
    )
    return path


def _merge_batches(opponent_names: list[str] | None = None) -> dict:
    """Load all batches (optionally filtered by opponent) and merge."""
    all_obs, all_acts, all_masks = [], [], []
    pattern = "*.npz"
    for f in sorted(_BATCH_DIR.glob(pattern)):
        if opponent_names:
            # filename is {opponent}_{idx}.npz
            opp = "_".join(f.stem.split("_")[:-1])
            if opp not in opponent_names:
                continue
        data = np.load(f)
        if len(data["actions"]) > 0:
            all_obs.append(data["observations"])
            all_acts.append(data["actions"])
            all_masks.append(data["masks"])

    if not all_obs:
        raise RuntimeError("No batch data found")

    return {
        "observations": np.concatenate(all_obs),
        "actions": np.concatenate(all_acts),
        "masks": np.concatenate(all_masks),
    }


class DataCollectingPlayer:
    """Mixin that records (obs, action, mask) at every turn and flushes to disk."""

    def __init__(self, *args, opponent_name: str = "unknown", batch_start_idx: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.observations: list[np.ndarray] = []
        self.actions: list[int] = []
        self.masks: list[np.ndarray] = []
        self._opponent_name = opponent_name
        self._batch_idx = batch_start_idx
        self._last_flush_battles = 0

    def choose_move(self, battle):
        order = super().choose_move(battle)

        # Only record during real move selection (not teampreview)
        if battle.active_pokemon is not None and not battle.teampreview:
            try:
                obs = embed_battle(battle)
                mask = np.array(SinglesEnv.get_action_mask(battle), dtype=bool)
                action = int(SinglesEnv.order_to_action(order, battle, strict=False))

                # Sanity: action should be within mask
                if 0 <= action < len(mask) and mask[action]:
                    self.observations.append(obs)
                    self.actions.append(action)
                    self.masks.append(mask)
            except Exception as e:
                log.debug("BC data collection skip: %s", e)

        # Flush batch to disk periodically
        battles_since_flush = self.n_finished_battles - self._last_flush_battles
        if battles_since_flush >= _BATCH_SIZE and self.observations:
            self._flush()

        return order

    def _flush(self):
        """Write current buffer to disk and clear it."""
        if not self.observations:
            return
        n_battles = self.n_finished_battles - self._last_flush_battles
        path = _save_batch(
            self._opponent_name,
            self._batch_idx,
            self.observations,
            self.actions,
            self.masks,
            n_battles,
        )
        print(
            f"  [{self._opponent_name}] Saved batch {self._batch_idx} "
            f"({len(self.observations)} transitions, {n_battles} battles) → {path.name}"
        )
        self._batch_idx += 1
        self._last_flush_battles = self.n_finished_battles
        self.observations.clear()
        self.actions.clear()
        self.masks.clear()

    def flush_remaining(self):
        """Flush any leftover transitions after collection ends."""
        if self.observations:
            self._flush()


def _make_collector_cls(base_cls):
    """Dynamically create a DataCollecting version of any Player subclass."""
    return type(f"DataCollecting{base_cls.__name__}", (DataCollectingPlayer, base_cls), {})


# ---------------------------------------------------------------------------
# Showdown server lifecycle
# ---------------------------------------------------------------------------


def _start_server(port: int) -> subprocess.Popen:
    """Start a Showdown server on the given port."""
    node = shutil.which("node")
    showdown_dir = _ROOT / "showdown"
    proc = subprocess.Popen(  # noqa: S603
        [node, "pokemon-showdown", "start", "--no-security", str(port)],
        cwd=str(showdown_dir),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )
    for _ in range(60):
        try:
            s = socket.create_connection(("localhost", port), timeout=1)
            s.close()
            _time.sleep(0.5)  # Allow WebSocket handler to initialize
            return proc
        except OSError:
            _time.sleep(0.25)
    proc.kill()
    raise RuntimeError(f"Showdown server on port {port} failed to start")


def _stop_server(proc: subprocess.Popen):
    """Stop a Showdown server process."""
    if proc.poll() is None:
        if sys.platform == "win32":
            proc.send_signal(signal.CTRL_BREAK_EVENT)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        else:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------


async def collect(
    n_battles: int, teacher: str, opponent_name: str, port: int = _BASE_PORT, resume: bool = False
) -> int:
    """Collect BC data, flushing to disk in batches. Returns total battles completed."""
    import random as _rnd
    import string as _str

    # Check for existing progress
    done_battles = 0
    batch_start_idx = 0
    if resume:
        done_battles, batch_start_idx = _count_existing_battles(opponent_name)
        if done_battles >= n_battles:
            print(f"  [{opponent_name}] Already complete ({done_battles}/{n_battles} battles)")
            return done_battles
        if done_battles > 0:
            print(f"  [{opponent_name}] Resuming: {done_battles}/{n_battles} battles already saved")

    remaining = n_battles - done_battles

    server_config = ServerConfiguration(
        f"ws://localhost:{port}/showdown/websocket",
        "https://play.pokemonshowdown.com/action.php?",
    )
    suffix = "".join(_rnd.choices(_str.ascii_lowercase, k=6))
    collector_cls = _make_collector_cls(_TEACHER_CLS[teacher])
    collector = collector_cls(
        battle_format=BATTLE_FORMAT,
        account_configuration=AccountConfiguration(f"BC{suffix}", None),
        server_configuration=server_config,
        max_concurrent_battles=100,
        ping_interval=None,
        ping_timeout=None,
        log_level=40,
        opponent_name=opponent_name,
        batch_start_idx=batch_start_idx,
    )
    opp_cls = _OPPONENT_CLS[opponent_name]
    opponent = opp_cls(
        battle_format=BATTLE_FORMAT,
        account_configuration=AccountConfiguration(f"BCOpp{suffix}", None),
        server_configuration=server_config,
        max_concurrent_battles=100,
        ping_interval=None,
        ping_timeout=None,
        log_level=40,
    )

    print(f"Collecting data: {teacher} vs {opponent_name} ({remaining} games)...")

    async def _log_progress():
        start = _time.time()
        last_count = 0
        while True:
            await asyncio.sleep(5)
            done = collector.n_finished_battles
            # transitions = in-memory buffer (unflushed) — total is on disk + buffer
            elapsed = _time.time() - start
            rate = done / elapsed if elapsed > 0 else 0
            if done != last_count:
                print(f"  [{opponent_name}] {done + done_battles}/{n_battles} battles ({rate:.1f} battles/s)")
                last_count = done
            if done >= remaining:
                break

    progress_task = asyncio.create_task(_log_progress())
    try:
        await collector.battle_against(opponent, n_battles=remaining)
    except (ConnectionResetError, OSError) as e:
        print(f"  [{opponent_name}] Connection lost: {e}")
    finally:
        # Always flush remaining data before exiting
        collector.flush_remaining()
        progress_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await progress_task

    total = done_battles + collector.n_finished_battles
    wins = collector.n_won_battles
    finished = collector.n_finished_battles
    win_pct = (wins / finished * 100) if finished > 0 else 0.0
    print(f"  [{opponent_name}] Done: {wins}/{finished} wins ({win_pct:.1f}%)")
    print(f"  [{opponent_name}] Total battles saved: {total}/{n_battles}")
    return total


# ---------------------------------------------------------------------------
# Multiprocessing workers
# ---------------------------------------------------------------------------


def _collect_worker(args: tuple) -> tuple[str, int]:
    """Run in a separate process: start server, collect data, stop server.
    Returns (opponent_name, total_battles_completed)."""
    n_battles, teacher, opponent_name, port, resume = args
    server = _start_server(port)
    try:
        total = asyncio.run(collect(n_battles, teacher, opponent_name, port, resume))
        return opponent_name, total
    except Exception as e:
        print(f"  [{opponent_name}] Worker failed: {e}")
        return opponent_name, 0
    finally:
        _stop_server(server)


def collect_mixed(n_battles: int, teacher: str, resume: bool = False) -> dict:
    """Collect data against ALL opponents in parallel processes."""
    per_opp = n_battles // len(_OPPONENT_CLS)
    opp_names = list(_OPPONENT_CLS.keys())

    work = [(per_opp, teacher, name, _BASE_PORT + 1 + i, resume) for i, name in enumerate(opp_names)]

    print(f"Spawning {len(opp_names)} workers (each with its own Showdown server)...")
    with ProcessPoolExecutor(max_workers=len(opp_names)) as executor:
        results = list(executor.map(_collect_worker, work))

    for name, total in results:
        print(f"  {name}: {total} battles")

    return _merge_batches(opp_names)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-battles", type=int, default=5000)
    parser.add_argument("--teacher", choices=list(_TEACHER_CLS), default="smart")
    parser.add_argument("--opponent", choices=[*_OPPONENT_CLS, "mixed"], default="mixed")
    parser.add_argument("--port", type=int, default=_BASE_PORT, help="Showdown server port (single-opponent mode)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing batches instead of starting fresh")
    parser.add_argument("--clean", action="store_true", help="Delete existing batches before starting")
    args = parser.parse_args()

    if args.clean and _BATCH_DIR.exists():
        shutil.rmtree(_BATCH_DIR)
        print(f"Cleaned {_BATCH_DIR}")

    _BATCH_DIR.mkdir(parents=True, exist_ok=True)

    if args.opponent == "mixed":
        data = collect_mixed(args.n_battles, args.teacher, args.resume)
    else:
        server = _start_server(args.port)
        try:
            asyncio.run(collect(args.n_battles, args.teacher, args.opponent, args.port, args.resume))
            opp_list = [args.opponent]
            data = _merge_batches(opp_list)
        finally:
            _stop_server(server)

    obs = data["observations"]
    actions = data["actions"]

    # Stats
    n = len(actions)
    print(f"\nDataset: {n} transitions, obs shape {obs.shape}")
    print("  Action distribution:")
    for i in range(10):
        count = (actions == i).sum()
        label = f"switch_{i}" if i < 6 else f"move_{i - 5}"
        print(f"    {label}: {count:>6} ({count / n * 100:>5.1f}%)")

    move_mask = actions >= 6
    if move_mask.any():
        print(f"\n  Move actions: {move_mask.sum()} ({move_mask.sum() / n * 100:.1f}%)")
        print(f"  Switch actions: {(~move_mask).sum()} ({(~move_mask).sum() / n * 100:.1f}%)")

    # Save merged file for BC training
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "bc_training_data.npz"
    np.savez_compressed(out_path, **data)
    print(f"\nSaved to {out_path} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
