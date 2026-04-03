"""Generate behavioral cloning data from a heuristic teacher.

Runs the teacher agent against one or more opponents, recording
(observation, action, action_mask) at every turn. Saves the dataset
for supervised pre-training of the RL policy.

Teachers:
  maxdamage    — MaxDamagePlayer (original, never switches)
  smart        — SmartHeuristicPlayer (MaxDamage moves + competitive switching)

Opponents: random, maxdamage, typematchup, aggressive_switcher, smart, mixed (all)

Usage:
    python scripts/generate_bc_data.py                         # smart vs mixed (default)
    python scripts/generate_bc_data.py --teacher maxdamage     # original BC
    python scripts/generate_bc_data.py --n-battles 10000
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


class DataCollectingPlayer:
    """Mixin that records (obs, action, mask) at every turn."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observations: list[np.ndarray] = []
        self.actions: list[int] = []
        self.masks: list[np.ndarray] = []

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

        return order


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


async def collect(n_battles: int, teacher: str, opponent_name: str, port: int = _BASE_PORT) -> dict:
    import random as _rnd
    import string as _str

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

    print(f"Collecting data: {teacher} vs {opponent_name} ({n_battles} games)...")

    async def _log_progress():
        start = _time.time()
        last_count = 0
        while True:
            await asyncio.sleep(5)
            done = collector.n_finished_battles
            transitions = len(collector.observations)
            elapsed = _time.time() - start
            rate = done / elapsed if elapsed > 0 else 0
            if done != last_count:
                print(
                    f"  [{opponent_name}] {done}/{n_battles} battles "
                    f"({transitions:,} transitions, {rate:.1f} battles/s)"
                )
                last_count = done
            if done >= n_battles:
                break

    progress_task = asyncio.create_task(_log_progress())
    await collector.battle_against(opponent, n_battles=n_battles)
    progress_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await progress_task

    wins = collector.n_won_battles
    win_pct = (wins / n_battles * 100) if n_battles > 0 else 0.0
    print(f"  [{opponent_name}] Done: {wins}/{n_battles} wins ({win_pct:.1f}%)")
    print(f"  [{opponent_name}] Transitions collected: {len(collector.observations)}")

    return {
        "observations": np.array(collector.observations),
        "actions": np.array(collector.actions, dtype=np.int64),
        "masks": np.array(collector.masks),
    }


# ---------------------------------------------------------------------------
# Multiprocessing workers
# ---------------------------------------------------------------------------


def _collect_worker(args: tuple) -> dict:
    """Run in a separate process: start server, collect data, stop server."""
    n_battles, teacher, opponent_name, port = args
    server = _start_server(port)
    try:
        return asyncio.run(collect(n_battles, teacher, opponent_name, port))
    finally:
        _stop_server(server)


def collect_mixed(n_battles: int, teacher: str) -> dict:
    """Collect data against ALL opponents in parallel processes, one Showdown server each."""
    per_opp = n_battles // len(_OPPONENT_CLS)
    opp_names = list(_OPPONENT_CLS.keys())

    work = [(per_opp, teacher, name, _BASE_PORT + 1 + i) for i, name in enumerate(opp_names)]

    print(f"Spawning {len(opp_names)} workers (each with its own Showdown server)...")
    with ProcessPoolExecutor(max_workers=len(opp_names)) as executor:
        results = list(executor.map(_collect_worker, work))

    results = [r for r in results if len(r["observations"]) > 0]
    if not results:
        raise RuntimeError("All workers returned empty results")

    return {
        "observations": np.concatenate([r["observations"] for r in results]),
        "actions": np.concatenate([r["actions"] for r in results]),
        "masks": np.concatenate([r["masks"] for r in results]),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-battles", type=int, default=5000)
    parser.add_argument("--teacher", choices=list(_TEACHER_CLS), default="smart")
    parser.add_argument("--opponent", choices=[*_OPPONENT_CLS, "mixed"], default="mixed")
    parser.add_argument("--port", type=int, default=_BASE_PORT, help="Showdown server port (single-opponent mode)")
    args = parser.parse_args()

    if args.opponent == "mixed":
        data = collect_mixed(args.n_battles, args.teacher)
    else:
        server = _start_server(args.port)
        try:
            data = asyncio.run(collect(args.n_battles, args.teacher, args.opponent, args.port))
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

    # Save
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "bc_training_data.npz"
    np.savez_compressed(out_path, **data)
    print(f"\nSaved to {out_path} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
