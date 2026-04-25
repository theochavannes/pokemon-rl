"""Evaluate pokemon_rl agents vs Metamon Kakuna (Gen 1 OU neural policy).

Uses a two-process architecture: Kakuna runs in the metamon_ft conda env
(poke-env 0.8.x + metamon); our agents run in the current env (poke-env 0.13.0).
Both connect to the same local Showdown server via send_challenges / accept_challenges.

Requires:
  - Showdown server running: cd showdown && node pokemon-showdown start --no-security
  - metamon_ft conda env with metamon installed
  - metamon-format-transfer repo at ~/code/metamon-format-transfer (or --mft_repo)

Usage:
    python scripts/eval_vs_kakuna.py
    python scripts/eval_vs_kakuna.py --n 100 --opponents frozen maxdamage
    python scripts/eval_vs_kakuna.py --metamon_python /path/to/metamon_ft/python.exe

Resume after crash:
    Re-run the same command — checkpoint at data/eval_vs_kakuna_checkpoint.json
    is loaded automatically and skips already-completed battles.
"""

import argparse
import asyncio
import json
import os
import queue
import random
import string
import subprocess
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
os.chdir(Path(__file__).parent.parent)

from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import ServerConfiguration

BATTLE_FORMAT = "gen1randombattle"
MODEL_PATH = "models/best_model_v1"

_METAMON_PYTHON_DEFAULT = str(
    Path.home() / "miniconda3" / "envs" / "metamon_ft" / ("python.exe" if sys.platform == "win32" else "bin/python")
)
_METAMON_CACHE_DEFAULT = str(Path.home() / ".metamon")
_KAKUNA_RUNNER = str(Path(__file__).parent / "kakuna_runner.py")
_DEFAULT_CHECKPOINT = "data/eval_vs_kakuna_checkpoint.json"


def _suffix() -> str:
    return "".join(random.choices(string.ascii_lowercase, k=6))


def _server_cfg(port: int):
    if port == 8000:
        return None
    return ServerConfiguration(
        f"ws://localhost:{port}/showdown/websocket",
        "https://play.pokemonshowdown.com/action.php?",
    )


def _make_frozen(username: str, server_cfg=None):
    from src.agents.policy_player import FrozenPolicyPlayer

    account = AccountConfiguration(username, None)
    kwargs: dict = dict(
        model_path=MODEL_PATH,
        battle_format=BATTLE_FORMAT,
        account_configuration=account,
        log_level=40,
    )
    if server_cfg is not None:
        kwargs["server_configuration"] = server_cfg
    return FrozenPolicyPlayer(**kwargs)


def _make_maxdamage(username: str, server_cfg=None):
    from src.agents.heuristic_agent import MaxDamagePlayer

    account = AccountConfiguration(username, None)
    kwargs: dict = dict(
        battle_format=BATTLE_FORMAT,
        account_configuration=account,
        log_level=40,
    )
    if server_cfg is not None:
        kwargs["server_configuration"] = server_cfg
    return MaxDamagePlayer(**kwargs)


def _load_checkpoint(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_checkpoint(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def _start_kakuna(
    metamon_python: str,
    kakuna_name: str,
    challenger_name: str,
    n: int,
    mft_repo: str,
    server_port: int,
    metamon_cache: str,
) -> tuple:
    cmd = [
        metamon_python,
        _KAKUNA_RUNNER,
        "--username",
        kakuna_name,
        "--challenger",
        challenger_name,
        "--n",
        str(n),
        "--mft_repo",
        mft_repo,
        "--metamon_cache",
        metamon_cache,
        "--server_port",
        str(server_port),
    ]
    os.makedirs("data", exist_ok=True)
    # Open stderr log with context manager; _reader closes it after stdout EOF
    stderr_log = Path(f"data/kakuna_{kakuna_name}.log").open("w")  # noqa: SIM115
    proc = subprocess.Popen(  # noqa: S603
        cmd,
        stdout=subprocess.PIPE,
        stderr=stderr_log,  # redirect to file — prevents 64KB pipe buffer deadlock
        text=True,
        bufsize=1,
        cwd=str(Path(__file__).parent.parent),
    )
    line_q: queue.Queue = queue.Queue()

    def _reader():
        for line in proc.stdout:
            line_q.put(line.strip())
        proc.stdout.close()
        stderr_log.close()

    threading.Thread(target=_reader, daemon=True).start()
    return proc, line_q


async def _run_batch(
    challenger,
    kakuna_name: str,
    proc: subprocess.Popen,
    line_q: queue.Queue,
    batch_n: int,
    ready_timeout: int = 120,
) -> tuple[int, int]:
    """Run one batch of battles. Returns (challenger_wins, kakuna_wins)."""
    loop = asyncio.get_event_loop()
    deadline = loop.time() + ready_timeout
    while True:
        try:
            line = line_q.get_nowait()
            if line == "READY":
                break
        except queue.Empty:
            if loop.time() > deadline:
                proc.kill()
                raise RuntimeError(
                    f"Kakuna subprocess timed out waiting for READY "
                    f"(>{ready_timeout}s). Check data/kakuna_{kakuna_name}.log"
                ) from None
            await asyncio.sleep(0.25)

    wins_before = challenger.n_won_battles
    losses_before = challenger.n_lost_battles

    await challenger.send_challenges(kakuna_name, n_challenges=batch_n)

    result_data: dict | None = None
    for _ in range(120):
        try:
            line = line_q.get_nowait()
            if line.startswith("{"):
                result_data = json.loads(line)
                break
        except queue.Empty:
            await asyncio.sleep(0.25)

    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()

    if result_data is not None:
        # Kakuna reports from its own perspective: wins = kakuna wins
        kakuna_wins = result_data["wins"]
        challenger_wins = result_data["losses"]
    else:
        # Fallback: infer from challenger's own counters
        challenger_wins = challenger.n_won_battles - wins_before
        kakuna_wins = challenger.n_lost_battles - losses_before

    return challenger_wins, kakuna_wins


def _print_progress(label: str, wins: int, losses: int, target: int) -> None:
    total = wins + losses
    wr = wins / total if total else 0.0
    bar_width = 20
    filled = int(bar_width * total / target) if target else 0
    bar = "#" * filled + "." * (bar_width - filled)
    print(
        f"  [{bar}] {total:>3}/{target}  {label:<20}  WR={wr * 100:5.1f}%  ({wins}W / {losses}L)",
        flush=True,
    )


async def main(args) -> None:
    mft_repo = os.path.expanduser(args.mft_repo)
    scfg = _server_cfg(args.server_port)

    checkpoint = _load_checkpoint(args.checkpoint)
    results: dict[str, dict] = {k: dict(v) for k, v in checkpoint.items()}

    opponent_cfgs: list[tuple[str, str, object]] = []
    if "frozen" in args.opponents:
        opponent_cfgs.append(("FrozenPolicy", "Frozen", lambda u: _make_frozen(u, scfg)))
    if "maxdamage" in args.opponents:
        opponent_cfgs.append(("MaxDamage", "MaxDmg", lambda u: _make_maxdamage(u, scfg)))

    for label, prefix, make_challenger in opponent_cfgs:
        prior = results.get(label, {"wins": 0, "losses": 0})
        done = prior["wins"] + prior["losses"]
        remaining = args.n - done

        if remaining <= 0:
            print(f"\n{label}: already complete ({done}/{args.n} from checkpoint)")
            _print_progress(label, prior["wins"], prior["losses"], args.n)
            continue

        sfx = _suffix()
        challenger_name = f"{prefix}{sfx}"
        challenger = make_challenger(challenger_name)

        cum_wins = prior["wins"]
        cum_losses = prior["losses"]

        print(f"\n{'=' * 60}")
        print(f"  {label} vs Kakuna  |  resuming from {done}/{args.n}")
        print(f"{'=' * 60}")

        while remaining > 0:
            batch_n = min(args.batch_size, remaining)
            kakuna_name = f"Kakuna{_suffix()}"

            proc, line_q = _start_kakuna(
                args.metamon_python,
                kakuna_name,
                challenger_name,
                batch_n,
                mft_repo,
                args.server_port,
                args.metamon_cache,
            )

            try:
                bw, bl = await _run_batch(challenger, kakuna_name, proc, line_q, batch_n)
            except Exception as e:
                proc.kill()
                total_done = cum_wins + cum_losses
                print(f"\n  Batch failed: {e}")
                print(f"  Progress saved at {total_done}/{args.n} battles. Re-run to resume.")
                results[label] = {"wins": cum_wins, "losses": cum_losses}
                _save_checkpoint(args.checkpoint, results)
                break

            resolved = bw + bl
            cum_wins += bw
            cum_losses += bl
            remaining -= resolved

            results[label] = {"wins": cum_wins, "losses": cum_losses}
            _save_checkpoint(args.checkpoint, results)
            _print_progress(label, cum_wins, cum_losses, args.n)

    # Final summary
    sep = "-" * 62
    print(f"\n{sep}")
    print(f"  {'Challenger':<20}  {'WR vs Kakuna':>12}  {'W':>5}  {'L':>5}  {'N':>5}")
    print(sep)
    for lbl, r in results.items():
        total = r["wins"] + r["losses"]
        wr = r["wins"] / total if total else 0.0
        print(f"  {lbl:<20}  {wr * 100:>11.1f}%  {r['wins']:>5}  {r['losses']:>5}  {total:>5}")
    print(sep)
    print(f"  Format: {BATTLE_FORMAT} | Target N={args.n}/matchup | Batch={args.batch_size}")
    print()

    all_labels = [lbl for lbl, _, _ in opponent_cfgs]
    all_done = all(
        results.get(lbl, {}).get("wins", 0) + results.get(lbl, {}).get("losses", 0) >= args.n for lbl in all_labels
    )
    if all_done and os.path.exists(args.checkpoint):
        os.remove(args.checkpoint)
        print("Checkpoint removed — eval complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="Total battles per opponent (default: 50)")
    parser.add_argument("--batch_size", type=int, default=10, help="Battles per checkpoint batch (default: 10)")
    parser.add_argument(
        "--opponents",
        nargs="+",
        choices=["frozen", "maxdamage"],
        default=["frozen", "maxdamage"],
    )
    parser.add_argument("--mft_repo", default="~/code/metamon-format-transfer")
    parser.add_argument("--server_port", type=int, default=8000)
    parser.add_argument(
        "--metamon_python",
        default=_METAMON_PYTHON_DEFAULT,
        help="Python executable for the metamon_ft conda env",
    )
    parser.add_argument(
        "--metamon_cache",
        default=_METAMON_CACHE_DEFAULT,
        help="METAMON_CACHE_DIR (default: ~/.metamon)",
    )
    parser.add_argument(
        "--checkpoint",
        default=_DEFAULT_CHECKPOINT,
        help="Checkpoint file for resume (default: data/eval_vs_kakuna_checkpoint.json)",
    )
    args = parser.parse_args()
    asyncio.run(main(args))
