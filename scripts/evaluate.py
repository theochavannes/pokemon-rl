"""
Evaluate a trained model against the standard heuristic opponents.

Requires the Showdown server to be running:
    cd showdown && node pokemon-showdown start --no-security

Usage:
    python scripts/evaluate.py --model models/best_model_v1.zip
    python scripts/evaluate.py --model runs/run_057/models/best_wr0650_step355164.zip --n 100
    python scripts/evaluate.py --model models/best_model_v1.zip --opponents maxdamage typematch
"""

import argparse
import asyncio
import random
import string
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import os

os.chdir(Path(__file__).parent.parent)

from poke_env.ps_client.account_configuration import AccountConfiguration

from src.agents.heuristic_agent import MaxDamagePlayer, SoftmaxDamagePlayer, TypeMatchupPlayer
from src.obs_transfer import obs_dim_of

BATTLE_FORMAT = "gen1randombattle"

OPPONENT_LABELS = {
    "maxdamage": "MaxDamage",
    "typematch": "TypeMatchup",
    "softmax": "SoftmaxDamage",
    "random": "Random",
}


def _suffix() -> str:
    return "".join(random.choices(string.ascii_lowercase, k=6))


def _make_heuristic(name: str, idx: int):
    account = AccountConfiguration(f"Eval{idx}{_suffix()}", None)
    kwargs = dict(battle_format=BATTLE_FORMAT, account_configuration=account, log_level=40)
    if name == "maxdamage":
        return MaxDamagePlayer(**kwargs)
    if name == "typematch":
        return TypeMatchupPlayer(**kwargs)
    if name == "softmax":
        return SoftmaxDamagePlayer(temperature=1.0, **kwargs)
    if name == "random":
        from poke_env.player.baselines import RandomPlayer

        return RandomPlayer(**kwargs)
    raise ValueError(f"Unknown opponent: {name}")


async def _run_one(model_path: str, opp_name: str, n: int, idx: int) -> dict:
    from src.agents.policy_player import FrozenPolicyPlayer

    agent_account = AccountConfiguration(f"Agent{idx}{_suffix()}", None)
    agent = FrozenPolicyPlayer(
        model_path=model_path,
        battle_format=BATTLE_FORMAT,
        account_configuration=agent_account,
        log_level=40,
    )
    opp = _make_heuristic(opp_name, idx)

    await agent.battle_against(opp, n_battles=n)

    total = agent.n_won_battles + opp.n_won_battles
    win_rate = agent.n_won_battles / total if total else 0.0
    return {"opponent": opp_name, "wins": agent.n_won_battles, "total": total, "win_rate": win_rate}


async def evaluate(model_path: str, opponents: list[str], n: int) -> None:
    path = Path(model_path)
    if not path.exists():
        sys.exit(f"Model not found: {model_path}")

    # Strip .zip for SB3 load
    model_str = str(path.with_suffix("")) if path.suffix == ".zip" else str(path)

    obs_dim = obs_dim_of(model_str)
    print(f"\nModel:   {path.name}  (obs_dim={obs_dim})")
    print(f"Battles: {n} per opponent  |  format: {BATTLE_FORMAT}\n")

    results = []
    for i, opp in enumerate(opponents):
        label = OPPONENT_LABELS.get(opp, opp)
        print(f"  Running vs {label}...", end="", flush=True)
        r = await _run_one(model_str, opp, n, i)
        results.append(r)
        print(f"  {r['win_rate'] * 100:.0f}%")

    # Print table
    sep = "-" * 48
    print(f"\n{sep}")
    print(f"  {'Opponent':<18}  {'Win Rate':>8}  {'Wins':>5}  {'Games':>5}")
    print(sep)
    heuristic_wrs = []
    for r in results:
        label = OPPONENT_LABELS.get(r["opponent"], r["opponent"])
        wr_pct = r["win_rate"] * 100
        print(f"  {label:<18}  {wr_pct:>7.1f}%  {r['wins']:>5}  {r['total']:>5}")
        if r["opponent"] != "random":
            heuristic_wrs.append(r["win_rate"])
    print(sep)
    if heuristic_wrs:
        mean_wr = sum(heuristic_wrs) / len(heuristic_wrs) * 100
        print(f"  {'vs heuristics (mean)':<18}  {mean_wr:>7.1f}%")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Pokemon RL model vs heuristic opponents")
    parser.add_argument("--model", required=True, help="Path to model checkpoint (.zip)")
    parser.add_argument("--n", type=int, default=50, help="Battles per opponent (default: 50)")
    parser.add_argument(
        "--opponents",
        nargs="+",
        choices=list(OPPONENT_LABELS),
        default=["maxdamage", "typematch", "softmax", "random"],
        help="Opponents to evaluate against (default: all)",
    )
    args = parser.parse_args()
    asyncio.run(evaluate(args.model, args.opponents, args.n))


if __name__ == "__main__":
    main()
