"""
Benchmark any agent against the mixed_league opponent pool.

Usage (Showdown server must be running):
    python scripts/benchmark_league.py --agent maxdamage --n 100
    python scripts/benchmark_league.py --agent models/best_model --n 100
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

BATTLE_FORMAT = "gen1randombattle"


def _make_account(prefix):
    suffix = "".join(random.choices(string.ascii_lowercase, k=6))
    from poke_env.ps_client.account_configuration import AccountConfiguration

    return AccountConfiguration(f"{prefix}{suffix}", None)


def _make_agent(name, prefix="Agent"):
    kw = dict(battle_format=BATTLE_FORMAT, account_configuration=_make_account(prefix), log_level=40)

    if name.lower() == "maxdamage":
        from src.agents.heuristic_agent import MaxDamagePlayer

        return MaxDamagePlayer(**kw)
    elif name.lower() == "random":
        from poke_env.player.baselines import RandomPlayer

        return RandomPlayer(**kw)
    elif name.lower() == "typematchup":
        from src.agents.heuristic_agent import TypeMatchupPlayer

        return TypeMatchupPlayer(**kw)
    elif name.lower() == "softmax_low":
        from src.agents.heuristic_agent import SoftmaxDamagePlayer

        return SoftmaxDamagePlayer(temperature=1.0, **kw)
    elif name.lower() == "softmax_high":
        from src.agents.heuristic_agent import SoftmaxDamagePlayer

        return SoftmaxDamagePlayer(temperature=2.0, **kw)
    else:
        from src.agents.policy_player import FrozenPolicyPlayer

        p = Path(name)
        if not p.suffix:
            p = p.with_suffix(".zip")
        if not p.exists():
            p = Path("models") / p
        if not p.exists():
            raise FileNotFoundError(f"Model not found: {name}")
        return FrozenPolicyPlayer(model_path=str(p.with_suffix("")), **kw)


async def benchmark(agent_name: str, n_battles: int):
    opponents = ["maxdamage", "typematchup", "softmax_low"]

    print(f"\nBenchmark: {agent_name} vs league opponents ({n_battles} battles each)\n")
    print(f"{'Opponent':<20} {'Wins':>5} {'Losses':>7} {'WR':>7}")
    print("-" * 45)

    total_wins = 0
    total_games = 0

    for opp_name in opponents:
        agent = _make_agent(agent_name, "BenchA")
        opp = _make_agent(opp_name, "BenchO")

        await agent.battle_against(opp, n_battles=n_battles)

        wins = agent.n_won_battles
        losses = opp.n_won_battles
        wr = wins / (wins + losses) if (wins + losses) > 0 else 0

        label = opp_name
        if opp_name == "softmax_low":
            label = "SoftmaxDmg(t=1.0)"

        print(f"  {label:<20} {wins:>3} {losses:>5} {wr:>7.1%}")
        total_wins += wins
        total_games += wins + losses

    print("-" * 45)
    overall_wr = total_wins / total_games if total_games > 0 else 0
    print(f"  {'OVERALL':<20} {total_wins:>3} {total_games - total_wins:>5} {overall_wr:>7.1%}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True, help="Agent to benchmark")
    parser.add_argument("--n", type=int, default=100, help="Battles per opponent")
    args = parser.parse_args()
    asyncio.run(benchmark(args.agent, args.n))


if __name__ == "__main__":
    main()
