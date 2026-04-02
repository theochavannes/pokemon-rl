"""
Phase 3 baseline benchmark.

Pits MaxDamagePlayer vs RandomPlayer for N battles and prints win rates.
Expected result: MaxDamage wins >80% — anything lower indicates an env bug.

Usage (Showdown server must be running first):
    C:/Users/theoc/miniconda3/envs/pokemon_rl/python.exe scripts/benchmark_heuristic.py
"""

import asyncio
import sys

N_BATTLES = 100
BATTLE_FORMAT = "gen1randombattle"


async def main() -> None:
    # Import here so the module is importable without triggering poke-env init
    from src.agents.heuristic_agent import MaxDamagePlayer, RandomPlayer

    max_player = MaxDamagePlayer(battle_format=BATTLE_FORMAT, log_level=25)
    rand_player = RandomPlayer(battle_format=BATTLE_FORMAT, log_level=25)

    print(f"Running {N_BATTLES} battles: MaxDamagePlayer vs RandomPlayer ...")
    await max_player.battle_against(rand_player, n_battles=N_BATTLES)

    wins = max_player.n_won_battles
    total = N_BATTLES
    win_rate = wins / total * 100

    print(f"\nResults ({total} battles):")
    print(f"  MaxDamagePlayer wins : {wins:>3} / {total}  ({win_rate:.1f}%)")
    print(f"  RandomPlayer wins    : {total - wins:>3} / {total}  ({100 - win_rate:.1f}%)")

    if win_rate >= 80:
        print("\nENVIRONMENT OK — win rate above 80% threshold.")
    else:
        print(f"\nWARNING — win rate {win_rate:.1f}% is below 80%. Check the environment.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
