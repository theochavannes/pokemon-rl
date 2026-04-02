"""
Phase 1 smoke test.

Connects two RandomPlayers to a running Showdown server and plays 1 battle.

Usage:
    Start the Showdown server first:
        cd showdown && node pokemon-showdown start --no-security

    Then run:
        C:/Users/theoc/miniconda3/envs/pokemon_rl/python.exe scripts/verify_setup.py
"""

import asyncio
import sys

from poke_env.player import RandomPlayer


async def main() -> None:
    p1 = RandomPlayer(battle_format="gen1randombattle", log_level=25)
    p2 = RandomPlayer(battle_format="gen1randombattle", log_level=25)

    print("Connecting to Showdown server at ws://localhost:8000 ...")
    print("Running 1 battle: RandomPlayer vs RandomPlayer ...")

    await p1.battle_against(p2, n_battles=1)

    total = p1.n_won_battles + p2.n_won_battles
    if total != 1:
        print(f"ERROR: expected 1 completed battle, got {total}", file=sys.stderr)
        sys.exit(1)

    winner = "P1" if p1.n_won_battles == 1 else "P2"
    print(f"Battle complete. Winner: {winner}")
    print("Setup verified — poke-env <-> Showdown connection is working.")


if __name__ == "__main__":
    asyncio.run(main())
