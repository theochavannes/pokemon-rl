"""
Battle simulation — pit any two agents against each other.

Usage (Showdown server must be running):
    python scripts/battle_sim.py --agent1 best_model --agent2 random --n 20
    python scripts/battle_sim.py --agent1 models/phase_A_final --agent2 models/phase_B_final --n 50
    python scripts/battle_sim.py --agent1 selfplay_final --agent2 maxdamage --n 30

agent arguments:
    "random"              — RandomPlayer
    "maxdamage"           — MaxDamagePlayer
    any other string      — treated as a path inside models/ (with or without .zip)
                            e.g. "best_model", "models/best_model", "models/selfplay_final"
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import os
os.chdir(Path(__file__).parent.parent)

BATTLE_FORMAT = "gen1randombattle"
MODEL_DIR = Path("models")


def _resolve_path(name: str) -> Path | None:
    """
    Returns a Path to the .zip (without extension) if name is a model,
    or None if name is a builtin agent type ("random", "maxdamage").
    """
    if name.lower() in ("random", "maxdamage"):
        return None
    p = Path(name)
    if not p.suffix:
        p = p.with_suffix(".zip")
    if not p.exists():
        p = MODEL_DIR / p
    if not p.exists():
        raise FileNotFoundError(f"Model not found: {name} (tried {p})")
    return p.with_suffix("")  # SB3 load() doesn't want the .zip extension


def _make_player(name: str, battle_format: str, username: str):
    """Instantiate any supported agent type."""
    import random, string
    from poke_env.ps_client.account_configuration import AccountConfiguration

    suffix = "".join(random.choices(string.ascii_lowercase, k=6))
    account = AccountConfiguration(f"{username[:10]}{suffix}", None)
    kwargs = dict(battle_format=battle_format, account_configuration=account, log_level=25)

    if name.lower() == "random":
        from poke_env.player.baselines import RandomPlayer
        return RandomPlayer(**kwargs)
    elif name.lower() == "maxdamage":
        from src.agents.heuristic_agent import MaxDamagePlayer
        return MaxDamagePlayer(**kwargs)
    else:
        from src.agents.policy_player import FrozenPolicyPlayer
        model_path = str(_resolve_path(name))
        return FrozenPolicyPlayer(model_path=model_path, **kwargs)


async def run_sim(agent1_name: str, agent2_name: str, n_battles: int) -> None:
    p1 = _make_player(agent1_name, BATTLE_FORMAT, "SimA")
    p2 = _make_player(agent2_name, BATTLE_FORMAT, "SimB")

    print(f"\nSimulation: {agent1_name} vs {agent2_name} ({n_battles} battles)\n")

    await p1.battle_against(p2, n_battles=n_battles)

    wins_p1 = p1.n_won_battles
    wins_p2 = p2.n_won_battles
    total = wins_p1 + wins_p2

    bar1 = "█" * int(wins_p1 / total * 30) if total else ""
    bar2 = "█" * int(wins_p2 / total * 30) if total else ""

    print(f"\n{'─'*50}")
    print(f"  {agent1_name:<20} {wins_p1:>3} wins  ({wins_p1/total*100:.1f}%)  {bar1}")
    print(f"  {agent2_name:<20} {wins_p2:>3} wins  ({wins_p2/total*100:.1f}%)  {bar2}")
    print(f"{'─'*50}")
    print(f"  Total battles: {total}")


def main():
    parser = argparse.ArgumentParser(description="Pokemon RL battle simulation")
    parser.add_argument("--agent1", required=True,
                        help="'random', 'maxdamage', or a model name/path")
    parser.add_argument("--agent2", required=True,
                        help="'random', 'maxdamage', or a model name/path")
    parser.add_argument("--n", type=int, default=20,
                        help="Number of battles (default: 20)")
    args = parser.parse_args()

    asyncio.run(run_sim(args.agent1, args.agent2, args.n))


if __name__ == "__main__":
    main()
