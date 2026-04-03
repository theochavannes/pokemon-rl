"""Generate behavioral cloning data from a heuristic teacher.

Runs the teacher agent against one or more opponents, recording
(observation, action, action_mask) at every turn. Saves the dataset
for supervised pre-training of the RL policy.

Teachers:
  maxdamage    — MaxDamagePlayer (original, never switches)
  smart        — SmartHeuristicPlayer (MaxDamage moves + competitive switching)

Opponents: random, maxdamage, typematchup, aggressive_switcher, smart, mixed (all)

Requires: Showdown server running (node pokemon-showdown start --no-security)

Usage:
    python scripts/generate_bc_data.py                         # smart vs mixed (default)
    python scripts/generate_bc_data.py --teacher maxdamage     # original BC
    python scripts/generate_bc_data.py --n-battles 10000
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

log = logging.getLogger(__name__)

import numpy as np
from poke_env.environment.singles_env import SinglesEnv
from poke_env.player.baselines import RandomPlayer
from poke_env.ps_client.account_configuration import AccountConfiguration

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


async def collect(n_battles: int, teacher: str, opponent_name: str) -> dict:
    import random as _rnd
    import string as _str

    suffix = "".join(_rnd.choices(_str.ascii_lowercase, k=6))
    collector_cls = _make_collector_cls(_TEACHER_CLS[teacher])
    collector = collector_cls(
        battle_format=BATTLE_FORMAT,
        account_configuration=AccountConfiguration(f"BC{suffix}", None),
        max_concurrent_battles=100,
        log_level=40,
    )
    opp_cls = _OPPONENT_CLS[opponent_name]
    opponent = opp_cls(
        battle_format=BATTLE_FORMAT,
        account_configuration=AccountConfiguration(f"BCOpp{suffix}", None),
        max_concurrent_battles=100,
        log_level=40,
    )

    print(f"Collecting data: {teacher} vs {opponent_name} ({n_battles} games)...")
    await collector.battle_against(opponent, n_battles=n_battles)

    wins = collector.n_won_battles
    win_pct = (wins / n_battles * 100) if n_battles > 0 else 0.0
    print(f"  Done: {wins}/{n_battles} wins ({win_pct:.1f}%)")
    print(f"  Transitions collected: {len(collector.observations)}")

    return {
        "observations": np.array(collector.observations),
        "actions": np.array(collector.actions, dtype=np.int64),
        "masks": np.array(collector.masks),
    }


async def collect_mixed(n_battles: int, teacher: str) -> dict:
    """Collect data against ALL opponents concurrently (equal games per opponent)."""
    per_opp = n_battles // len(_OPPONENT_CLS)

    results = await asyncio.gather(*[collect(per_opp, teacher, opp_name) for opp_name in _OPPONENT_CLS])

    return {
        "observations": np.concatenate([r["observations"] for r in results]),
        "actions": np.concatenate([r["actions"] for r in results]),
        "masks": np.concatenate([r["masks"] for r in results]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-battles", type=int, default=5000)
    parser.add_argument("--teacher", choices=list(_TEACHER_CLS), default="smart")
    parser.add_argument("--opponent", choices=[*_OPPONENT_CLS, "mixed"], default="mixed")
    args = parser.parse_args()

    if args.opponent == "mixed":
        data = asyncio.run(collect_mixed(args.n_battles, args.teacher))
    else:
        data = asyncio.run(collect(args.n_battles, args.teacher, args.opponent))

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
