"""Generate behavioral cloning data from MaxDamagePlayer vs RandomPlayer.

Runs MaxDamagePlayer against RandomPlayer for N games, recording
(observation, action, action_mask) at every turn. Saves the dataset
for supervised pre-training of the RL policy.

Requires: Showdown server running (node pokemon-showdown start --no-security)

Usage:
    python scripts/generate_bc_data.py                  # 5000 games (default)
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

from src.agents.heuristic_agent import MaxDamagePlayer
from src.env.gen1_env import embed_battle

BATTLE_FORMAT = "gen1randombattle"


class DataCollectingMaxDamage(MaxDamagePlayer):
    """MaxDamagePlayer that records (obs, action, mask) at every turn."""

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


async def collect(n_battles: int) -> dict:
    collector = DataCollectingMaxDamage(
        battle_format=BATTLE_FORMAT,
        account_configuration=AccountConfiguration("BCCollector", None),
        max_concurrent_battles=10,
        log_level=40,
    )
    opponent = RandomPlayer(
        battle_format=BATTLE_FORMAT,
        account_configuration=AccountConfiguration("BCOpponent", None),
        max_concurrent_battles=10,
        log_level=40,
    )

    print(f"Collecting data: MaxDamagePlayer vs RandomPlayer ({n_battles} games)...")
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-battles", type=int, default=5000)
    args = parser.parse_args()

    data = asyncio.run(collect(args.n_battles))

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

    # Damage quality check: what % of move actions picked the best move?
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
