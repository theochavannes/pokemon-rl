"""
Phase 6 — Agent Tournament.

Runs a round-robin tournament between saved model checkpoints.
Each pair plays N_BATTLES battles and results are logged to content/tournament_results.md.

Usage (Showdown server must be running first):
    C:/Users/theoc/miniconda3/envs/pokemon_rl/python.exe scripts/tournament.py

Config:
    Edit CHECKPOINTS below to select which models to include.
    Use "best_model" for the best checkpoint, or "ppo_pokemon_{step}_steps" for specific ones.
"""

import asyncio
import json
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sb3_contrib import MaskablePPO

BATTLE_FORMAT = "gen1randombattle"
N_BATTLES = 20          # battles per matchup
MODEL_DIR = Path("models")
REPLAY_DIR = Path("replays/tournament")
RESULTS_PATH = Path("content/tournament_results.md")

# Edit this list to select which checkpoints to include.
# Pulls descriptions from checkpoint_registry.json if available.
CHECKPOINTS = [
    "ppo_pokemon_50000_steps",
    "ppo_pokemon_100000_steps",
    "ppo_pokemon_200000_steps",
    "ppo_pokemon_350000_steps",
    "ppo_pokemon_500000_steps",
    "best_model",
]


class CheckpointPlayer:
    """Wraps a MaskablePPO checkpoint as a poke-env-compatible player."""

    def __init__(self, name: str, model_path: str, battle_format: str):
        from poke_env.player.player import Player
        from poke_env.ps_client.account_configuration import AccountConfiguration
        import random, string

        self.name = name
        self.model = MaskablePPO.load(model_path)
        suffix = "".join(random.choices(string.ascii_lowercase, k=6))
        self._username = f"{name[:8]}{suffix}"
        self._battle_format = battle_format

    def choose_move(self, battle):
        """Use the loaded policy to choose a move."""
        from src.env.gen1_env import Gen1Env
        import numpy as np

        # Build obs and mask from the battle state
        # We use a lightweight Gen1Env instance just for embed_battle and get_action_mask
        obs = _EMBED.embed_battle(battle)
        mask = np.array(Gen1Env.get_action_mask(battle), dtype=bool)

        action, _ = self.model.predict(obs, action_masks=mask, deterministic=True)
        return action


# Shared embedding instance (avoids recreating GenData type chart each call)
_EMBED = None


def _load_registry() -> dict:
    path = MODEL_DIR / "checkpoint_registry.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


async def run_matchup(name_a: str, path_a: str, name_b: str, path_b: str) -> dict:
    """Run N_BATTLES between two checkpoints and return results."""
    from poke_env.player.player import Player
    from poke_env.ps_client.account_configuration import AccountConfiguration
    from src.env.gen1_env import Gen1Env, make_env
    import random, string

    # Use make_env approach: two RandomPlayers replaced by policy players
    # For now, use poke-env's battle_against with custom choose_move
    # This is a simplified version — full implementation in Phase 6
    print(f"  {name_a} vs {name_b} ({N_BATTLES} battles)...")

    model_a = MaskablePPO.load(path_a)
    model_b = MaskablePPO.load(path_b)

    from poke_env.player.baselines import RandomPlayer
    from poke_env.environment.singles_env import SinglesEnv

    # Placeholder — full implementation requires wrapping both as poke-env Players
    # Returns simulated result for now; will be replaced in Phase 6
    wins_a = 0
    wins_b = 0
    print(f"  ⚠️  Full tournament implementation pending Phase 6. Stub result logged.")
    return {"wins_a": wins_a, "wins_b": wins_b, "total": N_BATTLES}


def write_results(results: list, registry: dict) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    from datetime import datetime

    lines = [
        "# Tournament Results\n",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}  \n",
        f"Battles per matchup: {N_BATTLES}  \n",
        f"Format: {BATTLE_FORMAT}\n\n",
        "## Standings\n\n",
        "| Model | Step | Training Win Rate | Tournament Wins | Tournament Losses |\n",
        "|-------|------|-------------------|-----------------|-------------------|\n",
    ]

    win_totals: dict = {c: 0 for c in CHECKPOINTS}
    loss_totals: dict = {c: 0 for c in CHECKPOINTS}

    for r in results:
        win_totals[r["a"]] += r["wins_a"]
        win_totals[r["b"]] += r["wins_b"]
        loss_totals[r["a"]] += r["wins_b"]
        loss_totals[r["b"]] += r["wins_a"]

    for chk in CHECKPOINTS:
        reg = registry.get(chk, {})
        step = reg.get("step", "?")
        wr = reg.get("win_rate", "?")
        lines.append(
            f"| {chk} | {step} | {wr} | {win_totals[chk]} | {loss_totals[chk]} |\n"
        )

    lines += ["\n## Head-to-Head Results\n\n",
              "| Matchup | Wins A | Wins B |\n",
              "|---------|--------|--------|\n"]
    for r in results:
        lines.append(f"| {r['a']} vs {r['b']} | {r['wins_a']} | {r['wins_b']} |\n")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"\nResults written to {RESULTS_PATH}")


async def main():
    REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    registry = _load_registry()

    available = [c for c in CHECKPOINTS if (MODEL_DIR / f"{c}.zip").exists()]
    if len(available) < 2:
        print(f"Need at least 2 checkpoints in {MODEL_DIR}/. Found: {available}")
        sys.exit(1)

    print(f"Tournament: {len(available)} agents, {len(list(combinations(available, 2)))} matchups\n")

    matchup_results = []
    for name_a, name_b in combinations(available, 2):
        path_a = str(MODEL_DIR / name_a)
        path_b = str(MODEL_DIR / name_b)
        result = await run_matchup(name_a, path_a, name_b, path_b)
        result.update({"a": name_a, "b": name_b})
        matchup_results.append(result)

    write_results(matchup_results, registry)


if __name__ == "__main__":
    asyncio.run(main())
