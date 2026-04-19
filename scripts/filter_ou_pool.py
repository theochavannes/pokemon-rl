"""Filter Showdown's Gen 1 random battle pool to OU-quality species only.

Rewrites `showdown/data/random-battles/gen1/data.json` in place, keeping only
species with a Smogon tier rating >= 3 plus Ditto (a special case — it
transforms into its opponent, making it effectively OU-quality regardless of
its own stats). The full original pool is preserved as `data_full.json`.

Mew and Mewtwo are excluded as Ubers — they create the same unwinnable
matchups (Magikarp vs Tauros) we are trying to remove, just in reverse.

Re-runnable: if `data_full.json` already exists, it is used as the source.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.tier_baseline import GEN1_TIER_RATINGS  # noqa: E402

DATA_DIR = _ROOT / "showdown" / "data" / "random-battles" / "gen1"
DATA_JSON = DATA_DIR / "data.json"
DATA_FULL_JSON = DATA_DIR / "data_full.json"

MIN_TIER = 3
EXCLUDED_UBERS = {"mew", "mewtwo"}
ALWAYS_KEEP = {"ditto"}  # transforms into opponent → behaves as OU


def _normalize(name: str) -> str:
    """Match the key format used by GEN1_TIER_RATINGS (lowercase, stripped)."""
    return name.lower().replace("-", "").replace(" ", "").replace("'", "").replace(".", "")


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON via temp file + rename, so an interrupted run cannot leave a truncated file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    tmp.replace(path)


def main() -> int:
    if not DATA_DIR.exists():
        print(f"ERROR: {DATA_DIR} does not exist. Did you run `git submodule update --init`?")
        return 1

    source = DATA_FULL_JSON if DATA_FULL_JSON.exists() else DATA_JSON
    with source.open("r", encoding="utf-8") as f:
        full_pool: dict[str, dict] = json.load(f)

    if not DATA_FULL_JSON.exists():
        _atomic_write_json(DATA_FULL_JSON, full_pool)
        print(f"Backed up original pool ({len(full_pool)} species) -> {DATA_FULL_JSON.name}")

    kept: dict[str, dict] = {}
    for species, entry in full_pool.items():
        norm = _normalize(species)
        if norm in EXCLUDED_UBERS:
            continue
        if norm in ALWAYS_KEEP:
            kept[species] = entry
            continue
        tier = GEN1_TIER_RATINGS.get(norm)
        if tier is None:
            print(f"  WARN: no tier rating for {species!r} (normalized: {norm!r}) — excluding")
            continue
        if tier >= MIN_TIER:
            kept[species] = entry

    _atomic_write_json(DATA_JSON, kept)

    removed = len(full_pool) - len(kept)
    print(f"Filtered pool: {len(full_pool)} -> {len(kept)} species ({removed} removed)")
    print(f"Wrote {DATA_JSON}")
    print("\nKept species:")
    for species in sorted(kept):
        tier = GEN1_TIER_RATINGS.get(species, "?")
        print(f"  {species:15s} (tier {tier})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
