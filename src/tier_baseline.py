"""
Matchup baseline: reduces variance from team RNG in gen1randombattle.

Computes a team-quality score from Smogon competitive tier ratings, then
adjusts the terminal reward so the agent gets more credit for winning hard
matchups and more blame for losing easy ones.

The baseline is ADDITIVE (subtracted from the terminal reward), which
preserves policy gradient unbiasedness while reducing variance.

Theory: this is an episode-level control variate. Any baseline that doesn't
depend on the agent's actions leaves the gradient unbiased. See panel
discussion in notes/panel_discussion_matchup_baseline.md for full rationale.
"""

# ---------------------------------------------------------------------------
# Gen 1 tier ratings (1–5 scale, based on Smogon RBY competitive viability)
#
# 5 = OU dominant
# 4 = OU solid / high UU
# 3 = UU / usable
# 2 = NU / niche
# 1 = NFE / near-useless
#
# In gen1randombattle, team quality correlates heavily with how many high-tier
# Pokemon you roll. This alone captures ~60% of matchup variance per our
# competitive RBY consultant.
# ---------------------------------------------------------------------------

GEN1_TIER_RATINGS: dict[str, int] = {
    # --- Tier 5: OU dominant ---
    "tauros": 5,
    "chansey": 5,
    "snorlax": 5,
    "exeggutor": 5,
    "starmie": 5,
    "alakazam": 5,
    # --- Tier 4: OU solid / high UU ---
    "jynx": 4,
    "zapdos": 4,
    "lapras": 4,
    "rhydon": 4,
    "golem": 4,
    "cloyster": 4,
    "gengar": 4,
    "slowbro": 4,
    "articuno": 4,
    "moltres": 4,
    # --- Tier 3: UU / usable ---
    "jolteon": 3,
    "hypno": 3,
    "dragonite": 3,
    "persian": 3,
    "dodrio": 3,
    "victreebel": 3,
    "venusaur": 3,
    "mrmime": 3,
    "electrode": 3,
    "charizard": 3,
    "blastoise": 3,
    "nidoking": 3,
    "nidoqueen": 3,
    "poliwrath": 3,
    "machamp": 3,
    "tentacruel": 3,
    "kangaskhan": 3,
    "vaporeon": 3,
    "flareon": 3,
    "omastar": 3,
    "kabutops": 3,
    "aerodactyl": 3,
    # --- Tier 2: NU / niche (fully evolved, limited competitive use) ---
    "raticate": 2,
    "arbok": 2,
    "sandslash": 2,
    "wigglytuff": 2,
    "golduck": 2,
    "primeape": 2,
    "arcanine": 2,
    "vileplume": 2,
    "parasect": 2,
    "dewgong": 2,
    "rapidash": 2,
    "magneton": 2,
    "dugtrio": 2,
    "clefable": 2,
    "ninetales": 2,
    "kingler": 2,
    "hitmonlee": 2,
    "hitmonchan": 2,
    "weezing": 2,
    "tangela": 2,
    "seaking": 2,
    "pinsir": 2,
    "gyarados": 2,
    "porygon": 2,
    "pidgeot": 2,
    "fearow": 2,
    "butterfree": 2,
    "beedrill": 2,
    "haunter": 2,
    "kadabra": 2,
    "electabuzz": 2,
    "magmar": 2,
    "scyther": 2,
    "marowak": 2,
    "venomoth": 2,
    "golbat": 2,
    "lickitung": 2,
    "muk": 2,
    "seadra": 2,
    "raichu": 2,
    # --- Tier 1: NFE / very weak ---
    "bulbasaur": 1,
    "ivysaur": 1,
    "charmander": 1,
    "charmeleon": 1,
    "squirtle": 1,
    "wartortle": 1,
    "caterpie": 1,
    "metapod": 1,
    "weedle": 1,
    "kakuna": 1,
    "pidgey": 1,
    "pidgeotto": 1,
    "rattata": 1,
    "spearow": 1,
    "ekans": 1,
    "pikachu": 1,
    "sandshrew": 1,
    "nidoranf": 1,
    "nidorina": 1,
    "nidoranm": 1,
    "nidorino": 1,
    "clefairy": 1,
    "vulpix": 1,
    "jigglypuff": 1,
    "zubat": 1,
    "oddish": 1,
    "gloom": 1,
    "paras": 1,
    "venonat": 1,
    "diglett": 1,
    "meowth": 1,
    "psyduck": 1,
    "mankey": 1,
    "growlithe": 1,
    "poliwag": 1,
    "poliwhirl": 1,
    "abra": 1,
    "machop": 1,
    "machoke": 1,
    "bellsprout": 1,
    "weepinbell": 1,
    "geodude": 1,
    "graveler": 1,
    "ponyta": 1,
    "slowpoke": 1,
    "magnemite": 1,
    "farfetchd": 1,
    "doduo": 1,
    "seel": 1,
    "grimer": 1,
    "shellder": 1,
    "gastly": 1,
    "onix": 1,
    "voltorb": 1,
    "exeggcute": 1,
    "cubone": 1,
    "koffing": 1,
    "rhyhorn": 1,
    "horsea": 1,
    "goldeen": 1,
    "staryu": 1,
    "magikarp": 1,
    "ditto": 1,
    "eevee": 1,
    "omanyte": 1,
    "kabuto": 1,
    "dratini": 1,
    "dragonair": 1,
    "drowzee": 1,
    "tentacool": 1,
    "krabby": 1,
    # Ubers (Mew, Mewtwo) are intentionally absent — they are excluded from
    # the random pool by scripts/filter_ou_pool.py, and we don't want a
    # fallback rating here to silently give them the highest matchup score
    # if the filter is ever skipped.
}

_DEFAULT_TIER = 3  # With OU-only pool, unknown species should default near-OU

_warned_species: set[str] = set()


def _normalize_species(raw: str) -> str:
    """Strip characters that vary between poke-env outputs (hyphens, spaces, apostrophes, periods)."""
    return raw.lower().replace("-", "").replace(" ", "").replace("'", "").replace(".", "")


# ---------------------------------------------------------------------------
# Competitive roles (12-dim multi-label binary vectors per species)
#
# Index | Role                Example mons
#   0   Physical Sweeper      Tauros, Rhydon, Dodrio
#   1   Special Sweeper       Alakazam, Starmie, Jynx
#   2   Mixed Attacker        Nidoking, Charizard, Dragonite
#   3   Physical Wall         Golem, Rhydon
#   4   Special Wall          Chansey, Slowbro, Lapras
#   5   Status Spreader       Exeggutor (Sleep), Jynx (Lovely Kiss), Venusaur
#   6   Revenge Killer        Jolteon, Electrode, Tauros, Persian
#   7   Pivot                 Slowbro, Chansey
#   8   Setup Sweeper         Snorlax (Amnesia), Slowbro (Amnesia)
#   9   Trapper               Cloyster (Clamp), Dragonite (Wrap)
#  10   Tank                  Snorlax, Lapras, Vaporeon
#  11   Utility               Starmie, Alakazam, Chansey (speed / support)
# ---------------------------------------------------------------------------

NUM_ROLES = 12

_ROLE = {
    "PHYS_SWEEP": 0,
    "SPEC_SWEEP": 1,
    "MIXED": 2,
    "PHYS_WALL": 3,
    "SPEC_WALL": 4,
    "STATUS": 5,
    "REVENGE": 6,
    "PIVOT": 7,
    "SETUP": 8,
    "TRAPPER": 9,
    "TANK": 10,
    "UTILITY": 11,
}


def _roles(*names: str) -> list[int]:
    vec = [0] * NUM_ROLES
    for n in names:
        vec[_ROLE[n]] = 1
    return vec


GEN1_ROLE_MAP: dict[str, list[int]] = {
    # --- Tier 5: OU dominant ---
    "tauros": _roles("PHYS_SWEEP", "REVENGE"),
    "chansey": _roles("SPEC_WALL", "STATUS", "UTILITY", "PIVOT"),
    "snorlax": _roles("TANK", "SETUP", "PHYS_SWEEP"),
    "exeggutor": _roles("STATUS", "MIXED", "TANK"),
    "starmie": _roles("SPEC_SWEEP", "REVENGE", "UTILITY"),
    "alakazam": _roles("SPEC_SWEEP", "REVENGE", "UTILITY"),
    # --- Tier 4: OU solid / high UU ---
    "jynx": _roles("STATUS", "SPEC_SWEEP"),
    "zapdos": _roles("SPEC_SWEEP", "STATUS"),
    "lapras": _roles("TANK", "SPEC_WALL", "SPEC_SWEEP"),
    "rhydon": _roles("PHYS_SWEEP", "PHYS_WALL", "TANK"),
    "golem": _roles("PHYS_WALL", "PHYS_SWEEP"),
    "cloyster": _roles("TRAPPER", "PHYS_WALL"),
    "gengar": _roles("SPEC_SWEEP", "STATUS"),
    "slowbro": _roles("SETUP", "SPEC_WALL", "PIVOT"),
    "articuno": _roles("SPEC_SWEEP", "TANK"),
    "moltres": _roles("SPEC_SWEEP"),
    # --- Tier 3: UU / usable ---
    "jolteon": _roles("SPEC_SWEEP", "REVENGE", "STATUS"),
    "hypno": _roles("STATUS", "UTILITY"),
    "dragonite": _roles("MIXED", "TRAPPER"),
    "persian": _roles("PHYS_SWEEP", "REVENGE"),
    "dodrio": _roles("PHYS_SWEEP", "REVENGE"),
    "victreebel": _roles("STATUS", "PHYS_SWEEP"),
    "venusaur": _roles("STATUS", "TANK"),
    "mrmime": _roles("UTILITY", "STATUS"),
    "electrode": _roles("REVENGE", "SPEC_SWEEP"),
    "charizard": _roles("MIXED"),
    "blastoise": _roles("TANK", "SPEC_WALL"),
    "nidoking": _roles("MIXED"),
    "nidoqueen": _roles("MIXED", "TANK"),
    "poliwrath": _roles("SETUP", "MIXED", "TANK"),
    "machamp": _roles("PHYS_SWEEP"),
    "tentacruel": _roles("SPEC_SWEEP", "SPEC_WALL"),
    "kangaskhan": _roles("PHYS_SWEEP", "TANK"),
    "vaporeon": _roles("TANK", "SPEC_WALL", "SETUP"),
    "flareon": _roles("MIXED"),
    "omastar": _roles("SPEC_SWEEP"),
    "kabutops": _roles("PHYS_SWEEP"),
    "aerodactyl": _roles("PHYS_SWEEP", "REVENGE"),
    # Ditto: transforms into opponent — no fixed role signal
    "ditto": [0] * NUM_ROLES,
}

_ZERO_ROLE_VEC = [0] * NUM_ROLES


def roles_for(species: str) -> list[int]:
    """Return the 12-dim role vector for a species, or all zeros if unknown."""
    return GEN1_ROLE_MAP.get(_normalize_species(species), _ZERO_ROLE_VEC)


def team_score(team: dict) -> float:
    """Sum tier ratings for a team dict (poke-env battle.team / battle.opponent_team)."""
    import logging

    total = 0.0
    for mon in team.values():
        raw = mon.species
        species = _normalize_species(raw)
        if species in GEN1_TIER_RATINGS:
            total += GEN1_TIER_RATINGS[species]
        else:
            total += _DEFAULT_TIER
            if species not in _warned_species:
                _warned_species.add(species)
                logging.getLogger("pokemon_rl.tier_baseline").warning(
                    "No tier rating for species %r (normalized %r) — using default %d",
                    raw,
                    species,
                    _DEFAULT_TIER,
                )
    return total


def matchup_baseline(battle) -> float:
    """Compute additive matchup baseline from revealed team quality.

    Returns a value in roughly [-2, +2] representing how favored the agent is.
    Positive = agent has the better team, negative = opponent has the better team.

    The baseline is normalized by team size so it represents per-Pokemon
    advantage, scaled to be meaningful relative to the victory_value (3.0).
    """
    own_score = team_score(battle.team)
    opp_score = team_score(battle.opponent_team)

    # Normalize: difference in average tier rating per Pokemon
    own_avg = own_score / max(len(battle.team), 1)
    opp_avg = opp_score / max(len(battle.opponent_team), 1)

    # Scale so a 1-tier average advantage ≈ 1.0 baseline adjustment
    # (meaningful relative to victory_value=3.0)
    return own_avg - opp_avg
