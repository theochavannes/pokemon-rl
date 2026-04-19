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
    # --- Ubers: excluded from the random pool (filter_ou_pool.py) ---
    "mew": 5,
    "mewtwo": 5,
}

_DEFAULT_TIER = 3  # With OU-only pool, unknown species should default near-OU


def team_score(team: dict) -> float:
    """Sum tier ratings for a team dict (poke-env battle.team / battle.opponent_team)."""
    total = 0.0
    for mon in team.values():
        species = mon.species.lower().replace("-", "").replace(" ", "")
        total += GEN1_TIER_RATINGS.get(species, _DEFAULT_TIER)
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
