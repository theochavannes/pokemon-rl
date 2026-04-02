# What the Agent Knows: MaskablePPO Observation Reference

**Format:** gen1randombattle — all Pokemon level 100, random teams each battle.
**Observation:** 64-dim float32 vector, range [-1, 1].
**Actions:** 0–5 = switch to team slot; 6–9 = use move by discovery order.

---

## Quick Reference

| Category | Agent SEES | Agent does NOT see |
|---|---|---|
| Own active | HP fraction, 6 stat boosts, status, fainted flag | Species, types, base attack/defense/special stats |
| Own moves | Base power, type index, PP fraction, type effectiveness vs opponent | Move names, accuracy, priority, secondary effects |
| Own bench | HP fraction + fainted per slot | Species, types, moveset of benched Pokemon |
| Opponent active | HP fraction, 4 boosts (atk/def/spe/spa), status, fainted flag | Species, types, base stats |
| Opponent bench | HP fraction + fainted + revealed flag per slot | Species, types, moves used |
| Speed context | Binary: own_base_speed > opp_base_speed | Exact speed values, post-boost speed comparison |
| Battle meta | — | Turn number, weather, terrain, field effects |

---

## 1. Does the Agent Learn Move Positions or Move Features?

**It learns from features, not slot numbers.**

The action-to-move mapping in SinglesEnv is:

```
action 6 → list(battle.active_pokemon.moves.values())[0]
action 7 → list(battle.active_pokemon.moves.values())[1]
action 8 → list(battle.active_pokemon.moves.values())[2]
action 9 → list(battle.active_pokemon.moves.values())[3]
```

Moves appear in the order they were first used in battle (discovery order). The observation is built with the same ordering — move slot 0 in the action space corresponds to the first 4 features in the move block of the observation.

This is intentional and important: in gen1randombattle, every battle deals a different team with different moves. Slot 0 might be Thunderbolt in one battle and Surf in the next. A policy that learned "always press slot 0" would not generalize.

What the agent actually learns is: "the move with base_power=0.60, effectiveness=0.50 vs this opponent HP fraction is a good action." The policy attaches value to feature combinations, not to slot indices.

The 4 features per move slot:
- `base_power / 150.0` — max 130 in Gen 1 (Selfdestruct), normalized to ~0.87
- `type.value / 18.0` — PokemonType enum index, a proxy for type identity
- `current_pp / max_pp` — lets the agent avoid PP-stalling itself
- `type_multiplier(opp_type_1, opp_type_2) / 4.0` — actual Gen 1 type chart applied to opponent's types

The effectiveness feature is the closest thing the agent has to type reasoning. It computes the real Gen 1 type chart multiplier (including the Ghost→Psychic immunity bug). The agent implicitly learns the type chart through this precomputed scalar.

**What it cannot learn from features alone:** accuracy, secondary effect chance (paralysis on Body Slam, burn on Fire Blast), priority, or whether a move makes contact.

---

## 2. What Does the Agent Know About Its Own Pokemon?

**Own active Pokemon — 10 dims:**

| Dim | Feature | Range |
|---|---|---|
| 0 | HP fraction | [0, 1] |
| 1 | Atk boost | [-1, 1] (boost / 6) |
| 2 | Def boost | [-1, 1] |
| 3 | Speed boost | [-1, 1] |
| 4 | Special boost | [-1, 1] (Gen 1 single Special stat) |
| 5 | Accuracy boost | [-1, 1] |
| 6 | Evasion boost | [-1, 1] |
| 7 | Status | 0/0.2/0.4/0.6/0.8/1.0 (none/psn/brn/slp/par/frz) |
| 8 | is_active placeholder | always 1.0 |
| 9 | Fainted flag | 0 or 1 |

**Own bench — 12 dims (6 slots × 2):**

Each slot: `[hp_fraction, fainted_flag]`. No type, no species, no moves.

**What the agent cannot infer about itself:**
- Its own Pokemon's species or types
- Base attack, defense, or special stats — so it cannot estimate damage dealt or received
- Whether its own type resists or is weak to the opponent's moves (no own-type encoding)

---

## 3. What Does the Agent Know About the Opponent?

**Opponent active — 7 dims:**

| Dim | Feature | Notes |
|---|---|---|
| 0 | HP fraction | [0, 1] |
| 1–4 | Atk/Def/Spe/Spa boosts | 4 boosts, no accuracy/evasion |
| 5 | Status | same encoding as own |
| 6 | Fainted flag | |

The opponent has 4 boosts tracked (not 6) — evasion and accuracy are omitted from the opponent block. This is a minor asymmetry worth noting for Phase 6 expansion.

**Opponent bench — 18 dims (6 slots × 3):**

Each slot: `[hp_fraction, fainted_flag, revealed_flag]`.

- Unrevealed slot: `[-1.0, 0.0, 0.0]` — the -1 HP is a distinct learnable signal, not a missing value.
- Revealed slot: actual HP fraction, fainted flag, and revealed=1.0.

As the battle progresses and the opponent sends out more Pokemon, `battle.opponent_team` grows. The agent can track which opponent slots are known and at what HP without any explicit memory mechanism — the observation carries the accumulated state directly.

**What the agent cannot know about the opponent:**
- Species or types of any opponent Pokemon, revealed or not
- Which moves the opponent has used (no opponent move tracking in the observation)
- Opponent base stats — cannot estimate whether an opponent move will KO

---

## 4. Levels and Base Stats

All Pokemon in gen1randombattle are level 100. There is no level variation to model.

The only base stat information in the observation is the speed comparison bit (dim 63):
- `1.0` if `own.base_stats["spe"] > opp.base_stats["spe"]`
- `0.0` otherwise

This matters in Gen 1 because the faster Pokemon's moves land first, and because crit rate is derived from Speed (critical hit chance = base_speed / 512). A Jolteon outspeeding a Slowbro is a meaningful strategic fact, and the agent can see it.

Attack, Defense, and Special base stats are not in the observation. The agent cannot estimate:
- Whether its Pokemon will one-shot the opponent
- Whether it can survive a hit
- Damage ranges for any move

These gaps mean the agent must learn HP threshold patterns from trial and error — e.g., "when opponent is at 30% HP and I have a high-power move, act aggressively" — without being able to calculate exact damage.

---

## 5. What Information Persists Through a Battle?

poke-env accumulates state across turns on the battle object. The observation reflects this accumulated state at every step.

**Persists correctly:**
- `battle.opponent_team` grows as opponent Pokemon are revealed — the revealed flag in bench slots captures this
- `battle.active_pokemon.moves` grows as own moves are used — move features are populated as moves are discovered
- HP fractions update each turn
- Stat boosts accumulate correctly (Swords Dance stacks visible in the boost dims)
- Status conditions persist (paralysis, sleep, freeze visible until cured)

**What does NOT persist in the observation (significant gap):**
- Opponent move history — the agent cannot see which moves the opponent has actually used, only HP and revealed status
- Own move history — the agent doesn't observe which of its own moves it used last turn
- Turn count — no temporal encoding; a fresh-team battle looks identical to a 30-turn stalemate except for HP values

The agent sees "opponent slot 2 is revealed and at 40% HP" but not "opponent slot 2 used Thunderbolt last turn." An opponent at low HP after a known Psychic move is indistinguishable from one at low HP from damage taken while the agent was not on the field.

---

## 6. Key Gaps — Phase 6 Expansion Targets

The current observation is sufficient to learn basic strategic patterns, but several important signals are missing:

**High-priority additions:**
1. **Own Pokemon types** — agent cannot assess its own defensive typing; cannot know if it resists an attack
2. **Opponent revealed Pokemon types** — opponent slot types appear nowhere; agent must implicitly infer from effectiveness values
3. **Opponent revealed moves** — 4 features × 6 slots; knowing the opponent has used Thunderbolt dramatically changes switch decisions
4. **Own and opponent base attack/special stats** — needed for damage estimation; currently the agent cannot tell if a 120-power move from a high-Special attacker will KO

**Lower priority:**
5. Opponent revealed speed (not just the binary comparison)
6. Move accuracy (currently treated as uniform by the agent)
7. Turn count / battle phase signal

---

## 7. Gen 1 Specifics the Agent Operates Within

**1/256 miss:** Every move with 100% stated accuracy actually has a 1/256 miss chance in the Gen 1 engine. The agent cannot predict or avoid this. A winning move can randomly fail regardless of the observation state.

**No held items:** The observation correctly omits items — there are none in Gen 1.

**No team preview:** gen1randombattle starts immediately. The agent's first observation is mid-battle with only the lead Pokemon known. The bench revealed flags will all start at -1 for the opponent.

**Single Special stat:** Gen 1 has one Special stat covering both Special Attack and Special Defense. The `spa` boost dimension covers both in the Gen 1 context — an Amnesia doubles the Pokemon's Special offense and defense simultaneously. This is correctly reflected in how boosts are tracked.

**Crit rate derived from Speed:** Higher base speed = higher crit rate. The agent has indirect access to this via the speed comparison bit, but does not see a dedicated crit-rate signal. High-speed Pokemon (Jolteon, Tauros, Alakazam) will win more crits in practice; the agent can only learn this correlatively through outcomes.

**Freeze is permanent:** The status encoding includes freeze (1.0). A frozen Pokemon is effectively removed from play, but the observation does not distinguish "frozen since turn 2" from "just got frozen." The agent can only observe that the status is present.

**Sleep Clause:** Limits opponent to one Pokemon asleep at a time. The agent does not see a sleep clause flag — it would need to count opponent sleep statuses across bench slots to reason about this, which it can do approximately from the status dims.
