# Plan: Restricted Pokemon Pool for Curriculum Training

## Context

The agent trains on gen1randombattle (146 Pokemon, random teams). The idea: restrict to a smaller pool first (e.g., OU tier), master it, then expand. This came from observing that the agent must simultaneously learn general battle tactics AND species-specific knowledge across a massive combinatorial space (146×146 = 21K pairings, C(146,6) = 12B possible teams).

**Current blocker:** BC warm-start is broken (46.6% accuracy). This must be fixed first regardless of pool restriction. The pool restriction is a parallel design decision.

## Team Brainstorm Summary

### Panel: [ML], [RL], [RL2] (new, curriculum specialist), [RBY], [SMOGON] (new, active RBY ladder), [SE], [REVIEW]

**Consensus: CONDITIONAL ACCEPT with modifications.**

### Key Decisions

1. **Pool size: 38 Pokemon (Tiers 5+4+3), NOT 16.**
   - 16 (pure OU) is too degenerate: Psychic dominates (5/16 are Psychic), no Bug/Dragon/Fighting types, C(16,6) = 8K teams leads to memorization not generalization
   - 38 gives C(38,6) = 2.76M teams — enough diversity, includes Dragonite, Jolteon, Nidoking, Poliwrath for type coverage
   - [RBY] and [SMOGON] both strongly against 16 — the OU metagame teaches bad habits (hyper-optimize for Psychic, ignore types that matter in full pool)

2. **Two-phase expansion: 38 → 146.** No intermediate step at 16.

3. **Fix BC warm-start FIRST.** The restricted pool work is designed in parallel but not deployed until BC is working.

4. **Transfer success criterion: 55% WR vs mixed league within 200K steps of expanding to 146.** If slower than training from scratch, the curriculum was a net negative.

5. **Weighted sampling as Plan B.** If hard restriction fails transfer, implement probability-weighted Pokemon selection (OU 5x more likely) that gradually flattens to uniform. Avoids the distribution cliff entirely.

### Key Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Transfer fails — restricted policy worse than random init | High | 55% WR criterion. Fall back to full-pool training. |
| Degenerate Nash equilibrium in self-play | Medium | 38 Pokemon (not 16). Monitor self-play WR convergence. |
| Showdown `enforceNoDirectCustomBanlistChanges()` blocks banlists | Medium | Modify `data.json` directly (filter entries). |
| Type coverage gaps cause blind spots | Medium | 38 Pokemon covers most types. Accept some gaps. |
| Covariate shift — policy calibrated to OU stat ranges | Medium | PPO on-policy collection adapts. May need BC refresh. |

### What [REVIEW] Challenged

- "Have we proven pool size is the bottleneck?" — No. run_043 hit 62% on the full pool. run_045 collapsed due to bad BC, not pool size. Fix BC first, run full-pool to 2M steps, and ONLY pursue restriction if that stalls below 65%.
- "Transfer is unproven" — Obs space is identical but distribution shifts (stat ranges, type frequencies). MLP has no explicit transfer mechanism.

The team accepted this: Phase 0 establishes the full-pool baseline before committing to restriction.

## Implementation Plan

### Phase 0: Prerequisites (do first)
- Fix BC warm-start (already in progress on `fix-bc-warmstart` branch)
- Run full-pool training to 2M steps to establish baseline WR
- **Gate:** Only proceed to Phase 1 if full-pool WR stalls below 65%

### Phase 1: Showdown Server — Create Restricted Format

**Step 1.1: Create filtered data file**

Copy `showdown/data/random-battles/gen1/data.json` → `data_restricted.json`, keeping only the 38 Tier 3+4+5 species. The format of each entry is unchanged — just delete the other 108 entries.

The 38 Pokemon to keep:
- T5: tauros, chansey, snorlax, exeggutor, starmie, alakazam
- T4: jynx, zapdos, lapras, rhydon, golem, cloyster, gengar, slowbro, articuno, moltres
- T3: jolteon, hypno, dragonite, persian, dodrio, victreebel, venusaur, mrmime, electrode, charizard, blastoise, nidoking, nidoqueen, poliwrath, machamp, tentacruel, kangaskhan, vaporeon, flareon, omastar, kabutops, aerodactyl

**Step 1.2: Modify team generator to support filtered pool**

In `showdown/data/random-battles/gen1/teams.ts`, the `randomTeam()` method loads Pokemon from `this.randomData` (line 110). Modify to support an alternative data source:

```typescript
// In randomTeam(), replace line 110:
const dataKeys = this.format.customRules?.includes('restrictedpool')
    ? Object.keys(require('./data_restricted.json'))
    : Object.keys(this.randomData);
const pokemonPool = Object.keys(this.getPokemonPool(type, pokemon, isMonotype, dataKeys)[0]);
```

Or simpler: just create a second format that loads `data_restricted.json` as its `randomData`.

**Step 1.3: Register custom format**

In `showdown/config/formats.ts`, add after the existing Gen 1 Random Battle:
```typescript
{
    name: "[Gen 1] Random Battle (Restricted)",
    mod: 'gen1',
    team: 'random',
    ruleset: ['Standard'],
    // Point at filtered data via custom team generator
},
```

**Step 1.4: Rebuild and test**
```bash
cd showdown && node build
```

### Phase 2: Python Pipeline — Parameterize Format

**Step 2.1: Add `--format` flag to all scripts**

Files to modify:
- `src/train.py` — `BATTLE_FORMAT` constant + CLI arg
- `scripts/generate_bc_data.py` — `BATTLE_FORMAT` constant + CLI arg
- `scripts/behavioral_cloning.py` — `make_env` call
- `scripts/benchmark_heuristic.py`, `scripts/battle_sim.py`, `scripts/tournament.py`

**Step 2.2: `tier_baseline.py` — no change needed**

With 38 tier-equal Pokemon, `matchup_baseline()` returns near-zero. This is correct (matchups are balanced). Harmless.

### Phase 3: Restricted-Pool Training

**Step 3.1: Generate restricted BC data**
```bash
python scripts/generate_bc_data.py --format gen1randombattlerestricted --teacher smart --opponent mixed --n-battles 20000
```

**Step 3.2: Train BC**
```bash
python scripts/behavioral_cloning.py
```

Expect higher accuracy than full-pool BC (fewer distinct situations to learn).

**Step 3.3: Train PPO**
```bash
python src/train.py --new-run --format gen1randombattlerestricted
```

Config adjustments for restricted pool:
- `ent_coef`: 0.01 → 0.02 (prevent premature convergence in smaller state space)
- `max_steps`: 2M → 1M (faster convergence expected)
- Target: 70% WR vs mixed league

### Phase 4: Expansion to Full Pool

**Step 4.1: Load restricted checkpoint, switch format**
```bash
python src/train.py --format gen1randombattle
# (auto-resumes from latest checkpoint)
```

The obs space is identical (1559 dims). No `obs_transfer` needed. Just load and continue.

**Step 4.2: Monitor transfer**

Success: 55% WR vs mixed league within 200K steps.
Failure: <45% WR after 200K steps → fall back to full-pool training from scratch.

### Phase 5 (Plan B): Weighted Sampling

If hard restriction fails transfer, implement tier-weighted Pokemon sampling in `teams.ts`:
- Tier 5: weight 5, Tier 4: weight 4, ..., Tier 1: weight 1
- Gradually flatten weights to uniform during training (controlled by an external config or callback)
- No distribution cliff, smooth transition

## Critical Files

- `showdown/data/random-battles/gen1/data.json` — source for filtered data
- `showdown/data/random-battles/gen1/teams.ts` — team generation logic (line 90: `enforceNoDirectCustomBanlistChanges`, line 110: Pokemon pool)
- `showdown/config/formats.ts` — format registration (line 4058)
- `src/train.py` — `BATTLE_FORMAT` (line 47)
- `scripts/generate_bc_data.py` — `BATTLE_FORMAT` (line 48)
- `src/tier_baseline.py` — `GEN1_TIER_RATINGS` dict

## Verification

1. Start restricted Showdown format → verify only 38 Pokemon appear in battles
2. Generate BC data with restricted format → verify obs shape is still (1559,)
3. Train BC → verify accuracy is higher than full-pool BC
4. Run PPO training for 500K steps → verify convergence to >65% WR
5. Switch to full-pool format → verify checkpoint loads and training continues
6. Monitor WR after expansion — must reach 55% within 200K steps
