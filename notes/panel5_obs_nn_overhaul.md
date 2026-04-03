# Expert Panel #5: Observation Space + NN Architecture Overhaul

Date: 2026-04-03

Panel: [RL-ACAD] Academic RL Expert, [RL-DM] DeepMind RL Expert, [SMOGON] Smogon Gen 1 Tournament Player, [RBY] Gen 1 Competitive Expert, full dev crew

## Context

run_043 with 704-dim obs (added move category, status_effect, priority, boost, heal) showed same plateau as run_041 (421-dim). Agent never learned to use new features because:
1. BC teacher (SmartHeuristicPlayer) doesn't use status moves
2. [64,64] NN (63K params) can't process 704 dims effectively

## Discussion

### RL Researchers: NN Architecture Problems

**[RL-ACAD]:**
- First hidden layer is 704->64 = 11:1 compression ratio (image classifiers use 3:1)
- Shared feature extractor between policy and value networks forces both to compromise
- The NN doesn't have capacity to learn patterns in 704+ dims with only 64 neurons

**[RL-DM]:**
- Linear encoding of categorical data (status effects as 0.0/0.2/0.4/0.6/0.8/1.0) implies ordering that doesn't exist. Functional for now but not ideal
- Recommendation: [256, 128] with separate pi/vf networks, batch_size 128
- Current 63K params is absurdly small for 704+ input dims. Typical RL would use 100K-500K

### Smogon Players: Missing Move Features

**[SMOGON]:**
1. Secondary effects are critical: Body Slam 30% para, Blizzard 10% PERMANENT freeze, Psychic 33% Special drop
2. Recoil: Submission, Double-Edge, Take Down all have 25% recoil
3. Self-destruct: Explosion (170bp) and Self-Destruct (130bp) kill your own mon
4. Fixed damage: Seismic Toss = level damage (always 100), shows as 0bp in obs

**[RBY]:** Body Slam (85bp + 30% para) beats Strength (100bp) in almost every situation. Without encoding secondary chances, agent can never learn this preference.

### Additional features per move (+4)

| Feature | Impact | Encoding | Example |
|---------|--------|----------|---------|
| secondary_status_chance | Critical | 0.0-1.0 probability | Body Slam = 0.3 |
| recoil | Important | 0.0-1.0 fraction | Take Down = 0.25 |
| self_destruct | Important | 0/1 flag | Explosion = 1.0 |
| fixed_damage | Niche | damage/100 | Seismic Toss = 1.0 |

### SmartHeuristicPlayer Needs Status Moves

Without status move demonstrations in BC data, the new obs features (category, status_effect, boost, heal, secondary_chance) are noise the NN ignores.

Updated SmartHeuristic to use:
- Thunder Wave / Stun Spore when opponent is faster or a sweeper
- Toxic against bulky opponents with >60% HP
- Sleep moves (Hypnosis, Sleep Powder) always valuable
- Swords Dance / Amnesia when HP > 70% and good matchup
- Recover / Softboiled when HP < 50%

## Decisions

1. **Obs space: 704 -> 928 dims** (14 features per move)
2. **NN: [64,64] shared -> [256,128] separate pi/vf** (~63K -> ~300K params)
3. **batch_size: 64 -> 128** for gradient stability
4. **SmartHeuristicPlayer updated** with status move logic
5. **Full BC retrain required** (obs space change is mid-vector, not append-only)

## Key Insight

The obs space and NN architecture changes are coupled: adding dims to a too-small NN just adds noise. The teacher, obs, and architecture must all upgrade together.
