# Expert Panel Discussion #3: Why The Agent Can't Learn

**Date:** 2026-04-03
**Context:** run_032 — Phase A graduated at 96%→91% in 7K steps, Phase B stuck at 50-55% for 200K steps, Phase C collapsed to 0-3% for 400K steps, Phase D declining from 66% to 22%.

## Participants

**Internal team:** [ML], [RL], [SE], [REVIEW], [SYS], [GYM], [SB3], [RBY], [PM], [MEDIA]
**External:**
- [Smogon1] — Top-5 Smogon RBY ladder, 3x tournament winner
- [Smogon2] — RBY teambuilder/analyst, Smogon tiering council
- [RL-Debug] — Principal Research Scientist, specializes in diagnosing RL training failures (ex-DeepMind)
- [RL-Curriculum] — Associate Professor, UC Berkeley, published on curriculum learning and catastrophic forgetting

---

## Part 1: Diagnosing The Collapse

### The Data

**Phase A (Random):** 96%→91% in 7K steps, graduated. Logit bias does all the work.

**Phase B (RandAttacker):** 50-55% for 200K steps. No learning trend. Coin-flip territory.

**Phase C (SoftmaxDmg):**
- Step 212K: 20% (first eval)
- Step 213K: drops to 1-3%
- Steps 213K–613K: stuck at 0-3% for 400K steps
- Avg battle length: 7-11 turns (normal = 45)
- Agent is dying in 3-4 real turns every game

**Phase D (Mixed+Self):** Started at 66% (opponents 95% random), declined to 22%.

### Root Cause: The Agent Never Learned To Play

[RL-Debug]: The policy learned in Phases A/B is tuned for opponents that act nearly randomly. Against Random, attacking with any move wins because Random wastes turns switching pointlessly. The 90% win rate is entirely the logit bias, not learned behavior.

Evidence: action distribution is identical across all phases:
- Phase A: moves 82%, switches 18%
- Phase B: moves 82%, switches 18%
- Phase C: moves 76%, switches 24%
- Move distribution uniform across slots everywhere

### The Death Spiral

[RL-Curriculum]: Textbook distribution shift → death spiral:
1. Policy is bad → loses every game → gets only negative rewards
2. Value function learns "every state is terrible" → V(s) ≈ -3.0 everywhere
3. Advantages ≈ 0 (reward matches value prediction) → no gradient
4. Policy stops updating → stuck at 0-3% forever

### The Competitive Player's Perspective

[Smogon1]: Even at temperature 2.0, SoftmaxDamagePlayer always prefers higher-damage moves. Random picks Splash and Thunderbolt equally. Softmax picks Thunderbolt 10x more. That's a huge difference in incoming damage.

In competitive RBY, this is about "tempo" — if you fall behind on damage output per turn, the game snowballs. A policy viable against Random becomes unviable against even soft SoftmaxDamage.

### The Value Function Problem

[RL-Curriculum]: Per-step reward from shaping ≈ 0.1-0.2. With explained_variance of 0.15, 85% of return variance is noise in the advantage estimate. The agent is training on mostly-random gradient updates.

---

## Part 2: The 31 Ideas

### Idea 1: Reduce observation space for early phases
421 dims → [64,64] MLP is enormous. Start with ~30 dims (active matchup only). Add bench info later.

### Idea 2: Single continuous curriculum (no phase transitions)
One SoftmaxDamage opponent, temperature 100.0→0.1 over entire run. No distribution shift.

### Idea 3: Massively increase step budget
200K is nothing. Try 2M+ per phase minimum.

### Idea 4: Start with 1v1, not 6v6
Train on single Pokemon each (no switching). Then 2v2, 3v3, 6v6.

### Idea 5: Fixed teams instead of random
gen1randombattle is hardest possible format. Start with fixed teams so the network sees the same matchups repeatedly.

### Idea 6: Progressive network growing
Start with [16,16] on simplified problem, expand to [64,64] then [128,128].

### Idea 7: Train move selection in isolation
Simplified env: one Pokemon each, pick one of 4 moves, battle resolves in one turn. Pure "which move is best?"

### Idea 8: Per-move feedback reward
+0.1 if super-effective, -0.1 if not-very-effective, +0.2 if highest-damage option. Direct credit assignment.

### Idea 9: Behavioral cloning warm-start
Generate 10,000 MaxDamage vs Random games. Train policy supervised (cross-entropy) to imitate MaxDamage. Then fine-tune with PPO.

### Idea 10: Hierarchical action space
Binary "attack or switch?" then move/Pokemon selection. Reduces effective action space.

### Idea 11: Value function clipping at transitions
Disable clip_range_vf at phase transitions so V(s) can adapt quickly.

### Idea 12: Separate larger value function network
net_arch=dict(pi=[64,64], vf=[128,128]) — give V(s) more capacity.

### Idea 13: Inspect replay losses
Pick 10 Phase C losses and manually read them. Understand the failure mode before building fixes.

### Idea 14: Add move category (physical/special)
Gen 1: all moves of a type are physical or special. Without this, agent can't predict damage for ~40% of moves.

### Idea 15: Speed comparison feature
own_speed / (own_speed + opp_speed) — "am I faster?" is the most important binary in RBY.

### Idea 16: "Can I KO?" signal
estimated_damage / opp_remaining_hp for each move. "Does this move kill?" is THE most important decision.

### Idea 17: Reward clipping/normalization
Clip per-step rewards to [-1, 1] to prevent shaped reward swings from dominating.

### Idea 18: Reset optimizer on phase transitions
Adam momentum from Phase A is wrong for Phase B. Reset Adam state at transitions.

### Idea 19: Elastic Weight Consolidation (EWC)
Regularize against changing weights important in previous phase. Prevents catastrophic forgetting.

### Idea 20: Population-Based Training (PBT)
8-16 parallel agents, worst clone best + mutate hyperparams. Auto-discovers right settings.

### Idea 21: Train directly against MaxDamage
Skip intermediates. If it can't beat MaxDamage in 2M steps, nothing else matters.

### Idea 22: LSTM/GRU policy
Add memory for multi-turn patterns. "They switched to Starmie last time I sent out Charizard."

### Idea 23: Move history in observation
Encode what moves the agent used this battle and results.

### Idea 24: Simplify to 2 actions
"Attack with best move" or "switch to best counter." Train the when-to-switch decision first.

### Idea 25: Deterministic Showdown
Remove damage rolls, crits, miss chance. Cleaner reward signal.

### Idea 26: Win prediction auxiliary loss
Auxiliary task: predict whether you'll win. Forces useful feature representations.

### Idea 27: Hindsight experience replay
After a loss, relabel: "if you had picked move X at turn Y..." Increases useful signal from losses.

### Idea 28: Pre-trained game-state encoder
Unsupervised/self-supervised encoder on heuristic games → compact observation.

### Idea 29: Self-play from the start
Self-play provides natural difficulty scaling. Start from step 0 instead of Phase D.

### Idea 30: MCTS component
Policy as prior for shallow search (2-3 turns). What AlphaZero does.

### Idea 31: Return decomposition
Decompose returns into "this turn's action caused X" vs "future actions caused Y."

---

## Part 3: Full Ranking

| Rank | # | Idea | Effort |
|------|---|------|--------|
| 1 | 9 | Behavioral cloning warm-start | Low |
| 2 | 2 | Single continuous curriculum | Low |
| 3 | 7 | Train move selection in isolation | Medium |
| 4 | 4 | Start with 1v1, not 6v6 | Medium |
| 5 | 8 | Per-move feedback reward | Low |
| 6 | 16 | Add "can I KO?" signal | Low |
| 7 | 15 | Speed comparison feature | Trivial |
| 8 | 13 | Inspect replay losses | Trivial |
| 9 | 12 | Separate value function network | Low |
| 10 | 5 | Fixed teams instead of random | Medium |
| 11 | 1 | Reduced observation space for early phases | Medium |
| 12 | 18 | Reset optimizer on phase transitions | Trivial |
| 13 | 14 | Add move category (phys/special) | Low |
| 14 | 10 | Hierarchical action space | High |
| 15 | 3 | Increase step budget | Trivial |
| 16 | 11 | Disable value function clipping at transitions | Trivial |
| 17 | 21 | Train directly against MaxDamage | Trivial |
| 18 | 6 | Progressive network growing | High |
| 19 | 17 | Reward clipping/normalization | Low |
| 20 | 29 | Self-play from the start | Low |
| 21 | 22 | LSTM/GRU policy | High |
| 22 | 26 | Win prediction auxiliary loss | Medium |
| 23 | 25 | Deterministic Showdown | Medium |
| 24 | 19 | Elastic Weight Consolidation | Medium |
| 25 | 28 | Pre-trained encoder | High |
| 26 | 20 | Population-Based Training | Very High |
| 27 | 23 | Move history in observation | Low |
| 28 | 24 | Simplify to 2 actions | Medium |
| 29 | 30 | MCTS | Very High |
| 30 | 27 | Hindsight experience replay | Very High |
| 31 | 31 | Return decomposition | Very High |

---

## Part 4: Recommended Implementation Order

1. Inspect replay losses (30 min, reframes everything)
2. Add KO estimation + speed comparison features (trivial, high info)
3. Per-move feedback reward (direct credit for good moves)
4. Behavioral cloning warm-start (non-random starting policy)
5. Single continuous curriculum (eliminate phase transition shock)
6. Verify reward flow end-to-end (rule out bugs)
7. Separate value function network (more capacity for V(s))
8. Reset optimizer on phase transitions (if keeping discrete phases)
