# Team Decision Log

Running record of architectural decisions, debates, and conclusions.
Updated after every significant team discussion.

---

## 2026-04-02 — PPO vs DQN

**Decision:** PPO (MaskablePPO). DQN's experience replay goes stale during self-play when opponent policy changes. PPO is on-policy and has first-class action masking in sb3-contrib.

---

## 2026-04-02 — Observation Space Evolution

| Version | Dims | What changed |
|---------|------|-------------|
| v1 | 64 | Basic: HP, boosts, status, types, moves |
| v2 | 127 | Added bench types, opponent bench, accuracy, trapping |
| v3 | 139 | Added acc/eva boosts, base stats, opponent revealed moves |
| v4 | 153 | Added Pokemon level (gen1randombattle has varying levels 58-100) |
| v5 | 435 | Full bench obs — all 4 moves per bench mon w/ effectiveness |
| v6 (current) | 421 | Removed level (forced to 100 in Showdown config, always 1.0) |

Transfer learning via `obs_transfer.py` handles loading old checkpoints into new obs spaces — zero-pads new input columns. v5→v6 is a breaking change (feature positions shifted).

---

## 2026-04-02 — Reward Shaping Journey

This was the single biggest source of problems. Timeline:

| Attempt | Values | Result |
|---------|--------|--------|
| Sparse only | fainted=0, hp=0, victory=1.0 | 0% win rate. No gradient signal at all. |
| First shaping | fainted=0.15, hp=0.10, victory=1.0 | Worked vs Random (~64%). Too weak vs MaxDamage. |
| Reduced shaping | fainted=0.05, hp=0.05, status=0.02, victory=1.0 | Agent couldn't differentiate moves. Signal drowned by terminal reward. |
| Expert panel recommendation | fainted=0.5, hp=0.5, status=0.1, victory=1.0 | **PENDING** — 10x increase to make intermediate events matter |

**Key insight from expert panel:** With fainted=0.05 and hp=0.05, total shaping across a full game sums to ~0.6. That's less than the 1.0 victory bonus, but the per-step signal is ~0.01 — too small for PPO to learn which moves cause the reward. 10x makes each KO worth 0.5, clearly visible in the gradient.

---

## 2026-04-02 — Opponent Architecture

### Heuristic Opponents (4 types)
| Opponent | Strategy | Skill tested |
|----------|----------|-------------|
| MaxDamage | Always highest base_power x type_effectiveness | Survive raw offense |
| TypeMatchup | Best type move, switches out of bad matchups | Handle type-aware play |
| Stall | Status moves first (TWave, Toxic), then damage | Break through walls |
| AggressiveSwitcher | Switches to type-counter aggressively | Punish switches |

**Bug found:** StallPlayer used `m.category == "status"` (string) but poke-env uses `MoveCategory.STATUS` (enum). StallPlayer was actually a MaxDamage clone. Fixed.

### Epsilon Blending
Each heuristic opponent uses `_EpsilonMixin` — per-turn coin flip between frozen self-play policy and pure heuristic strategy. No random play once a frozen model is loaded.

---

## 2026-04-02 — Curriculum Evolution

### v1: Sequential phases (Random -> MaxDamage)
**Problem:** Agent graduates Phase A, hits MaxDamage wall (85% -> 2%), never recovers.
**Root cause:** Skills learned vs Random don't transfer. No gradual difficulty ramp.

### v2: Epsilon annealing with step triggers
**Problem:** Epsilon drops 0.1 per trigger, death spiral. Agent briefly hits threshold, epsilon drops, win rate crashes, slowly recovers, hits threshold again, epsilon drops again.

### v3: Mixed opponents with smooth epsilon
**Problem:** Phases graduated before epsilon could anneal (win rate target hit while opponents still 95% random).

### v4: Epsilon = f(win_rate) with rate limiting
- `epsilon = 1.0 - win_rate_over_500_games`
- Max drop 0.03 per eval, max rise 0.05 per eval
- Phase can't graduate until epsilon reaches 0.0
**Status:** Still stuck at 20-35% vs MaxDamage even with epsilon 0.68

### v5 (proposed): 10x reward shaping
Expert panel diagnosis: reward signal too weak by an order of magnitude. Agent can't learn which moves are better because per-step reward differences are ~0.01.

---

## 2026-04-02 — Parallel Training

- DummyVecEnv (not SubprocVecEnv) because epsilon annealing needs opponent object access
- 4 envs currently (laptop has 20 CPU cores, RTX 4050 6GB — plenty of headroom)
- Each env gets a different opponent from the mixed pool
- Self-play opponent on env3, heuristics on env0-2

---

## 2026-04-02 — PPO Hyperparameter History

| Config | n_steps | batch | epochs | ent_coef | net_arch | Result |
|--------|---------|-------|--------|----------|----------|--------|
| Default | 2048 | 64 | 10 | 0.0 | [64,64] | Worked vs Random, stuck vs MaxDamage |
| Big batch | 4096 | 512 | 4 | 0.01 | [128,128] | Learned slower than default |
| Bigger net | 4096 | 1024 | 3 | 0.05 | [256,256] | Stuck at 10-20%, too many params |
| Reverted | 2048 | 64 | 10 | 0.0 | [64,64] | Best performer so far |

**Lesson:** The default SB3 config for PPO was closest to optimal. Bigger networks and batches didn't help — the bottleneck was reward signal, not model capacity.

---

## 2026-04-02 — Known poke-env Issues

1. **Disable desync:** poke-env occasionally sends moves that were Disabled mid-turn. Showdown rejects and picks default action. Non-fatal.
2. **Switch-to-active:** Action mask doesn't always filter "switch to already active Pokemon." Showdown rejects. Non-fatal.
3. **Forced switch desync:** `order_to_action` raises ValueError when opponent's forced switch doesn't match valid orders. Caught in SB3Wrapper, returns neutral reward (0.0).

---

## 2026-04-02 — Codebase Improvement Pass (10-Agent Audit)

### Bugs Found and Fixed
1. **`shaping_decay_battles` never passed to WinRateCallback** — used hardcoded default 5000 instead of explicit parameter. Fixed.
2. **`selfplay_train.py` missing `env` param in WinRateCallback** — shaping decay was silently disabled during self-play. Fixed.
3. **`reset_num_timesteps` logic always False** — `model is None` check happened after model was already assigned. Fixed with `is_fresh_run` flag.
4. **Temperature annealing completely broken for SoftmaxDamagePlayer** — callback checked `hasattr(opp, "epsilon")` which returned False for SoftmaxDamagePlayer (has `temperature`, not `epsilon`). Phase C could never graduate via annealing — only by hitting max_steps. Fixed.
5. **`opponent_epsilon` parameter name confusing** — renamed to `opponent_difficulty` since it means temperature for softmax and epsilon for epsilon-greedy opponents.
6. **Temperature not saved on phase completion** — resume only saved epsilon, not temperature. `SoftmaxDamagePlayer` temperature is now persisted so Phase C resumes at the correct difficulty.

### Obs Space Change: 435 → 421
Removed `level` feature (14 dims total: 1 per Pokemon slot). All Pokemon forced to level 100 in Showdown — the feature was always 1.0, wasting dimensions. This is a breaking change for existing checkpoints (feature positions shift throughout the observation vector).

### Testing
Added 23 unit tests covering: `embed_battle` shape/dtype/values, `_expected_damage`, `SoftmaxDamagePlayer` temperature behavior, `_EpsilonMixin` blend, `obs_transfer` weight expansion, reward shaping decay, and integration smoke tests.
