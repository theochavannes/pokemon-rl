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

---

## 2026-04-02 — Git Cleanup & Developer Tooling Overhaul

### Git Cleanup
Removed all replay HTML files from tracking (10 files, ~150KB). Expanded `.gitignore` to cover: IDE files (`.idea/`, `.vscode/`), model checkpoints (`*.pt`, `*.pth`, `*.zip`), pytest cache, W&B tracking dirs, all of `runs/`, and meta-prompts. Previously only `runs/*/models/logs/replays/` were ignored — run metadata (`run_info.json`, `training_log.md`) was leaking into `git status`.

### Tooling Decisions

| Tool | Replaces | Why |
|------|----------|-----|
| **Ruff** | Pylint + Black + isort + Flake8 | Single tool, 10-100x faster, more rules, auto-fix |
| **Pre-commit hooks** | Nothing (had none) | Block bad commits before they enter history |
| **GitHub Actions CI** | pylint.yml (lint only) | Lint then test (sequential), pip caching, proper Ruff action |
| **CodeRabbit** | No code review | AI reviewer on every PR, free for public repos |
| **Dependabot** | Manual dep updates | Auto-PRs for security vulnerabilities |
| **Claude Code hooks** | CLAUDE.md formatting instructions | Deterministic auto-format on every AI edit |
| **Context7 MCP** | Claude's training data | Real-time docs for poke-env, SB3, PyTorch |

### What We Chose NOT To Add (and why)
- **W&B** — requires modifying training code, deferred to when we're ready for a training session
- **mypy/type checking** — valuable but high effort to retrofit; will add incrementally
- **Docker** — overkill for a solo research project running locally
- **MkDocs** — documentation site unnecessary until we share/publish
- **pixi/uv** — conda works fine for CUDA deps, no reason to switch mid-project

---

## 2026-04-02 — Logging & Observability Overhaul

### Problem
Zero Python logging — everything was bare `print()`. No log files written to disk. If the terminal closed during a run, all training output was lost. poke-env's internal logger emitted per-Pokemon warnings (e.g. about Vaporeon) hundreds of times per run, polluting the console.

### Solution

| Component | Before | After |
|-----------|--------|-------|
| **Console output** | `print()` only | `print()` + `logging.Logger` (dual output) |
| **File logs** | None | `runs/run_NNN/logs/training.log` — full DEBUG-level history |
| **Duplicate suppression** | None | `DuplicateFilter` — after 3 repeats, silences until a new message, then emits "suppressed N duplicates" |
| **poke-env noise** | `log_level=25` (between INFO/WARNING) | `log_level=40` (ERROR only) + WARNING filter on poke-env's Python logger |
| **Structured eval logs** | Formatted for humans only | `EVAL step=X vs=Y win_rate=Z ...` format — parseable for post-run analysis |

### Key Design Decisions
- **Dual output (print + logger):** Kept existing `print()` for console readability (progress bars, Unicode). Logger writes structured lines to file. Console uses DuplicateFilter(max=3), file uses DuplicateFilter(max=5) for more detail.
- **Heartbeats at DEBUG level:** Per-rollout step updates go to file only (DEBUG), not console, reducing noise during long runs.
- **poke-env raised to ERROR:** All useful battle info (win/loss, turns, actions) is already tracked by WinRateCallback. poke-env's INFO/WARNING messages about individual Pokemon are noise during training.

---

## 2026-04-03 — Privileged Matchup Evaluator Discussion (Expert Panel #2)

### The Problem
Agent plateaus at ~90% vs Random in Phase A. Training doesn't improve beyond the initial logit bias. Root cause: team RNG variance drowns the learning signal. Some games are unwinnable regardless of play, some are free wins — the agent can't tell the difference.

### The Proposal
A privileged evaluator that sees both full teams (information the agent never gets) and estimates win probability. Use this to scale/adjust rewards so the agent gets proper credit for skill vs. luck.

### Panel: Internal team + 3 external RL experts + competitive RBY player

**Key conclusions:**

1. **Additive baseline, NOT multiplicative scaling.** Multiplicative changes the optimization objective (maximizes "surprise" not "winning"). Additive preserves unbiased policy gradients while reducing variance. `reward = R - baseline(teams)`.

2. **Three approaches proposed (in order of effort):**
   - **VecNormalize** (free): SB3's built-in reward + observation normalization. Doesn't capture per-matchup variance but normalizes reward scale. We weren't using this at all.
   - **Tier-score heuristic**: Sum competitive tier ratings per team (Tauros=5, Magikarp=1). Captures ~60% of matchup variance per RBY expert. Computed retroactively from revealed Pokemon at episode end — no Showdown hooks needed.
   - **Asymmetric actor-critic**: Privileged team features fed to value function only. Policy still acts on partial info. Used in Meta's Diplomacy AI. Requires custom SB3 policy class.

3. **RBY expert insight: Type charts are NOT the main factor.** In Gen 1 randoms, matchup quality is dominated by: (a) which tier Pokemon you rolled, (b) speed tiers, (c) sleep move access, (d) special vs physical balance. A pure type-effectiveness heuristic would miss most of the signal.

4. **Static vs dynamic baseline:** A start-of-game evaluator handles "I got a bad team" variance. Per-turn re-evaluation as info is revealed is what the value function should learn. They target different variance sources — complementary, not redundant.

### Decision
Phased approach: VecNormalize first (free), then tier-score heuristic baseline, then asymmetric critic only if needed.

---

## 2026-04-03 — Behavioral Cloning Success + What's Next (Expert Panel #4)

### BC Results
- 99.0% validation accuracy imitating MaxDamagePlayer (142K transitions, 5000 games)
- Agent starts training at 94% vs Random, 70-80% vs RandAttacker (vs 2% before BC)
- BestMv% at 80%+ (was 25% random)
- Death spiral completely eliminated

### BC Policy Degradation Problem
PPO was eroding the BC policy — BestMv% dropped from 82% to 64% over 130K steps. Root cause: noisy advantages (from bad value function) + entropy bonus push toward uniform policy. Fixes applied:
- Reward simplified to faint differential + win/loss only (removed HP/status noise)
- Learning rate lowered from 3e-4 to 1e-4
- Dynamic entropy: ent_coef auto-increases (up to 0.1) when win rate <10%, auto-decreases (down to 0.01) when >50%. Prevents death spiral by forcing exploration when stuck.
- RandomDamagePlayer added as midpoint opponent (random moves, never switches) — fills gap between RandomPlayer and SoftmaxDamagePlayer

### Panel #4: Post-BC Training Strategy
14-expert panel (including DeepMind AlphaStar engineer, CMU professor, 2 Smogon RBY champions).

**Key insight:** Self-play alone between two agents that never switch is useless — they just trade attacks. Diverse opponents are needed to force the full skill set (especially switching).

**Unanimous recommendation (#1 of 35 ideas):** Mixed opponent pool using 4 existing envs:
- Env 0: Self-play (frozen BC model)
- Env 1: MaxDamagePlayer
- Env 2: TypeMatchupPlayer (forces switching)
- Env 3: SoftmaxDamagePlayer (temp annealing)

Plus KL penalty against BC policy to prevent forgetting, and BC regression tests.

Full ranked list of 35 ideas in notes/panel4_full_conversation.md

---

## 2026-04-03 — Sprint 3A: Mixed League Implementation

### The Shift: Sequential Curriculum → Simultaneous League

**Before (4 phases):** Random → RandomAttacker → SoftmaxDamage → Mixed+Self. Each phase trains against one opponent type until graduation, then moves on. Problem: skills learned in one phase erode in the next (catastrophic forgetting). The agent beats Random, then forgets how to play once it faces MaxDamage.

**After (1 phase, 4 envs):** All opponents every rollout.

| Env | Opponent | What It Tests |
|-----|----------|--------------|
| 0 | FrozenPolicyPlayer (BC warm-start) | Self-play — can it beat itself? |
| 1 | MaxDamagePlayer (pure) | Raw damage output |
| 2 | TypeMatchupPlayer (pure) | Type-aware switching |
| 3 | SoftmaxDamagePlayer (temp 2.0→0.1) | Smooth difficulty ramp |

### Anti-Forgetting: Conservative Clipping
`clip_range` lowered from 0.2 to 0.1. This is a well-known trick from RLHF literature (InstructGPT used it) — acts as an implicit KL constraint, limiting how far the policy can drift per PPO update. Preserves BC knowledge without needing an explicit KL penalty term.

### Self-Play Snapshots
Every 50K steps, the current model is frozen as the new self-play opponent and saved to `league/snapshot_NNNN.zip`. This builds a library of past selves for potential future fictitious self-play.

### Per-Opponent Win Rate Tracking
New TensorBoard metrics: `train/wr_SelfPlay`, `train/wr_MaxDmg`, `train/wr_TypeMatch`, `train/wr_SoftmaxDmg`. Also printed in console eval lines. Critical for diagnosing whether the agent is improving broadly or just beating one opponent type.

### Expanded Team
Sprint 3A assembled the largest team yet: core 11 agents plus 5 new specialists (second code reviewer, academic RL expert, DeepMind RL expert, staff engineer, test engineer). 16 experts total for 4 files changed.

---

## 2026-04-03 — Critical Bug: Silent Forfeit on Every Heuristic Game

### The Bug
First mixed_league training run (run_036) showed 25% win rate, 9-turn averages, 0% voluntary switches. Replays revealed the agent was forfeiting every game against heuristic opponents (envs 1-3). Only self-play (env 0) played real games.

### Root Cause
poke-env's `SinglesEnv.order_to_action()` validates opponent orders with `strict=True` by default. When a heuristic opponent's chosen move order doesn't match `battle.valid_orders` (due to battle state timing), it raises `ValueError`. Our `SB3Wrapper.step()` catches ValueError and calls `reset()` — which forfeits the current battle.

FrozenPolicyPlayer (env 0) was immune because it uses `strict=False` internally for its own order conversion.

### Fix
Pass `strict=False` to `Gen1Env` constructor. Invalid opponent orders now fall back to a random valid move instead of crashing the battle.

### Monitoring Added
- `SB3Wrapper` now tracks `desync_count` and `step_count`
- `WinRateCallback` reports desync stats in eval lines and TensorBoard
- Each desync logs a WARNING with the specific error message

### Lesson
Silent error handling that "recovers" by forfeiting is worse than crashing. The training ran 24K steps of garbage data with no indication anything was wrong. The win rate (25%) looked plausible for a new run. Without checking replays, this could have burned hours of GPU time.

---

## 2026-04-03 — PPO Cannot Learn Switching (Post-Mortem)

### Evidence
| Run | Steps | Vol.Switch% | BestMv% trend | clip_range | Switch bias |
|-----|-------|-------------|---------------|------------|-------------|
| run_037 | 68K | 0.0% | 89→85% (eroding) | 0.1 | -1.37 (BC default) |
| run_038 | 70K | 0.0% | 90→85% (eroding) | 0.2 | -0.69 (halved) |

### Root Cause Analysis
1. **Reward signal:** Switching gives 0.0 immediate reward. Benefit is 3-10 turns delayed. Value function (ExplVar=0.2) can't bridge the gap.
2. **BC bias:** MaxDamagePlayer never switches → BC learned -1.37 switch bias → PPO can't overcome it even with clip_range=0.2.
3. **No exploration pressure:** ent_coef=0.01 is too low to force random switches. Higher entropy would degrade move quality without targeted switching improvement.

### Decision: Re-do BC with a Switching Teacher
Pure RL cannot discover switching from the current reward signal. Need to teach switching through imitation, then let PPO refine.

Approach: Build a competitive-informed heuristic that combines strong move selection with strategic switching decisions. Generate BC data, retrain, then resume mixed league PPO with the new warm-start.

### Key Insight
This matches how all successful game AIs work: supervised pre-training on expert data → RL fine-tuning. The pre-training establishes the behavioral repertoire. If the expert data lacks a skill, RL alone won't discover it.

### Future: Human Replay Data
Showdown has a public replay API (replay.pokemonshowdown.com). Thousands of rated Gen 1 battles with real switching decisions. Would need a parsing pipeline to convert replay logs into (observation, action) pairs for BC. Significant engineering effort but unlimited expert-quality data. Bookmarked for if the heuristic teacher approach doesn't work.

---

## 2026-04-03 — Observation Space Gap: Agent Can't See Move Effects

### The Problem
Move features are: base_power, type, PP, effectiveness, accuracy. All status moves (Thunder Wave, Swords Dance, Toxic, Recover) appear as "0bp, type X, 100% acc" — indistinguishable from each other and indistinguishable from useless moves like Splash. The agent has zero reason to ever use a status move.

This is a fundamental ceiling: even perfect RL training cannot teach the agent to use Thunder Wave if the observation gives it no way to know Thunder Wave paralyzes.

### Missing Features (prioritized by competitive impact)

**Tier 1 — Must have:**
- Move category (physical/special/status): agent can't distinguish attack types
- Status effect type (paralyze/poison/boost/heal/other): agent can't value status moves
- Speed comparison: who goes first determines the entire turn

**Tier 2 — Important:**
- Own/opp alive counts: game state awareness
- Move priority flag: Quick Attack, Counter

**Tier 3 — Nice to have:**
- Volatile status (confusion, leech seed, substitute, reflect)
- Recharge/charging state (Hyper Beam, Fly)
- Trapping state (Wrap/Bind)

### Impact
Tier 1 adds ~107 dims (421 -> 528). Breaking change for existing checkpoints but obs_transfer.py handles expansion.

### Decision
Implemented: expanded 421->704 in sprint 3A. But run_043 showed same plateau — agent never learned to use new features because BC teacher doesn't use status moves and [64,64] NN can't process 704 dims effectively. Led to second expansion (see below).

---

## 2026-04-03 — Obs Space + NN Architecture Overhaul (Expert Panel)

### Panel: 2 RL Researchers + 2 Smogon Competitive RBY Players

**Additional move features needed (4 per move):**
- secondary_status_chance: Body Slam 30% para, Blizzard 10% freeze (PERMANENT in Gen 1)
- recoil: Take Down 25%, Double-Edge 25%
- self_destruct: Explosion/Self-Destruct kill your own mon
- fixed_damage: Seismic Toss = level damage (always 100), shows as 0bp otherwise

Obs: 704 -> 928 dims (14 features per move slot).

**NN architecture upgrade:**
- [64,64] shared extractor (63K params) -> [256,128] separate pi/vf (~300K params)
- batch_size 64 -> 128 for gradient stability with larger network
- First layer was bottlenecking: 704+ features squeezed through 64 neurons

**SmartHeuristicPlayer updated to use status moves:**
Without status move demonstrations in BC data, the new obs features are noise. Teacher updated to use Thunder Wave, Swords Dance, Recover, Toxic.

### Key Insight from Smogon Players
Secondary effects dominate Gen 1. Body Slam (85bp + 30% para) > Strength (100bp) in nearly every situation. Blizzard (120bp + 10% PERMANENT freeze) is the best move in the game. Without encoding these probabilities, the agent can never learn why pros prefer these moves.

---

## 2026-04-03 — Sprint 5: Observation Completeness (928 → 1222 dims)

### Changes (9 steps, all implemented)

**Per-move features expanded (14 → 19 features per move):**
1. **secondary_effect_type**: What the secondary does, not just the chance. Body Slam → PAR(0.8), Blizzard → FRZ(1.0), Psychic → stat drop(-0.17). Agent can now distinguish moves by their side effects.
2. **One-hot category**: Replaced single float (0.0/0.5/1.0) with [is_physical, is_special]. Status = [0,0]. Removes false ordering between categories.
3. **Separate heal/drain**: Were combined into one float. Now heal (Recover=0.5) and drain (Mega Drain=0.5) are distinct — they work completely differently.
4. **Trapping flag**: Wrap/Bind/Clamp/Fire Spin = 1.0. These prevent ALL opponent actions in Gen 1 — one of the most broken mechanics.
5. **Status immunity**: Target's type blocks this status? Toxic vs Poison = immune. Burn vs Fire = immune. Agent no longer wastes turns on doomed status moves.

**Own active move extras (+4 dims):**
6. **target_statused**: Is the opponent already statused? Prevents wasting Thunder Wave on already-paralyzed mons.

**Global features (+10 dims):**
7. **Volatile status (8)**: Substitute, Reflect, Light Screen, Confusion, Leech Seed — for both own and opponent.
8. **Opponent status move threat (1)**: Has the opponent revealed status moves? Increases switching urgency.
9. **Toxic escalation counter (1)**: How long Toxic has been ticking (0-15 → 0-1). Determines healing urgency.

### Dimension Breakdown
| Section | Old | New | Delta |
|---------|-----|-----|-------|
| Own moves | 56 | 80 | +24 |
| Own bench | 390 | 510 | +120 |
| Opp bench | 390 | 510 | +120 |
| Opp revealed | 56 | 76 | +20 |
| Global | 5 | 15 | +10 |
| **Total** | **928** | **1222** | **+294** |

### SmartHeuristicPlayer Updated
Added `_is_status_immune()` check — the teacher no longer attempts Toxic on Poison types or Burn on Fire types. BC data quality improves.

### Breaking Change
Mid-vector feature insertion. All existing checkpoints incompatible. Must retrain BC from scratch.

---

## 2026-04-03 — Sprint 6: Neural Network Architecture + Encoding

### Two-Tower Feature Extractor
Replaced flat MLP with a split architecture that processes own-team and opponent-team info separately before merging:
- **Own tower**: active(16) + moves(80) + bench(510) = 606 dims → 256 → 128
- **Opp tower**: active(15) + bench(510) + revealed(76) = 601 dims → 256 → 128
- **Global**: 15 dims (trapping, speed, alive, volatile, status_threat, toxic)
- **Merge**: 128 + 128 + 15 = 271 → 256 → action heads [128, 128]

Total params: ~500K (up from ~300K flat MLP).

### VecNormalize
SB3's built-in observation + reward normalization. Automatically maintains running mean/std for all 1222 obs features. Save/load stats alongside model checkpoints.

### Learning Rate Schedule
Linear anneal: 3e-4 → 1e-4 over training. Higher initial LR for exploration, lower for fine-tuning.

### Gradient Clipping
max_grad_norm=0.5 explicitly set (was SB3 default but never verified).

### Skipped: Learned Type Embeddings
Incompatible with VecNormalize (embeddings need integer indices, VecNormalize would normalize them to non-integers). Float type encoding with linear projection learns a similar representation.

### BC Script Updated
Behavioral cloning now uses the same two-tower architecture so the warm-start model matches the training architecture exactly.

---

## 2026-04-03 — Sprint 7: Training Refinement (1222 → 1559 dims)

### New Per-Move Features (25 per move, was 19)
- **must_recharge**: Hyper Beam requires skipping next turn (unless it KOs)
- **requires_sleep**: Dream Eater only works on sleeping targets
- **pp_max_norm**: max_pp/35 — indicates move scarcity (Blizzard has 5 PP vs Tackle's 35)
- **is_contact**: Physical contact flag
- **is_sound**: Sound-based move flag
- **ignore_accuracy**: Swift/moves that bypass accuracy checks

### Turn Phase Feature (+1 global dim)
`min(battle.turn / 50, 1.0)` — helps distinguish early game (set up, preserve mons) from endgame (trade aggressively).

### Self-Play Population (Sprint 7 Step 8)
Instead of always playing against the latest frozen model, the self-play opponent is randomly sampled from the league snapshot pool. Every 50K steps, a new snapshot is added. This prevents overfitting to a single opponent's weaknesses — the core idea from AlphaStar's fictitious self-play.

### Reverted: VecNormalize (run_045 post-mortem)
VecNormalize was removed. All obs features are already manually normalized to [0,1] or [-1,1]. Double-normalization distorted sparse binary features (status_immune: 0→-0.23, 1→4.32). run_045 with VecNormalize collapsed from 39%→16% WR; run_043 without it hit 62%.

### Reverted: Two-Tower Architecture (run_045 post-mortem)
PokemonFeatureExtractor deferred. BC cross-entropy couldn't optimize it — loss plateaued at epoch 1, accuracy stuck at 46.6% (vs 99% with flat MLP). Code preserved in `src/env/feature_extractor.py` for future use. Using flat MLP [256,128] pi/vf.

Full migration plan for re-introducing two-tower at the right time: `notes/two_tower_migration_plan.md`. Key insight from [RL]: train flat MLP to 60% WR first, then distill weights into two-tower structure. Every successful game AI introduces architectural complexity gradually — we tried to skip the crawl phase.

### Obs Dimension: 1559
| Section | S5 (928→1222) | S7 (1222→1559) | Delta |
|---------|---------------|----------------|-------|
| Per-move features | 19 | 25 | +6 |
| Own moves | 80 | 104 | +24 |
| Own bench | 510 | 654 | +144 |
| Opp bench | 510 | 654 | +144 |
| Opp revealed | 76 | 100 | +24 |
| Global | 15 | 16 | +1 |
| **Total** | **1222** | **1559** | **+337** |

---

## 2026-04-04 — Root Cause Analysis: Why PPO Cannot Learn (Full Team Review)

### The Problem
Across ALL training runs (043, 046, 047, 048), ExplVar stays negative and the BC policy erodes. The value function never learns, meaning PPO operates on random advantage estimates.

### Previous Diagnosis Was WRONG

`context_for_planning.md` blamed `share_features_extractor=True` for preventing the critic from learning independently. Full team review revealed:

**For MlpPolicy, FlattenExtractor has ZERO learnable parameters.** The actor MLP (1559→256→128→10) and critic MLP (1559→256→128→1) already have completely separate parameters. There is no shared learned layer. Setting `share_features_extractor=False` creates two FlattenExtractors (still no params each) — it changes nothing.

The warmstart_critic.py was training the correct parameters (mlp_extractor.value_net + value_net). It was NOT hampered by shared features — there are no shared features to be hampered by.

### Actual Root Cause: Chicken-and-Egg Value Bootstrap

PPO's GAE advantages depend on V(s). V(s) starts random. Its training targets (R_t = A_t + V_t) depend on V(s) itself. This creates a circular dependency:

1. Random V → garbage advantages → random policy gradient
2. Random gradient erodes BC prior
3. Policy changes → state distribution shifts → V can't catch up
4. Spiral continues until all BC knowledge is gone

This is a **fundamental limitation** of bootstrapped advantage estimation (GAE with lambda < 1). The value function is trying to learn from targets that are themselves corrupted by the bad value function.

### Why Critic Warmstart Alone Failed (Run 047)

The warmstart trained the critic offline on (obs, return) data from 1000 games. But during PPO:
- The critic's predictions are used to compute GAE targets
- GAE targets are used to re-train the critic
- The online updates quickly overwrite the warmstart weights with self-referential garbage
- Within 10-20K steps, the warmstart effect is gone

### The Fix: Actor Freeze Protocol

**Freeze actor → train only critic → unfreeze when ExplVar > 0.**

By freezing the actor, the policy (and thus the state distribution) stays fixed at BC quality. The value function can learn V^{π_BC}(s) as a standard supervised regression problem — no circular dependency because the policy doesn't change.

Once V is reliable (ExplVar > 0), unfreeze the actor with conservative settings. PPO can now compute accurate advantages and improve the policy in the correct direction.

### Secondary Finding: Monitoring Bug

`callbacks.py:_move_damages_from_obs` uses stride 14 for moves, but the actual obs layout has stride 26 (25 features + 1 target_statused). BestMv% and DmgEff metrics are wrong for moves 1-3 in ALL runs since Sprint 5.

### Decision

Implement actor freeze protocol. See `notes/plan_fix_value_function_2026-04-05.md` for full execution plan.

| What | Action |
|------|--------|
| `share_features_extractor=False` | **NOT doing it** — no functional difference for MlpPolicy |
| Critic warmstart | Improve (more games, match reward fn) but don't rely on it alone |
| Actor freezing | **PRIMARY FIX** — freeze actor for first 100K PPO steps |
| Conservative unfreeze | clip_range 0.1→0.2 ramp, lower effective actor LR |
| Monitoring fix | Fix stride bug in callbacks.py |

---

## 2026-04-04 — Process Discipline: Antigravity Skills Added to Team

### The Problem
Every plan the team produced was built on an unverified hypothesis. 100% failure rate. The pattern: hypothesize root cause → build 500-line plan → execute → discover hypothesis was wrong → repeat.

Examples:
- "shared_features_extractor causes critic to not learn" → wrong (FlattenExtractor has zero params)
- "critic warmstart will fix it" → failed (online PPO updates overwrote warmstart in 10-20K steps)
- "two-tower architecture will help" → made it worse (BC couldn't optimize it, 46.6% accuracy)

### Decision
Added 5 process discipline skills to AGENTS.md. These are not new team members — they are process gates the existing team must follow.

| Skill | What it enforces |
|---|---|
| `/systematic-debugging` | No fixes without verified root cause (4-phase) |
| `/phase-gated-debugging` | Can't edit source code until root cause confirmed by evidence |
| `/planning-with-files` | Save findings every 2 actions during research/debug sessions |
| `/closed-loop-delivery` | Task incomplete until acceptance criteria verified in actual metrics |
| `/evaluation` | Regression tests catch training quality regressions automatically |

**Hard rule added:** Before any plan proposing code changes, team must complete a debugging pass with empirical evidence.

### New Prompt Structure
The next execution prompt (`notes/prompt_diagnostic_plan_2026-04-05.md`) separates diagnosis from planning with hard gates. Phase 1 forces empirical root cause verification. Phase 2 is team design review. Phase 3 writes the plan ONLY after evidence exists. Phase 3.2 requires a quick validation experiment before committing to the full plan.

### Rationale
The team has domain expertise but lacked process discipline. The skills don't replace the team — they add the verification rigor that was missing. The cost is ~30 min of upfront diagnosis. The benefit is not wasting days on plans built on wrong assumptions.

---

## 2026-04-05 �� Deep Diagnostic: The "Chicken-and-Egg" Hypothesis Was INCOMPLETE

### The Prompt That Changed Everything

Prompted with `/systematic-debugging` methodology and hard gates requiring
empirical evidence before ANY fix proposals. The previous plan
(`notes/plan_fix_value_function_2026-04-05.md`) proposed actor freezing based on
the theory that the value function "always stays negative." The instruction was:
**PROVE OR DISPROVE this before building another plan.**

### Finding #1: Run 043 Had POSITIVE ExplVar

The foundational claim — "ExplVar stays negative across all runs (043, 046, 047,
048)" — was wrong. Run 043 training logs show:

| Step range | ExplVar readings |
|------------|------------------|
| 0-50K | +0.01, +0.12, +0.05, +0.11, +0.15, +0.14, +0.02, +0.08 |
| 50K-100K | +0.19, +0.17, +0.26, +0.17, +0.13, +0.10, +0.12 |

The value function WAS learning in run 043. It explained 1-26% of return
variance. But PPO STILL didn't improve — win rate stuck at ~50%.

**Implication:** Fixing the value function (the entire goal of the previous plan)
is necessary but NOT sufficient. There's a deeper problem.

### Finding #2: Architecture Changes Don't Explain the Difference

SB3 v1.8.0+ has NO shared hidden layers. Checked the source:
`MlpExtractor` in `torch_layers.py` creates separate `nn.Sequential` for
`policy_net` and `value_net` regardless of whether `net_arch` is a list or dict.

Both `net_arch=[64,64]` and `net_arch=dict(pi=[256,128], vf=[256,128])` create
completely independent networks with zero shared parameters.

**Implication:** The "shared features extractor" diagnosis from the previous day
was wrong (correctly identified), AND the hypothesis that run 043's architecture
was fundamentally different is ALSO wrong. Same architecture pattern, just
different sizes.

### Finding #3: Three Interacting Problems, Not One

| Layer | Problem | Evidence |
|-------|---------|----------|
| 1. Value init | 432K-param value_net starts random, can't converge at 3e-4 LR | Run 043 ([64,64], 104K params, LR 1e-4): ExplVar positive by 8K. Runs 046/048 ([256,128], 432K params, LR 3e-4): ExplVar mostly negative |
| 2. BC erosion | PPO updates destroy BC knowledge at ~0.04%/K steps | BestMv% declines monotonically in ALL runs regardless of ExplVar |
| 3. Signal/noise | Even with positive ExplVar, advantage estimates are too noisy for improvement | Run 043: ExplVar up to 0.26, win rate still 50% over 365K steps |

The previous plan proposed actor freezing, which addresses Layer 2 only.

### Finding #4: Key Hyperparameter Differences Between Runs

| Parameter | Run 043 (partial success) | Runs 046/048 (failure) |
|-----------|--------------------------|------------------------|
| net_arch | [64,64] (default) | dict(pi=[256,128], vf=[256,128]) |
| learning_rate | 1e-4 constant | 3e-4 → 1e-4 linear |
| batch_size | 64 | 128 |
| Value net params | ~104K | ~432K |

The 3x higher starting LR in runs 046/048 is a likely contributor to both:
- Faster BC erosion (larger policy updates)
- Value function instability (larger, noisier value updates)

### Decision: Three-Pronged Fix

| What | Why |
|------|-----|
| **Copy policy_net → value_net** (NEW) | Give value function pre-trained features from BC. Reduces training from 432K random params to 129-param value head. |
| **Actor freeze** (from previous plan, kept) | Protect BC policy while value head fine-tunes on stable distribution. |
| **gae_lambda=1.0 during warmup** (NEW) | Use pure MC returns during freeze phase. No value bootstrap = no circular dependency. |
| **LR=1e-4 constant** (CHANGED from previous plan) | Run 043's LR worked. The 3e-4 schedule was too aggressive. |
| **clip_range=0.1** (CHANGED from previous plan) | Conservative policy updates to slow BC erosion. |

### What Was Kept From the Previous Plan
- Actor freeze/unfreeze mechanism (correct approach for Layer 2)
- Fix `_move_damages_from_obs` stride bug in callbacks.py
- Fix stale OBS_DIM in behavioral_cloning.py
- Escalation plans (refined with new failure modes)

### What Was Changed
- Root cause expanded from 1 layer to 3
- Added weight copy (addresses Layer 1 — previous plan had no fix for this)
- Added gae_lambda=1.0 during warmup (addresses Layer 3's bootstrap issue)
- Changed LR from schedule to constant 1e-4 (evidence-based)
- Added two quick validation experiments BEFORE implementation (30 min each)
- Added "ExplVar positive but no improvement" escalation (run 043 scenario)
- Reduced warmup steps from 100K to 50K (weight copy means faster convergence)

### What Was Rejected
- Separate critic warmstart script (replaced by weight copy — simpler, faster)
- Two-tower architecture (still deferred — BC can't optimize it)
- VecNormalize (still proven harmful)
- share_features_extractor=False (no functional effect for MlpPolicy)

### Validation Gate
Two 30-minute experiments must complete before committing to the full fix:
1. Train from scratch (no BC) → does ExplVar go positive?
2. BC warm-start with LR=1e-4 constant → does ExplVar match run 043?

Full plan at `notes/plan_2026-04-06.md`.

---

## 2026-04-05 — Validation Experiments: n_epochs is the Key (Verified)

### The Smoking Gun: bc_warmstart.zip Already Has ExplVar 0.99

Before running experiments, investigated the critic warmstart directly. The file
`models/bc_warmstart.zip` was saved AFTER `warmstart_critic.py` ran — it already
has a trained value function with ExplVar=0.99 on its training data.

Run 047 used this model and ExplVar collapsed to -0.26 in 23K PPO steps. The
weight diffs were TINY (mean abs 0.02 per layer). PPO's online value updates
destroyed a nearly-perfect value function with small perturbations.

### Experiment Results (7 configurations tested)

| Config | ExplVar@50K | Key Variable |
|--------|-------------|-------------|
| From scratch, LR=3e-4, n_epochs=10 | +0.106 | Baseline: value CAN learn |
| BC+warmstart, LR=1e-4, n_epochs=10 | -0.121 | n_epochs=10 destroys warmstart |
| BC+warmstart, LR=3e-4, n_epochs=10 | -0.166 | Higher LR makes it worse |
| **BC+warmstart, LR=1e-4, n_epochs=3** | **+0.048** | **n_epochs=3 preserves warmstart** |
| BC+freeze, LR=1e-4, n_epochs=10 | -0.134 | Freeze alone not enough |
| **BC+freeze, LR=1e-4, n_epochs=3** | **+0.185** | **WINNER: freeze + fewer epochs** |
| BC+freeze, gae_lambda=1.0 | -0.802 | Pure MC returns catastrophic |

### Root Cause: n_epochs=10 Overfits Per Rollout

With n_epochs=10 and batch_size=128 on 8192 samples per rollout, PPO does 640
gradient steps per rollout on the value function. A 432K-param network memorizes
this small dataset, losing generalization. With n_epochs=3, it does 192 gradient
steps — enough to learn but not enough to overfit.

### Decision: Applied Configuration

```python
PPO_KWARGS = dict(
    n_epochs=3,          # Was 10 — prevents value function overfitting per rollout
    clip_range=0.1,      # Was 0.2 — conservative to preserve BC knowledge
    learning_rate=1e-4,  # Was 3e-4→1e-4 — constant rate, proven in run 043
)
CRITIC_WARMUP_STEPS = 50_000  # Actor frozen, only critic trains
```

### Run 052 Status (in progress)

Warmup completed successfully (ExplVar climbing from -0.03 to +0.17 during
freeze). Actor unfroze at step ~57K. Curriculum phase "League" running.

Early post-unfreeze metrics:
- ExplVar: +0.067 (positive, stable)
- BestMv%: 58.0-58.3% (BC knowledge preserved — NOT eroding)
- Win rate: 45-67% (oscillating but not collapsing)

This is the first run with the [256,128] architecture to maintain positive
ExplVar AND stable BC knowledge simultaneously.

### What Was Rejected (with evidence)

| Rejected | Reason | Evidence |
|----------|--------|----------|
| gae_lambda=1.0 | Pure MC returns catastrophic for value learning | Experiment C3: ExplVar = -0.80 |
| Actor freeze alone (n_epochs=10) | Freeze not sufficient without reducing epochs | Experiment C: ExplVar = -0.13 |
| Weight copy as primary fix | Warmstart already works (ExplVar 0.99). Weight copy kept in BC script as defensive safety net, not the primary fix | Warmstart investigation |
| Higher LR (3e-4) | Destroys warmstart faster | Experiment B2 vs B |
