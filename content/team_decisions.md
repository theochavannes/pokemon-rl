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
Pending — evaluating whether to expand now (before more training time is invested) or let current run_041 with SmartHeuristic BC complete first.
