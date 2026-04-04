# Diagnostic Findings: Why PPO Cannot Learn
## Date: 2026-04-05

---

## Previous Hypothesis: "Chicken-and-Egg Value Bootstrap" — PARTIALLY CORRECT

The plan at `notes/plan_fix_value_function_2026-04-05.md` diagnosed a circular
dependency: random V → garbage advantages → random gradient → erodes BC → state
shift → V can't learn. The proposed fix was actor freezing.

**This diagnosis is directionally correct but incomplete.** The evidence below
shows the problem has THREE layers, not one. Actor freezing alone won't work.

---

## Evidence: Run 043 vs Runs 046/047/048

### Run 043 — ExplVar IS POSITIVE
| Metric | Value |
|--------|-------|
| BC teacher | MaxDamagePlayer (99% accuracy) |
| Architecture | `net_arch=[64,64]` (SB3 default) |
| Learning rate | 1e-4 constant |
| Batch size | 64 |
| Obs dims | 1559 |
| ExplVar | **+0.01 to +0.26** (consistently positive from 8K steps) |
| Win rate | Stuck at ~50% over 365K steps |
| BestMv% | 84% → 70% (slow erosion) |

### Run 048 — ExplVar MOSTLY NEGATIVE
| Metric | Value |
|--------|-------|
| BC teacher | SmartHeuristicPlayer (92.8% accuracy) |
| Architecture | `net_arch=dict(pi=[256,128], vf=[256,128])` |
| Learning rate | 3e-4 → 1e-4 linear schedule |
| Batch size | 128 |
| Obs dims | 1559 |
| ExplVar | **-0.16 to +0.08** (mostly negative, briefly positive ~80-150K steps) |
| Win rate | Stuck at ~50% over 236K steps |
| BestMv% | 32% → 25.6% (steady erosion) |

### The Critical Fact Previous Analysis Missed

**Run 043 had POSITIVE ExplVar the entire run.** The statement in
`context_for_planning.md` that "ExplVar stays negative across all runs
(043, 046, 047, 048)" is INCORRECT for run 043.

**But even with positive ExplVar, run 043 NEVER improved beyond 50% win rate.**
This means the value function being "correct" (ExplVar > 0) is NECESSARY but
NOT SUFFICIENT for PPO to improve.

---

## Verified Architecture: No Shared Layers in EITHER Run

SB3 v1.8.0+ eliminated shared hidden layers. Checked the source:

```
# MlpExtractor (stable_baselines3/common/torch_layers.py)
# BOTH net_arch=[64,64] and net_arch=dict(pi=..., vf=...)
# create SEPARATE nn.Sequential modules for policy_net and value_net.
# No shared learnable parameters whatsoever.
```

The previous plan correctly noted FlattenExtractor has zero params. But it
also claimed "the actor and critic already have completely separate MLPs" as
if this were specific to the dict config. In fact, ALL SB3 MlpPolicy configs
since v1.8.0 have separate MLPs. Run 043's [64,64] was also separate.

**Implication:** The shared-vs-separate architecture is NOT the explanation for
the ExplVar difference between run 043 and runs 046/048.

---

## Root Cause: THREE Interacting Problems

### Problem 1: Value Network Too Large for Random Init

Run 043 value_net: Linear(1559, 64) → Linear(64, 64) → Linear(64, 1)
  = ~104K trainable parameters

Run 048 value_net: Linear(1559, 256) → Linear(256, 128) → Linear(128, 1)
  = ~432K trainable parameters

The [256,128] value network has **4.2x more parameters** starting from random
initialization. Combined with a 3x higher starting LR (3e-4 vs 1e-4), the
larger network oscillates instead of converging.

**Evidence:**
- Run 043 ExplVar positive by step 8K (fast convergence with 104K params)
- Run 048 ExplVar briefly positive around 80K-150K steps (slow, oscillating)
- Run 046 ExplVar negative throughout except one reading at 74K (+0.116)

### Problem 2: PPO Erodes BC Knowledge Before Value Catches Up

PPO updates the policy every `n_steps * n_envs` = 8192 steps using advantages
computed from the (bad) value function. Each update slightly degrades the BC
prior.

**Evidence:**
- BestMv% declines monotonically in ALL runs:
  - Run 043: 84% → 70% (0.04%/K steps)
  - Run 046: 30% → 25.6% (0.04%/K steps)
  - Run 048: 32% → 25.6% (0.03%/K steps)
- The erosion rate is similar across runs (~0.03-0.04%/K steps)
- Erosion continues regardless of ExplVar sign

### Problem 3: Signal-to-Noise Too Low for Policy Improvement

Even with positive ExplVar (run 043), win rate stays at ~50%. Why?

1. **ExplVar 0.1-0.2 means the value function explains only 10-20% of variance.**
   The remaining 80-90% is noise in the advantage estimates. PPO's policy
   gradient is overwhelmed by this noise.

2. **Return variance is dominated by random team composition.**
   gen1randombattle assigns random teams. Some matchups are unwinnable. The
   `matchup_baseline` captures some of this variance but not enough.

3. **Reward signal is sparse.** Most steps have zero reward. Faint events
   (+/-0.5) happen ~6 times per 30-turn game. The per-step signal is ~0.01
   relative to the episode return range of [-4, +4].

**Evidence:**
- Run 043: ExplVar = 0.26 (best reading), win rate = 0.49
- Win rate oscillates ±10% between evals across ALL runs
- No sustained improvement trend in ANY run

---

## What the Previous Plan's Actor Freeze Would Actually Do

**It would address Problem 2** (prevent BC erosion while value learns).

**It would NOT address Problem 1** (value network still has 432K random params).
During the freeze period, the value function would try to learn from 1559-dim
observations with random weights. Based on run 048 data, it would likely need
80K+ steps to even become occasionally positive. The previous plan allocated
100K steps, which might be barely enough — or might not.

**It would NOT address Problem 3** (even with positive ExplVar, PPO can't
improve win rate). Run 043 proves this: positive ExplVar ≠ improving policy.

---

---

## CRITICAL FINDING: Warmstart Investigation (post-initial-analysis)

### The bc_warmstart.zip Already Contains Critic Warmstart

The file `models/bc_warmstart.zip` was saved by `warmstart_critic.py` which
writes to the SAME path as `behavioral_cloning.py`. The model includes both:
- BC-trained actor (policy_net + action_net)
- Warmstarted critic (value_net MLP + value head)

### Warmstart Quality: ExplVar = 0.9919

Evaluated the BC model's value function on the critic training data (29,498
samples):

```
Value predictions: min=-3.594, max=3.930, mean=0.577, std=1.678
Actual returns:    min=-3.593, max=3.824, mean=0.607, std=1.692
ExplVar: 0.9919
```

The critic warmstart achieved **99.2% explained variance** on its training data.
The value function is NOT randomly initialized — it's well-trained.

### PPO Destroys It in 23K Steps

Run 047 used this exact warmstarted model. After 23K PPO steps:

```
Value predictions: min=-4.097, max=3.327, mean=0.433, std=1.260
ExplVar: -0.2612
```

Weight changes were tiny (mean abs diff from BC model: 0.018-0.025 per layer).
Small perturbations to a 432K-param network = catastrophic value degradation.

### This Changes Everything About the Root Cause

Previous analysis of "three problems" was partially wrong:

**Problem 1 (value init) was ALREADY SOLVED.** The warmstart works. ExplVar 0.99.
We don't need weight copying or a bigger/smaller value network. The critic
warmstart in `warmstart_critic.py` does the job.

**Problem 2 (PPO destroys the warmstart) is THE primary problem.** This is not
about "BC erosion from noisy advantages." It's more specific: PPO's online value
function updates (n_epochs=10, LR=3e-4, 8192 samples per rollout) overfit each
rollout and destroy the warmstarted value function's generalization.

**Problem 3 (signal-to-noise) may not exist.** Run 043's "positive ExplVar but
no improvement" might be because ExplVar 0.1-0.26 is just not enough. With a
properly preserved warmstart (ExplVar 0.99→~0.5 range), PPO might actually
improve.

### Verified Mechanism: How PPO Destroys the Warmstart

1. Rollout collects 2048×4 = 8192 steps
2. GAE computes advantage targets (using the good value function)
3. PPO trains for 10 epochs on these 8192 samples (640 gradient steps at
   batch_size=128)
4. 432K-param value network memorizes this small dataset
5. Next rollout: value function performs poorly on slightly different states
6. GAE targets are now computed from a degraded value function
7. Each rollout degrades the value function further
8. After ~3-4 rollouts: ExplVar collapses to negative

### The Fix is About PRESERVATION, Not Initialization

The previous plan focused on initializing the value function. We now know it's
already well-initialized. The fix must PREVENT PPO from destroying it.

Three mechanisms:
1. **Actor freeze** — prevents state distribution shift between rollouts
2. **Fewer value epochs** (n_epochs=3 not 10) — prevents per-rollout overfitting
3. **Lower LR** (1e-4 not 3e-4) — smaller weight changes per update

---

## Recommended Fix: Attack All Three Problems

### Fix 1: Initialize value_net from policy_net weights

After BC training, copy `mlp_extractor.policy_net` weights to
`mlp_extractor.value_net`. The architectures are identical
([256,128] → [256,128]). This gives the value function:
- Pre-trained first layer that already knows how to process 1559-dim obs
- Feature representations learned for action prediction (useful for value too)
- Only the value head (Linear(128, 1)) needs to learn from scratch

This converts the value learning problem from:
  "Train 432K params from random on noisy targets" (hard)
to:
  "Fine-tune value head (129 params) on pre-trained features" (easy)

### Fix 2: Actor freeze (from previous plan, still valid)

Freeze actor for N steps so the value head can learn on a stable policy.
With pre-initialized features (Fix 1), this should converge much faster
than the previous plan expected.

### Fix 3: Reduce advantage noise

Options (try in order of simplicity):
a) `gae_lambda=1.0` — pure MC returns, no value bootstrap. Eliminates the
   circular dependency entirely. Higher variance but unbiased targets.
b) Lower clip_range (0.1 → 0.05) — slower policy change = less noise
c) Separate LR for actor (lower) and critic (higher)

---

## Quick Validation Experiment

**Train PPO from scratch (no BC) for 50K steps.** Check ExplVar.

If ExplVar goes positive → value function CAN learn from 1559 dims. Problem is
specifically about BC→PPO handoff.

If ExplVar stays negative → value function can't learn from 1559 dims even with
a random policy. The obs space or reward structure is the issue.

This takes ~30 min and is the single most informative experiment we can run.

---

## Summary Table

| Factor | Run 043 | Runs 046/048 |
|--------|---------|-------------|
| Value net params | 104K | 432K |
| Starting LR | 1e-4 | 3e-4 |
| Batch size | 64 | 128 |
| ExplVar | +0.01 to +0.26 | -0.29 to +0.12 |
| Win rate improvement | None (stuck ~50%) | None (stuck ~50%) |
| BC erosion rate | ~0.04%/K steps | ~0.03%/K steps |
| BC teacher | MaxDamage (simple) | SmartHeuristic (complex) |
| State diversity | Low | High |

---

## 2026-04-04 — Run 052 Status (Phase 0 Analysis)

**Run 052** (started 2026-04-04 09:30) applied the n_epochs=3 fix with 50K critic
warmup and LR=1e-4 constant. This is the first [256,128] run to maintain positive
ExplVar AND stable BC knowledge simultaneously. However, after 230K post-unfreeze
steps (287K total), training has plateaued. Win rate oscillates 44–67% per eval
(mean ~54%) with zero upward trend. BestMv% eroded only 0.7pp (58.3→57.6%) over
228K steps — 10x slower than old runs — confirming BC preservation works. ExplVar
averages ~0.12 (range +0.03 to +0.26), positive but low. The agent has near-zero
voluntary switching (0.6–0.7%). Temperature is stuck at ~0.95–1.05 (equilibrium:
at 54% WR the annealing formula targets temp=0.99). Per-opponent: SelfPlay ~49%,
MaxDmg ~54%, TypeMatch ~44% (declining from 62%), SoftmaxDmg ~67%.

**Bottom line:** The value function fix (n_epochs=3, critic warmup) solved the
ExplVar collapse and BC erosion problems. But PPO still can't improve. The agent
is a slightly fuzzy MaxDamagePlayer with no switching skill, hitting a hard
ceiling against diverse opponents.

---

## 2026-04-04 — Phase 1: Bottleneck Diagnosis

### What is the agent failing to do?

1. **Win rate not improving:** 54% aggregate WR at step 287K, essentially identical
   to step 59K (52%) when the actor unfroze. 228K steps of PPO with no gain.
2. **No voluntary switching:** 0.6–0.7% vol switch rate. The agent never switches
   strategically. Against TypeMatchupPlayer (which switches aggressively), WR has
   DECLINED from 62% → 44% over 170K steps.
3. **Temperature stuck at equilibrium:** SoftmaxDmg temp oscillates 0.88–1.05
   since step 80K. At 54% WR, the formula gives target_temp=0.99. The system is
   self-stabilizing at the agent's current skill — opponents never get harder.

### Specific numbers

| Metric | Step 59K (post-unfreeze) | Step 287K (latest) | Change |
|--------|--------------------------|---------------------|--------|
| Win rate | 52% | 54% | +2pp (noise) |
| BestMv% | 58.3% | 57.6% | -0.7pp |
| ExplVar | +0.074 | +0.103 | +0.03 (noise) |
| Vol.Sw% | 0.3% | 0.7% | +0.4pp |
| Temp | 0.92 | 0.95 | +0.03 |
| TypeMatch WR | 42% | 44% | -18pp from peak (62%) |
| MaxDmg WR | 43% | 57% | oscillating |
| SoftmaxDmg WR | 67% | 67% | flat |
| SelfPlay WR | 58% | 49% | oscillating ~50% |

### What changed vs last working state?

No previous run achieved sustained improvement. Run 043 (the best historical run)
also plateaued at ~50% WR with positive ExplVar. Run 052 is marginally better
(54% vs 50%) but the pattern is identical: value function works, policy doesn't
improve.

### Top 3 Hypotheses

**H1: Agent is at the skill ceiling for a non-switching policy**

The BC warm-start taught the agent MaxDamagePlayer behavior. MaxDamagePlayer
itself wins ~50% against a mixed pool. The agent is at 54% — only slightly above
this baseline. Without switching, it cannot:
- Escape bad type matchups (TypeMatch WR declining)
- Preserve low-HP Pokemon for later
- Counter opponent switches

*Confirming evidence:* Vol.Sw% = 0.7%, TypeMatch WR declining from 62%→44%
*Disconfirming evidence:* MaxDmg WR = 50-57% suggests some improvement over pure MaxDamage
*Experiment:* Benchmark MaxDamagePlayer against the same mixed_league pool for
100 battles. If MaxDmg WR ≈ 50-54%, the agent has hit the non-switching ceiling.

**H2: ExplVar ~0.12 is too low for PPO to make progress**

ExplVar 0.12 means advantage estimates are ~88% noise. PPO's policy gradient
needs at least moderately accurate advantages to identify which actions are
better. With this noise level, the gradient signal from "switch was good here"
is overwhelmed by random variance.

*Confirming evidence:* ExplVar 0.12 across 228K steps; no WR improvement
*Disconfirming evidence:* Run 043 had ExplVar 0.26 and also no improvement
*Experiment:* Check if ExplVar is improving at all. If it's been flat at 0.12
for 200K+ steps, the value function itself has plateaued.

**H3: Temperature equilibrium trap prevents difficulty escalation**

The temp formula `2.0*(1-wr) + 0.1*wr` with rate limiting creates a stable
equilibrium. At 54% WR → target temp ≈ 1.0. SoftmaxDmg at temp 1.0 is
effectively near-MaxDamage. All 4 opponents are now near their hardest setting,
leaving no "easy" games for clean learning signal.

*Confirming evidence:* Temp stuck at 0.88-1.05 for 200K steps. SoftmaxDmg WR
dropped from 78% (temp 2.0) to 67% (temp 1.0).
*Disconfirming evidence:* MaxDmg and SelfPlay are fixed-difficulty. If the agent
can't improve against fixed opponents, temperature isn't the issue.
*Experiment:* Not needed as standalone — MaxDmg WR (fixed difficulty) is also
flat at ~54%, confirming the bottleneck is the agent, not opponent difficulty.

### Assessment

H1 is the most likely primary bottleneck. The agent is functionally a
MaxDamagePlayer with slightly degraded move selection. PPO cannot discover
switching from this foundation because:
- Switching gives 0.0 immediate reward; benefit is 3-10 turns delayed
- BC bias against switching (action_net.bias[:6] halved but still negative)
- ent_coef=0.01 is insufficient to force random switching exploration
- ExplVar=0.12 means even if the agent accidentally switched well once, the
  noisy advantage estimate wouldn't reinforce it

H2 and H3 are secondary contributors but not the root cause. Even with perfect
ExplVar and easy opponents, a policy that never switches will plateau.

---

## Phase 2: Benchmark Experiments (2026-04-04)

### Experiment: Benchmark agents vs league opponents (100 battles each)

| Agent | vs MaxDmg | vs TypeMatch | vs SoftmaxDmg(t=1.0) | Overall |
|-------|-----------|--------------|----------------------|---------|
| MaxDamagePlayer | 41.0% | 47.9% | 65.7% | **51.5%** |
| TypeMatchupPlayer | 51.0% | 46.4% | 55.6% | **51.0%** |
| **Run 052 best model** | **55.6%** | **54.5%** | **64.6%** | **58.2%** |

### Key Finding: H1 (Non-Switching Ceiling) REFUTED

TypeMatchupPlayer, which switches aggressively for type advantage, gets only
51.0% overall — WORSE than our non-switching model at 58.2%. Switching alone
does not help. In fact, aggressive switching HURTS against damage-focused
opponents (TypeMatch gets 55.6% vs SoftmaxDmg, our model gets 64.6%).

### Key Finding: The Model IS Learning

The run 052 model beats BOTH heuristic baselines by 6.7–7.2pp. PPO training
DID improve the policy beyond the BC warm-start. The model's edge is primarily
in move selection against damage-focused opponents (55.6% vs MaxDmg, vs MaxDmg's
own 41.0% mirror match).

### Key Finding: Training WR vs Benchmark WR Discrepancy

Training WR (~54%) is lower than benchmark WR (58.2%) because training includes
SelfPlay (~50%, mirror match by definition). The actual heuristic-pool WR is ~58%.

### Revised Assessment

**Primary bottleneck: ExplVar ≈ 0.12 limits fine-grained policy improvement.**

The model learned the "big" things (basic move selection) quickly after
unfreezing. But the remaining improvements (optimal move choice in edge cases,
strategic switching, status move usage) require precise advantage estimates that
ExplVar=0.12 cannot provide. The value function is positive but explains only
12% of return variance, leaving 88% as noise in the policy gradient.

**Secondary: Temperature equilibrium is a symptom, not a cause.** The agent's
54% training WR locks temperature at ~1.0. But SoftmaxDmg isn't the bottleneck
opponent — it's the easiest at 64.6% WR.

**Switching is NOT the priority.** TypeMatch proves switching without good
judgment is counterproductive. The agent needs better value estimates first,
then can learn WHEN to switch (not just that switching exists).

### Committed Approach

**Improve value function quality (target ExplVar > 0.3) to unlock the next
phase of policy learning.** The model has shown it CAN learn when advantages
are moderately accurate (early post-unfreeze gains). The fix is to give it
better advantages, not to force new behaviors directly.

---

## 2026-04-04 — Run 054 Post-Mortem: Config Tuning Failed

Run 054 (n_steps=4096, n_epochs=4, vf_coef=1.0) killed at 177K steps. Compared
with run 052 at same step count: ExplVar 0.078 vs 0.175 (WORSE), BestMv% 58.7%
vs 57.3% (better BC preservation), WR 52% vs 53% (same), Temp 0.97 vs 0.97
(same). The config changes improved BC stability but had zero effect on the
ExplVar ceiling. Confirms anti-pattern: config-only fixes don't solve structural
problems.

---

## 2026-04-04 — ExplVar Ceiling Experiment: Returns Are Unpredictable

### Experiment Design
Collected 9515 (observation, discounted_return) pairs from run 054 model
playing 300 games (100 each vs MaxDmg, TypeMatch, SoftmaxDmg). Trained
offline regressors with 5-fold cross-validation to measure the maximum
achievable ExplVar given the current observation space.

### Results

| Model | Params | CV R² |
|-------|--------|-------|
| Mean predictor | 0 | -0.06 |
| Ridge (alpha=1.0) | 1559 | -2.28 |
| Tiny MLP [16] | ~25K | -1.19 |
| Small MLP [32,16] | ~51K | -0.56 |
| Large MLP [256,128] | ~432K | -0.33 |
| 3 simple features (HP, alive counts) | 3 | -0.04 |

**EVERY model performs WORSE than predicting the mean.** Even a regularized
linear model with only 3 features (own HP, own alive, opp alive) cannot
predict returns from observations.

### Root Cause: Returns Are Structurally Unpredictable

Three factors combine to make returns unpredictable from per-step observations:

1. **gamma=0.99 spreads returns across the entire game.** At step 1, the return
   includes rewards from step 30. The observation at step 1 cannot predict what
   happens at step 30 (crits, misses, opponent's hidden team revealed).

2. **matchup_baseline removes the most predictable signal.** Team quality
   (which IS observable via types/stats) is explicitly subtracted from terminal
   reward. The remaining variance is pure game-play uncertainty.

3. **Partial observability.** Opponent's unrevealed Pokemon (1-5 hidden) have
   massive impact on returns but aren't in the observation.

### Implication: PPO's ExplVar=0.12 Is Likely Overfitting

If the true ExplVar ceiling is ≤0 (offline CV proves this), then PPO's
reported ExplVar=0.12 is in-sample memorization of the current rollout, not
real predictive ability. Advantages computed from this V(s) are essentially
random noise with a bias.

### Why PPO Cannot Learn

1. V(s) predicts noise → advantages = random
2. Random advantages → random policy gradient direction
3. BC knowledge erodes from random updates
4. No improvement possible regardless of training duration/config

This is NOT a training problem. It's a problem formulation problem.

### Proposed Fix: Lower Gamma

With gamma=0.99, effective horizon = 100 steps (entire game).
With gamma=0.95, effective horizon = 20 steps (next few events).
With gamma=0.90, effective horizon = 10 steps.

Shorter gamma makes returns depend on NEAR-FUTURE events (next faint, next
few turns), which ARE observable from current state. The value function can
predict "opponent active is at 10% HP → faint incoming → positive near-term
return." It CANNOT predict "will we win in 30 turns against hidden Tauros."

Trade-off: agent becomes more myopic. Won't sacrifice now for benefit 20+
turns later. But currently it can't learn ANYTHING. Better to learn short-term
tactics well than to attempt long-horizon planning with zero learning signal.
