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
