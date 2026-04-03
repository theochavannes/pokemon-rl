# Two-Tower Architecture Migration Plan

## Status: DEFERRED — Waiting for flat MLP to reach 60% WR

The PokemonFeatureExtractor (two-tower) is theoretically sound but was introduced
too early. This document captures the full plan for when and how to introduce it.

---

## What Happened (run_045 Post-Mortem)

### The Failure
- BC accuracy dropped from 99% (flat MLP) to 46.6% (two-tower)
- BC loss plateaued at epoch 1 — the model couldn't learn at all
- run_045 peaked at 39% WR, collapsed to 16% over 600K steps
- run_043 (flat MLP, pre-sprint) hit 62% WR

### Root Cause
The two-tower feature extractor has ~450K parameters that must be jointly optimized
with the ~40K policy head parameters. BC's simple cross-entropy + Adam optimizer
couldn't solve this coupled optimization — the feature extractor produced random
features, so the policy head couldn't learn meaningful actions.

A flat MLP has a much simpler optimization landscape: each layer directly feeds the
next. The feature extractor adds a branching structure (own tower + opp tower + merge)
that requires coordinated gradient flow across three paths simultaneously.

### Why It Works in AlphaStar / OpenAI Five
These systems use:
1. Massive supervised pre-training datasets (millions of human games)
2. Custom training recipes (layer-wise warm-up, separate LR per component)
3. Distributed training across thousands of workers
4. The architecture is introduced after an initial flat training phase

We tried to go from 0 to two-tower in one step with 163K BC transitions.

---

## Migration Plan: 4 Phases

### Phase 1: Flat MLP Baseline (CURRENT)
**Goal:** Achieve 60%+ WR with flat MLP [256, 128] on 1559-dim obs.

- Architecture: `net_arch=dict(pi=[256, 128], vf=[256, 128])`
- BC warm-start: flat MLP, target >90% accuracy
- Training: mixed league, self-play pool
- **Exit criteria:** Stable 60% WR vs mixed league for 50K+ steps

### Phase 2: Weight Extraction
**Goal:** Extract trained flat MLP weights for two-tower initialization.

The flat MLP processes all 1559 dims through shared layers. We can extract
the learned weight matrix and split it:

```python
# Flat MLP first layer: W shape (1559, 256)
flat_weights = model.policy.mlp_extractor.shared_net[0].weight  # (256, 1559)

# Split by obs layout:
# Own features: dims 0:774
# Opp features: dims 774:1543
# Global: dims 1543:1559

own_weights = flat_weights[:, :774]      # (256, 774)
opp_weights = flat_weights[:, 774:1543]  # (256, 769)
global_weights = flat_weights[:, 1543:]  # (256, 16)
```

### Phase 3: Two-Tower Initialization
**Goal:** Create a two-tower model pre-initialized from flat MLP weights.

```python
# Initialize own tower first layer from flat MLP's own-feature weights
extractor.own_tower[0].weight.data = own_weights  # or padded version

# Initialize opp tower from flat MLP's opp-feature weights
extractor.opp_tower[0].weight.data = opp_weights

# Initialize merge layer to combine tower outputs + global features
# This requires more careful construction since the merge layer input
# is (128 + 128 + 16 = 272) not 256.
```

**Key challenge:** The flat MLP's first layer maps (1559 → 256) as one operation.
The two-tower splits this into three parallel paths that merge later. The weight
transfer isn't a perfect mapping — the flat MLP may have learned cross-feature
interactions (own-vs-opp) that don't decompose cleanly into separate towers.

**Mitigation:** After initialization, fine-tune the two-tower with a LOW learning rate
(1e-5) for a few thousand PPO steps. The towers should quickly specialize from their
pre-initialized starting point.

### Phase 4: Two-Tower PPO Training
**Goal:** Demonstrate two-tower matches or exceeds flat MLP performance.

- Learning rate: 1e-5 initially (preserve initialization), ramp to 1e-4
- clip_range: 0.1 (conservative, preserve transferred knowledge)
- Compare against flat MLP baseline over 500K steps
- Monitor: does the own-tower develop different representations than the opp-tower?

**Success criteria:** Two-tower WR >= flat MLP WR after 200K steps.

---

## Alternative Approaches (If Weight Transfer Fails)

### A: Layer-Wise BC Training
1. Freeze feature extractor, train only policy head on BC data
2. Unfreeze merge layer, continue BC training
3. Unfreeze towers, continue BC training
4. This gives each component a stable training signal before the next unfreezes

### B: Auxiliary Losses
Add auxiliary prediction tasks that force each tower to learn useful features:
- Own tower: predict own team's total HP, predict number alive
- Opp tower: predict opponent type matchup quality, predict revealed move count
- These supervised signals help the towers converge faster

### C: Knowledge Distillation
Train the two-tower to match the flat MLP's output distribution:
- Input: same observations
- Target: flat MLP's action probabilities (soft labels, temperature=2)
- Loss: KL divergence between two-tower and flat MLP outputs
- This is gentler than hard BC labels and preserves more knowledge

---

## Architecture Reference

Code: `src/env/feature_extractor.py`

```text
Observation (1559 dims)
    ├── Own tower (774 dims) → 256 → ReLU → 128 → ReLU ──┐
    ├── Opp tower (769 dims) → 256 → ReLU → 128 → ReLU ──┼── Merge (272) → 256 → ReLU
    └── Global (16 dims) ─────────────────────────────────┘         │
                                                              ┌─────┴─────┐
                                                           pi=[128]   vf=[128]
                                                              │          │
                                                         10 actions   1 value
```

Total parameters: ~598K (vs ~870K for flat MLP [256,128] with 1559 inputs)

---

## Timeline Estimate

- Phase 1 (flat MLP baseline): Current work, ~1-2 training runs
- Phase 2-3 (weight extraction + init): ~2 hours implementation
- Phase 4 (two-tower training): ~1 training run to validate
- Total: 2-3 training runs + 2 hours of code

**Do not start Phase 2 until Phase 1 achieves stable 60% WR.**
