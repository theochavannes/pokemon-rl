# run_057 Post-Mortem

**Killed:** 2026-04-19, step 382K
**Obs dim:** 1739 (OU pool + role features + KO features)
**Warm-started from:** run_056 best checkpoint (57% WR, step ~89K)

---

## Trajectory

| Steps | WR range | Notable |
|-------|----------|---------|
| 0–50K | ~47% noise | Critic warmup, actor frozen |
| 50–176K | 47%→59% | Actor unfreezes, new OU-pool best |
| 176–314K | 40–52% | Stagnation, temp oscillating |
| 314–355K | 39–65% | Spike to 65%, two milestones fired |
| 355–382K | 43–50% | Collapsed back, BestMv% frozen at 45.0% |

Peak: **65% WR at step 355K** (new OU-pool record, up from run_056's 58%)

---

## Key Metrics at Kill

- WR: ~47% average (65% peak)
- BestMv%: 45.0% (frozen for last ~60K steps)
- vol_switch: 1.2% (from 1.0% at start — no real movement)
- dmg_eff: 0.59 (up from 0.56 in run_056)
- ExplVar: ~0.25
- Temperature: oscillating 1.05–1.18, not converging
- SelfPlay pool: 7 snapshots, Elo ~1268

---

## What KO Features Contributed

**Helped:**
- BestMv%: +1–2pp over run_056 (44% → 45–46%)
- dmg_eff: +0.03 (0.56 → 0.59)
- Peak WR: +7pp over run_056 (58% → 65%)
- `prob_ko` appears to improve within-turn move selection

**Did not help:**
- vol_switch: 1.0% → 1.2%, essentially no change
- `recharge_trap` did not change switch behavior
- Plateau pattern identical to run_056

---

## Hypotheses for Plateau (priority order)

1. **n_steps 2048 → 4096** ← strongest signal
   - run_054 (n_steps=4096): BestMv% 58–59%, peak WR 69% vs all-species
   - run_057 (n_steps=2048): BestMv% 45%, peak WR 65% vs OU-only
   - Larger rollouts → stabler gradients → better credit assignment for switch rewards
   - Cheapest to test: single hyperparam change, no code change

2. **OU pool difficulty** (may not be fixable — by design)
   - No weak filler mons means smaller margin between best/worst moves
   - Lower practical WR ceiling for any given policy quality

3. **Network capacity** (not tested)
   - 256×128 MLP over 1739 features
   - Feature-to-parameter ratio has grown 4× since obs=421
   - Test: try 512×256×128

4. **Role features noise** (not ablated)
   - BestMv% dropped from 58% → 44% when role features were added (obs 1559→1727)
   - Could be pool change confounding, but worth isolating

---

## Recommended Next Run

**run_058 — n_steps=4096 ablation**

Change only:
```python
n_steps = 4096  # was 2048
```

Everything else identical to run_057 (obs 1739, OU pool, same net_arch, same LR).
Do NOT warm-start — train from scratch to avoid locking into run_057's attractor.

If BestMv% recovers to ~55%+ and/or WR ceiling rises above 65%, n_steps was the bottleneck.
If metrics are the same, move to hypothesis 3 (network capacity).

---

## Comparison Table

| Run | Obs | Pool | n_steps | Peak WR | BestMv% | vol_switch |
|-----|-----|------|---------|---------|---------|-----------|
| run_054 | 1559 | all-species | 4096 | 69% | 58–59% | 0.6% |
| run_055 | 1559 | all-species | 2048 | 65% | ~58% | 0.6% |
| run_056 | 1727 | OU-only | 2048 | 58% | 44–45% | 0.8% |
| run_057 | 1739 | OU-only | 2048 | 65% | 45–46% | 1.2% |
| run_058 (proposed) | 1739 | OU-only | 4096 | ? | ? | ? |

Note: run_054/055 vs run_056/057 are not directly comparable (different pool).
