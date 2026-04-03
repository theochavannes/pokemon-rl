# Sprint 3A: Mixed Opponent Pool + Anti-Forgetting

Based on Expert Panel #4 consensus (unanimous #1 pick). Implement now.

---

## Step 1: Mixed opponent pool in train.py

Replace the single-opponent curriculum with a mixed pool using the 4 existing envs:

```python
CURRICULUM = [
    dict(
        name="A",
        opponent_type="mixed_league",
        phase_label="League",
        target_wr=0.70,
        max_steps=2_000_000,
        shaping_factor=1.0,
    ),
]
```

In `make_env()`, add a new `opponent_type="mixed_league"`:
- Env 0: Self-play (FrozenPolicyPlayer from BC warm-start, swapped every 50K steps)
- Env 1: MaxDamagePlayer
- Env 2: TypeMatchupPlayer
- Env 3: SoftmaxDamagePlayer (temperature starts at 2.0, anneals based on win rate)

No phase transitions. The agent sees all opponent types every rollout.

**Important:** The self-play opponent starts from BC warm-start (not random), so it's immediately a competent opponent.

---

## Step 2: KL penalty against BC policy

Prevent PPO from eroding the BC knowledge.

In `train.py`, after loading the BC warm-start model:
1. Save a frozen copy of the BC policy weights
2. During PPO's loss computation, add: `loss += beta * KL(pi_current || pi_BC)`

**Implementation approach:** SB3 doesn't natively support KL penalties. Two options:

**Option A (simple):** Use SB3's `clip_range` more aggressively (0.1 instead of 0.2). This indirectly limits how far the policy can drift per update. Not a true KL penalty but achieves a similar effect.

**Option B (proper):** Create a custom callback that, after each PPO update, checks if the policy has drifted too far from BC. If the KL divergence exceeds a threshold, partially revert the policy toward BC weights:
```python
for param, bc_param in zip(model.policy.parameters(), bc_params):
    param.data = (1 - revert_rate) * param.data + revert_rate * bc_param.data
```
Use `revert_rate = 0.1` and `kl_threshold = 0.1`.

Start with Option A (just lower clip_range to 0.1). Implement Option B only if the policy still degrades.

---

## Step 3: BC regression test

Every 50K steps, automatically play 50 games against the frozen BC model.

In `WinRateCallback._evaluate()`:
- If `self.num_timesteps % 50_000 < eval_freq` and we have a frozen BC path:
  - Load BC as a FrozenPolicyPlayer
  - Run 50 games (can reuse battle_sim logic)
  - Log result: "BC regression: X% vs BC baseline"
  - If win rate < 40%: print warning "REGRESSION: agent losing to its own starting point"

**Simpler alternative for now:** Skip the automated test. Instead, after each training run, manually run:
```bash
python scripts/battle_sim.py best_model models/bc_warmstart 100
```

Start with the manual approach. Automate later if needed.

---

## Step 4: Self-play opponent swap

The self-play opponent (env 0) should update periodically:
- Use the existing `OpponentSwapCallback` logic
- Swap every 50K steps (or when win rate against self > 70%)
- Save snapshots to `models/league/snapshot_NNNN.zip` for future fictitious self-play

---

## Verification

1. Agent trains against 4 diverse opponents simultaneously (check logs show battles against all types)
2. BestMv% stays above 70% (BC knowledge preserved) while win rate improves
3. Vol.Switch% starts increasing (TypeMatchup/AggressiveSwitcher force switching)
4. Agent beats all heuristics at >50% within 500K steps
5. No regression below BC baseline
