# Sprint 7: Training Refinement + Data Quality

Priority: LOWER — Ranked items #12-15, #18-19 from panel #5.
Optimization and polish after architecture is settled.

---

## Step 1: Larger BC dataset (10K-20K games)

928-dim obs + status move diversity needs more training data. Current 5000 games may underrepresent rare status situations.

```bash
python scripts/generate_bc_data.py --teacher smart --opponent mixed --n-battles 20000
```

Also consider generating data from SmartHeuristic vs SmartHeuristic (mirror matches) for more complex switching patterns.

---

## Step 2: Turn counter / game phase feature (+1 dim)

Normalized turn number (turn / 50). Helps the agent distinguish early game (preserve mons, set up) from endgame (trade aggressively, sacrifice for KOs).

In `embed_battle()`:
```python
turn_phase = min(battle.turn / 50.0, 1.0)
obs.append(turn_phase)
```

---

## Step 3: Hyper Beam recharge flag (+1 per move)

"This move requires skipping next turn." Critical for Gen 1 — Hyper Beam is 150bp but costs two turns unless it KOs.

```python
# In _move_features():
must_recharge = 1.0 if "recharge" in move.flags else 0.0
```

Check poke-env: `move.entry.get("self", {}).get("volatileStatus") == "mustrecharge"` or check flags.

---

## Step 4: Dream Eater conditional flag (+1 per move)

Dream Eater (100bp + 50% drain) only works on sleeping targets. Without this flag, the agent might waste turns attempting it against awake opponents.

```python
requires_sleep = 1.0 if move.id == "dreameater" else 0.0
```

---

## Step 5: PP count as absolute, not just fraction

PP fraction (15/15 = 1.0 and 35/35 = 1.0) hides absolute PP. A move with 5 PP total (Blizzard) is more precious than one with 35 PP (Tackle). Add `pp_absolute = current_pp / 35.0`.

Alternatively, add `max_pp / 35.0` as a feature to indicate move scarcity.

---

## Step 6: Move-specific flags

| Flag | Moves | Encoding |
|------|-------|----------|
| is_contact | Body Slam, Submission | 0/1 |
| is_sound_based | Screech, Growl | 0/1 |
| ignore_accuracy | Swift | 0/1 |

These are minor but help the NN understand move mechanics. Check `move.flags` dict in poke-env.

---

## Step 7: Reward experimentation (controlled)

If the agent still struggles to learn switching after architecture changes:
- **hp_value=0.25**: Small immediate reward for HP preservation makes switching visible without dominating the reward signal
- Test with a controlled A/B: same obs space, same NN, different hp_value (0.0 vs 0.25)

---

## Step 8: Self-play population

Instead of one frozen opponent, maintain a pool of 5-10 past checkpoints. Each self-play game randomly picks an opponent from the pool. Prevents overfitting to one opponent's weaknesses. This is the "league training" concept from AlphaStar.

---

## Estimated effort

- Steps 1: Just running longer data generation (~20 min)
- Steps 2-6: Small obs space changes, ~1-2 hours total
- Step 7: Config change + A/B test infrastructure
- Step 8: Significant engineering (~3-4 hours)

---

## Verification

1. Larger dataset: verify action distribution still has >10% switches and >5% status moves
2. Turn counter: verify values 0.0-1.0 across different game lengths
3. A/B test: compare Elo curves with and without hp_value over 100K steps
