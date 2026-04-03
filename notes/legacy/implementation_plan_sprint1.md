# Implementation Plan: Fix Agent Learning

Based on Expert Panel #3 consensus. Execute in order.

---

## Step 1: Fix Phase B target (trivial)

**File:** `src/train.py`
**Change:** In the CURRICULUM list, change Phase B `target_wr` from `0.95` to `0.75`.

---

## Step 2: Add observation features (KO estimation + speed comparison)

**File:** `src/env/gen1_env.py`

### 2a: Add speed comparison feature
In `embed_battle()`, after the trapping section (line ~237), add one new feature:
```python
# Speed comparison: who attacks first? (most important binary in RBY)
own_speed = own.base_stats.get("spe", 0)
opp_speed = opp.base_stats.get("spe", 0)
if own.status == Status.PAR:
    own_speed *= 0.25
if opp.status == Status.PAR:
    opp_speed *= 0.25
speed_total = own_speed + opp_speed
obs.append(own_speed / speed_total if speed_total > 0 else 0.5)
```

### 2b: Add "can I KO?" estimation per move
In `_move_features()`, add a 6th feature after accuracy:
```python
# KO estimation: estimated_damage / opp_remaining_hp
# Uses base_power * effectiveness as damage proxy, normalized by opponent HP
opp_hp = opponent.current_hp_fraction if not opponent.fainted else 0.0
if opp_hp > 0 and base_power > 0:
    damage_proxy = (base_power * effectiveness * 4.0) / 150.0  # rough normalized damage
    ko_ratio = min(damage_proxy / opp_hp, 2.0) / 2.0  # cap at 1.0
else:
    ko_ratio = 0.0
```
This adds 1 feature per move = +4 for own active moves, +4 per bench mon moves, +4 for opp revealed moves.

### 2c: Update observation dimensions
Total new dims: 1 (speed) + 4 (own moves KO) + 24 (own bench 6x4) + 4 (opp revealed moves KO) + 24 (opp bench 6x4) = 57 new dims.
New total: 421 + 57 = 478.

Update `describe_embedding()` shape to `(478,)`.
Update all shape assertions and docstrings.
Update CLAUDE.md observation dimension references.

### 2d: Update obs_transfer.py
The `is_compatible()` function checks obs dimensions. Existing checkpoints (421 dims) will need transfer via `load_with_expanded_obs()`. Verify this still works.

---

## Step 3: Add per-move feedback reward

**File:** `src/env/gen1_env.py`

In `calc_reward()`, add a small shaping reward based on move quality:
```python
# Per-move feedback: reward for picking good moves
# Only when we actually attacked (not forced switch, not start of battle)
if not (battle.won or battle.lost):
    last_move = battle.active_pokemon.last_move  # or however poke-env exposes this
    if last_move and last_move.base_power > 0:
        effectiveness = last_move.type.damage_multiplier(
            battle.opponent_active_pokemon.type_1,
            battle.opponent_active_pokemon.type_2,
            type_chart=_GEN1_TYPE_CHART
        ) / 4.0
        # +0.05 for super effective, -0.05 for not very effective, 0 for neutral
        move_feedback = (effectiveness - 0.25) * 0.2 * self.shaping_factor
        reward += move_feedback
```

**Important:** Check how poke-env exposes the last move used. Look at `battle.active_pokemon` attributes. May need to use a different API. Research first.

---

## Step 4: Separate policy and value networks

**File:** `src/train.py` and `src/selfplay_train.py`

In PPO_KWARGS, add:
```python
policy_kwargs=dict(
    net_arch=dict(pi=[64, 64], vf=[128, 128])
),
```

This gives the value function double the capacity of the policy network. Both use separate feature extractors (no weight sharing).

---

## Step 5: Lower gamma for early phases

**File:** `src/train.py`

Add `gamma` to each phase in the CURRICULUM:
```python
dict(name="A", ..., gamma=0.95),  # short episodes vs random
dict(name="B", ..., gamma=0.95),  # still relatively short
dict(name="C", ..., gamma=0.97),  # longer games as agent improves
dict(name="D", ..., gamma=0.99),  # full horizon for strategic play
```

Then when creating the model or calling `model.learn()`, set `model.gamma = phase["gamma"]`.

Note: Changing gamma mid-training requires updating the model's gamma attribute. Check if SB3 allows this or if it needs a new model instance.

---

## Step 6: Behavioral cloning warm-start

### 6a: Generate training data
Create `scripts/generate_bc_data.py`:
- Run MaxDamagePlayer vs RandomPlayer for 10,000 games
- For each turn, record: (observation_421_dims, action_taken_by_maxdamage, action_mask)
- Save as a numpy archive (.npz)

### 6b: Implement behavioral cloning
Create `scripts/behavioral_cloning.py`:
- Load the generated data
- Create a MaskablePPO model with the same architecture
- Extract the policy network
- Train supervised (cross-entropy loss) on the (obs, action) pairs
- Only train on actions where the mask was valid
- Save the pre-trained model as `models/bc_warmstart.zip`

### 6c: Integrate into training
In `src/train.py`, before the curriculum loop:
- If `models/bc_warmstart.zip` exists and no resume_path, load it as the starting model
- Print "Starting from behavioral cloning warm-start"

---

## Step 7: Single continuous curriculum (eliminate phase transitions)

**File:** `src/train.py`

Replace the 4-phase CURRICULUM with a single training loop:
- One SoftmaxDamagePlayer opponent starting at temperature=100.0
- Temperature anneals continuously based on win rate (already have this logic in callbacks)
- No phase transitions, no environment rebuilding, no distribution shift
- One long training run of 2M+ steps

The annealing logic already exists in `_maybe_anneal_epsilon()` for temperature-based opponents. Just need to restructure `train.py` to use a single phase.

Keep the mixed opponent pool for later (Phase D equivalent), but only after the agent can beat SoftmaxDmg at temp 0.1.

---

## Step 8: Wider first hidden layer

**File:** `src/train.py` and `src/selfplay_train.py`

Combined with Step 4, the full policy_kwargs becomes:
```python
policy_kwargs=dict(
    net_arch=dict(pi=[256, 64], vf=[256, 128])
),
```

This gives the first layer 256 neurons (4x current) to handle the 478-dim observation, then narrows to 64/128 for the second layer.

---

## Verification after implementation

1. Run `pytest tests/` — all tests must pass (some will need obs dim updates)
2. Run a short training (10K steps) and verify:
   - New observation features are populated correctly
   - Explained variance improves over the first 5K steps
   - Win rate vs Random starts at ~90% (logit bias + BC warmstart)
   - No crashes or NaN rewards
3. Check that obs_transfer handles 421->478 expansion correctly
4. Update test_core.py with new obs dimensions and features
