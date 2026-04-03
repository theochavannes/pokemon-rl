# Sprint 3: 1v1 Training Mode + Structural Changes

Based on Expert Panel #3 consensus. Execute after Sprint 2 if agent still struggles with full 6v6.

---

## Step 1: 1v1 Training Mode

The panel's most consistently recommended structural change. Eliminates switching entirely so the agent can focus on learning which move to pick.

### 1a: Create a 1v1 battle format in Showdown

Option A — Use an existing format:
- Check if Showdown supports `gen1randombattle` with team size 1
- May need to create a custom format in `showdown/config/formats.ts`

Option B — Wrapper approach:
- Keep gen1randombattle but create a wrapper that ends the battle after the first Pokemon on either side faints
- In Gen1Env, override the termination condition: `done = True` when any Pokemon faints
- Mask all switch actions (only moves 6-9 are valid)

Option B is simpler and doesn't require Showdown modifications.

### 1b: Simplified observation for 1v1

Create `embed_battle_1v1()` — a reduced version of `embed_battle()`:
- Own active: 16 dims (same)
- Own moves: 24 dims (4 moves x 6 features including ko_ratio)
- Opp active: 15 dims (same)
- Opp revealed moves: 24 dims (same)
- Speed comparison: 1 dim (same)
- **Total: 80 dims** (vs 478 for full)

No bench Pokemon at all. The network only needs to learn move selection.

### 1c: Training curriculum for 1v1

```python
# 1v1 training: master move selection before adding switching
CURRICULUM_1V1 = [
    dict(
        name="1v1",
        opponent_type="softmax_damage",
        phase_label="1v1-SoftmaxDmg",
        target_wr=0.70,
        max_steps=500_000,
        shaping_factor=1.0,
        epsilon_start=100.0,
        epsilon_end=0.1,
    ),
]
```

Use a smaller network for 1v1: `net_arch=dict(pi=[64, 32], vf=[64, 32])` — 80 dims is small enough.

### 1d: Transfer from 1v1 to 6v6

After the agent masters 1v1 move selection:
1. Save the 1v1 model
2. Create a new 6v6 model with the full 478-dim observation
3. Transfer the move-selection weights from the 1v1 model into the corresponding positions of the 6v6 model's first layer
4. This is similar to the existing `obs_transfer.py` but mapping from 80 dims to 478 dims

The agent starts 6v6 already knowing "which move to pick" and only needs to learn "when to switch."

---

## Step 2: Fixed teams for intermediate training

Between 1v1 and full random battles, train on a fixed set of teams.

### 2a: Create fixed team configs

Create `data/fixed_teams.json` with 5-10 balanced Gen 1 teams:
```json
[
    {
        "name": "OU_Standard",
        "pokemon": ["Tauros", "Starmie", "Snorlax", "Exeggutor", "Chansey", "Rhydon"]
    },
    ...
]
```

### 2b: Custom Showdown format

May need to create a `gen1fixedteam` format or use `gen1ou` with pre-built teams. Research Showdown's team validation and format system.

### 2c: Training with fixed teams

Reduces combinatorial explosion — the agent sees the same 10 matchups repeatedly. Type effectiveness patterns become learnable. Once the agent masters fixed teams, graduate to random battles.

---

## Step 3: Progressive network expansion

After 1v1 mastery, when transitioning to 6v6:

1. Train 1v1 with [64, 32] network on 80-dim obs → save model
2. Create [256, 64] network on 478-dim obs
3. Map the 1v1 weights into the corresponding input positions
4. Train 6v6 starting from these transferred weights

This is the "net2net" approach — preserves learned features while expanding capacity.

---

## Verification

1. 1v1 agent achieves >80% vs SoftmaxDmg(temp=0.1) within 500K steps
2. 1v1 agent's move selection accuracy matches or exceeds MaxDamagePlayer
3. Transfer to 6v6 preserves move selection skill (WR doesn't collapse)
4. Fixed-team training shows clear learning curve (WR improves steadily)
