# Sprint 2: Behavioral Cloning + Architecture Improvements

Based on Expert Panel #3 consensus. Execute after Sprint 1 changes are validated.

---

## Step 1: Behavioral Cloning Warm-Start

### 1a: Generate training data

Create `scripts/generate_bc_data.py`:

```python
"""Generate behavioral cloning data from MaxDamagePlayer vs RandomPlayer."""
```

- Use poke-env to run MaxDamagePlayer vs RandomPlayer for 10,000 games
- For each turn where MaxDamagePlayer acts, record:
  - `obs`: the 478-dim observation (call `embed_battle(battle)` from the player's perspective)
  - `action`: the action MaxDamagePlayer would choose (integer 0-9)
  - `mask`: the valid action mask (boolean array of length 10)
- Save as `data/bc_training_data.npz` with keys `observations`, `actions`, `masks`
- Print stats: number of transitions, action distribution, avg game length

### 1b: Train the behavioral cloning model

Create `scripts/behavioral_cloning.py`:

```python
"""Pre-train a MaskablePPO policy via imitation learning on MaxDamagePlayer."""
```

- Load data from `data/bc_training_data.npz`
- Create a MaskablePPO model with the same architecture as training (policy_kwargs matching train.py)
- Extract the policy network and train supervised:
  - Loss: masked cross-entropy (only over legal actions)
  - Optimizer: Adam, lr=1e-3
  - Epochs: 50-100 (until loss plateaus)
  - Batch size: 256
  - Train/val split: 90/10
- Save the pre-trained model as `models/bc_warmstart.zip`
- Print final accuracy on validation set

### 1c: Integrate into training loop

In `src/train.py`:
- Before the curriculum loop, check if `models/bc_warmstart.zip` exists
- If it exists and no resume_path, load it as the starting model instead of creating from scratch
- Skip the logit bias initialization (the BC model already has learned biases)
- Print "Starting from behavioral cloning warm-start (X% accuracy on MaxDamage imitation)"

---

## Step 2: Per-move feedback reward

In `src/env/gen1_env.py`, `calc_reward()`:

Research how poke-env exposes the last move used. Candidates:
- `battle.active_pokemon.last_move` (if it exists)
- Checking move order history in the battle object
- Comparing battle state between steps

If accessible, add:
```python
# Per-move feedback: small reward for picking type-effective moves
if not (battle.won or battle.lost) and last_move and last_move.base_power > 0:
    effectiveness = _compute_effectiveness(last_move, opponent)
    if effectiveness > 1.0:   # super effective
        reward += 0.05 * self.shaping_factor
    elif effectiveness < 1.0: # not very effective or immune
        reward -= 0.05 * self.shaping_factor
```

Keep the feedback small (0.05) relative to other shaping (0.5 for faints) to avoid overwhelming the main signal.

---

## Step 3: Dynamic entropy based on win rate

In `src/callbacks.py`, add a method `_maybe_adjust_entropy()`:

```python
def _maybe_adjust_entropy(self, win_rate: float) -> None:
    """Increase entropy when stuck, decrease when winning."""
    if win_rate < 0.10 and self.model.ent_coef < 0.1:
        self.model.ent_coef = min(self.model.ent_coef * 1.5, 0.1)
        print(f"  [Entropy] Stuck at {win_rate:.0%} — increased ent_coef to {self.model.ent_coef:.3f}")
    elif win_rate > 0.50 and self.model.ent_coef > 0.01:
        self.model.ent_coef = max(self.model.ent_coef * 0.9, 0.01)
        print(f"  [Entropy] Winning at {win_rate:.0%} — decreased ent_coef to {self.model.ent_coef:.3f}")
```

Call this from `_evaluate()` after computing win_rate. This is an adaptive exploration mechanism that prevents the death spiral.

---

## Step 4: Add RandomDamage opponent

In `src/agents/heuristic_agent.py`, create `RandomDamagePlayer`:

```python
class RandomDamagePlayer(Player):
    """Picks a random MOVE every turn (never switches voluntarily).

    Sits between RandomPlayer (60% switch) and SoftmaxDamagePlayer (smart damage).
    Tests if the agent can learn 'pick the better move' when the opponent
    isn't also picking smart moves.
    """
    def choose_move(self, battle):
        if battle.available_moves:
            return self.create_order(random.choice(battle.available_moves))
        return self.choose_random_move(battle)
```

Add to `make_env()` as `opponent_type="random_damage"`.

---

## Step 2: Per-move feedback reward — CANCELLED

Originally planned: +0.05/-0.05 reward for super-effective/not-very-effective moves.

**Cancelled because:** After simplifying the reward to faint differential + win/loss only (removing HP and status shaping), adding per-move effectiveness shaping would reintroduce the same per-step noise that was degrading the BC policy. The RL expert diagnosed that noisy advantages (from noisy rewards → bad value function → random gradients) were causing BestMv% to decay from 82% to 64%. Adding more per-step shaping goes against that fix. The BC policy already picks effective moves ~80% of the time — we don't need a reward to teach it that.

---

## Verification

1. BC model achieves >80% action accuracy on MaxDamage imitation — **DONE: 99.0%**
2. Training with BC warm-start begins at ~70% win rate vs SoftmaxDmg(temp=100)
3. Dynamic entropy kicks in when win rate drops, recovers within 50K steps
4. All tests pass with new features — **DONE: 23 passed**
