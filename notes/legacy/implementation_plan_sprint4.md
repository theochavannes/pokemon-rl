# Sprint 4: Advanced Techniques (If Needed)

Based on Expert Panel #3 lower-ranked but valuable ideas. Execute only if Sprints 1-3 don't achieve satisfactory results.

---

## Step 1: Supervised pre-training on type effectiveness

### 1a: Generate type effectiveness dataset

Create a dataset of (move_type, opponent_type_1, opponent_type_2) → effectiveness_multiplier for all Gen 1 type combinations.

```python
# Generate all possible type matchups
for move_type in gen1_types:
    for def_type_1 in gen1_types:
        for def_type_2 in gen1_types + [None]:
            effectiveness = move_type.damage_multiplier(def_type_1, def_type_2, ...)
            dataset.append((move_type_idx, def1_idx, def2_idx, effectiveness))
```

### 1b: Pre-train first layer

Train the first layer of the policy network to predict effectiveness from type inputs. This teaches the network "Fire > Grass" before it ever plays a battle.

Use a simple auxiliary loss during the first N training steps:
```python
aux_loss = MSE(first_layer_output[effectiveness_neurons], true_effectiveness)
total_loss = ppo_loss + 0.1 * aux_loss  # small weight, don't dominate RL
```

---

## Step 2: Feature grouping / lightweight attention

**Note:** Dimensions below assume the current 421-dim obs. If obs features are expanded (e.g., KO ratio, speed comparison from the fix-agent-learning branch), adjust accordingly.

Replace the flat MLP with a structured architecture:

```
Own active (16) → FC(16, 32) → own_active_embed (32)
Own moves (24)  → FC(24, 32) → own_moves_embed (32)
Own bench (198) → FC(198, 64) → own_bench_embed (64)
Opp active (15) → FC(15, 32) → opp_active_embed (32)
Opp bench (198) → FC(198, 64) → opp_bench_embed (64)
Opp moves (24)  → FC(24, 32) → opp_moves_embed (32)
Speed+Trap (3)  → FC(3, 8)   → misc_embed (8)

concat(all embeds) = 264 dims → FC(264, 128) → FC(128, 64) → action/value heads
```

This gives each feature group its own encoder, so "my move features" and "opponent's bench" are processed separately before combining. The network doesn't need to learn which features belong together.

Implementation: custom SB3 feature extractor class.

---

## Step 3: LSTM/GRU recurrent policy

For multi-turn strategy (predicting opponent switches, remembering revealed info):

- Use `RecurrentPPO` from sb3-contrib (if available for MaskablePPO)
- If not, implement a custom recurrent policy with GRU:
  - Hidden state: 64 dims
  - Observation → FC(478, 128) → GRU(128, 64) → policy/value heads
  - Hidden state carries information across turns within a battle
  - Reset hidden state at battle start (episode boundary)

**Warning:** Recurrent PPO is significantly harder to train. Only attempt after MLP-based approaches are exhausted.

---

## Step 4: Asymmetric actor-critic (from Panel #2)

The privileged critic sees full team information:
- Policy input: standard 478-dim observation (partial info)
- Value function input: 478-dim obs + full opponent team features (~200 additional dims)

Requires a custom SB3 policy class where:
- `forward()` uses the standard observation for the actor
- `predict_values()` uses the extended observation for the critic
- Full opponent team info is passed via the `info` dict from the environment

This directly addresses the variance from partial observability.

---

## Step 5: Self-play with Elo-based matchmaking

Once the agent can beat heuristic opponents:
- Maintain a pool of N frozen model snapshots at different skill levels
- Assign each snapshot an Elo rating based on match results
- Select opponents for training based on Elo proximity (within ±200)
- This ensures the agent always faces appropriately challenging opponents

Prevents the "mud wrestling" problem where two weak agents learn nothing from each other, and the "unreachable skill ceiling" where the opponent is too far ahead.

---

## Step 6: Population-Based Training (PBT)

Full PBT implementation for hyperparameter search:
- Run 8 agents in parallel with varied: learning_rate, ent_coef, gamma, temperature_anneal_speed
- Every 100K steps, evaluate all agents
- Bottom 25% clone top 25% weights and perturb hyperparameters by ±20%
- Run for 5M+ total steps across all agents

Requires significant compute but finds optimal hyperparameters automatically.

---

## Verification

These are research-tier techniques. Success criteria:
1. Type pre-training: agent picks super-effective moves >80% of the time when available
2. Feature grouping: explained variance improves by >50% over flat MLP
3. LSTM: agent demonstrates multi-turn planning (e.g., predicting opponent switches)
4. Asymmetric critic: variance in win rate across evaluation windows decreases by >30%
5. Self-play: agent discovers non-obvious strategies (verify via replay analysis)
