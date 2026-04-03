# Behavioral Cloning: Teaching the Agent to Imitate MaxDamagePlayer

## The Problem

Our RL agent starts with a randomly initialized neural network. The only prior knowledge it has is a hardcoded logit bias that makes it prefer attacking over switching. Against any opponent that attacks competently, this random policy loses ~98% of the time, and PPO cannot learn from consistent losses (the "death spiral" — see panel discussion #3).

## The Idea

Before RL training begins, we teach the agent to imitate an existing competent player — MaxDamagePlayer (our heuristic bot that always picks the highest-damage move). This is called **behavioral cloning** (BC).

After BC, the agent starts RL training already knowing "pick the high-damage move." It won't be perfect, but it'll win ~60-70% against weak opponents instead of ~2%. That gives PPO the variance in outcomes it needs to learn.

## How It Works — Step by Step

### Step 1: Generate Training Data

We run MaxDamagePlayer vs RandomPlayer for thousands of games. At every turn where MaxDamagePlayer makes a decision, we record three things:

1. **Observation** (421 floats): The full game state as the neural network sees it — our Pokemon's HP, moves, types, the opponent's Pokemon, etc.

2. **Action** (integer 0-9): What MaxDamagePlayer chose to do.
   - 0-5 = switch to team member 0-5
   - 6-9 = use move 1-4

3. **Action mask** (10 booleans): Which of the 10 actions are currently legal. For example, if we only have 2 moves with PP remaining and 3 living bench Pokemon, only those 5 actions are True.

This gives us a dataset of ~200,000 (observation, action, mask) tuples from ~5,000 games.

### Step 2: Train the Policy via Supervised Learning

We create a fresh MaskablePPO model (same architecture as for RL training) and train its **policy network** to predict MaxDamagePlayer's actions.

The training process:
1. Feed an observation into the policy network
2. The network outputs 10 "logits" (raw scores, one per action)
3. We mask the logits: set illegal actions to -infinity so they can never be chosen
4. We compute **cross-entropy loss**: how surprised is the model by the correct action?
   - If the model gives 90% probability to the correct action → low loss
   - If the model gives 10% probability to the correct action → high loss
5. Backpropagate the loss and update the network weights

We repeat this for 50 epochs (50 passes through the entire dataset), shuffling the data each time.

### Why Masked Cross-Entropy?

Regular cross-entropy would penalize the model for not assigning probability to illegal actions. But the agent can never choose illegal actions (the mask prevents it), so we shouldn't train on them.

By masking illegal actions to -infinity before computing the softmax, we effectively tell the model: "only distribute probability among the legal options."

```
Raw logits:     [0.5,  0.3, -0.1,  0.8,  0.2,  0.1,  1.2,  0.9,  0.4,  0.7]
Mask:           [ T,    F,    F,    F,    F,    F,    T,    T,    T,    F  ]
Masked logits:  [0.5, -inf, -inf, -inf, -inf, -inf,  1.2,  0.9,  0.4, -inf]
After softmax:  [0.12,  0,    0,    0,    0,    0,   0.38, 0.28, 0.17,  0  ]

If MaxDamage chose action 6 (move 1), loss = -log(0.38) = 0.97
If MaxDamage chose action 8 (move 3), loss = -log(0.17) = 1.77  (model less confident)
```

### Step 3: Save and Use as Warm-Start

After training, we save the model as `models/bc_warmstart.zip`. When `train.py` starts a new training run, it checks for this file:
- If found: loads the BC model as the starting policy (skips the logit bias)
- If not found: creates a fresh model with the logit bias (old behavior)

The RL training then fine-tunes this pre-trained policy. The agent starts with MaxDamagePlayer-level play and improves from there.

## What the Agent Learns from BC

- **Move selection**: Given a type matchup, which move deals the most damage?
- **Type effectiveness**: Fire moves are good against Grass Pokemon (high effectiveness in the observation → MaxDamage picks them → BC model learns the correlation)
- **When to switch**: MaxDamagePlayer only switches when it has no moves available (forced switch). The BC model learns this pattern too.

## What the Agent Does NOT Learn from BC

- **When to voluntarily switch**: MaxDamagePlayer never switches by choice. The agent must learn this through RL.
- **Defensive play**: MaxDamagePlayer doesn't consider incoming damage. The agent must learn to survive through RL.
- **Multi-turn strategy**: MaxDamagePlayer is purely reactive (best move this turn). Planning ahead comes from RL.

## Why This Works Better Than Starting from Scratch

| Metric | Random Start | BC Warm-Start |
|--------|-------------|---------------|
| Starting win rate vs Random | ~50% (logit bias helps) | ~90%+ (MaxDamage level) |
| Starting win rate vs SoftmaxDmg | ~2% (death spiral) | ~60-70% (competitive) |
| Best-move selection rate | 25% (random among 4 moves) | ~80%+ (learned from data) |
| PPO can learn from? | No (0% variance in outcomes) | Yes (mix of wins and losses) |

## Files

- `scripts/generate_bc_data.py` — Data collection (MaxDamage vs Random)
- `scripts/behavioral_cloning.py` — Supervised training
- `models/bc_warmstart.zip` — The saved pre-trained model
- `data/bc_training_data.npz` — The raw dataset
