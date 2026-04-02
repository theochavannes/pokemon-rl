# PPO Explained — From Q-Learning to Proximal Policy Optimization

## The RL Family Tree

```
Action-Value Methods          Policy Gradient Methods
(Q-learning, DQN)             (REINFORCE → A2C → PPO)
         |                              |
  Learn Q(s,a)                 Learn π(a|s) directly
  Policy is implicit            Policy is explicit
  Off-policy                    On-policy
```

---

## Step 1 — What You Already Know: Q-Learning

Learn the **action-value function**:

```
Q(s, a) = expected total return from state s, taking action a
```

Policy is implicit — always pick the best action:
```
π(s) = argmax_a Q(s,a)
```

**Limitation for this project:** off-policy (learns from old data).
Bad for self-play — old battles against past opponents pollute the update.

---

## Step 2 — Policy Gradient (Sutton Ch. 13: REINFORCE)

Directly parameterize the policy as a neural network: **π_θ(a|s)**
→ outputs a probability distribution over actions.

**Update rule:**
```
∇θ J(θ) = E[ ∇θ log π_θ(a|s) · Gt ]
```
- `log π_θ(a|s)` → direction to make action a more/less probable
- `Gt` → how good was this trajectory (full discounted return)

**The problem:** `Gt` has enormous variance.
A battle might end +1 after 40 moves — which of those 40 moves was actually good?
Training is very slow and unstable.

---

## Step 3 — Actor-Critic: Killing the Variance

Replace the noisy `Gt` with the **advantage function**:

```
A(s, a) = Q(s, a) − V(s)
```

- `V(s)` = "how good is this state on average" → learned **baseline** (the Critic)
- `A(s, a)` = "how much BETTER was this action than average"

**Two networks:**
```
┌─────────────────────────────────────────────────────┐
│  INPUT: 64-dim observation vector                   │
│         (HP, boosts, moves, team state, speed...)   │
└──────────────────┬──────────────────────────────────┘
                   │  shared layers
          ┌────────┴────────┐
          ▼                 ▼
     ACTOR HEAD         CRITIC HEAD
   π_θ(a|s)              V_φ(s)
  (prob over actions)  (scalar value)
  "what to do"         "how good is this position"
```

---

## Step 4 — PPO: Don't Destroy Your Policy

**The core problem with A2C:**
A large gradient step can move the policy too far → catastrophic forgetting.
The agent unlearns everything and has to start over.

**PPO's fix — clip the update ratio:**

```
         π_θ(a|s)
r_t(θ) = ──────────    (how much has the policy changed?)
         π_θ_old(a|s)
```

**Clipped objective:**
```
L = E[ min(
      r_t(θ) · A_t,
      clip(r_t(θ), 1−ε, 1+ε) · A_t
    )]
```

With `ε = 0.2` (standard):
- If `r_t > 1.2` → the policy moved too far in the positive direction → STOP
- If `r_t < 0.8` → the policy moved too far in the negative direction → STOP

**Plain English:** collect experience, do several gradient steps,
but never let the policy drift more than 20% away from where it started.

---

## The PPO Training Loop (Phase 4)

```
Repeat until 500k steps:
│
├── 1. ROLLOUT
│       Run 4 parallel battles for N steps
│       Store (state, action, reward, next_state) for each step
│
├── 2. COMPUTE ADVANTAGES
│       Use critic V(s) + discounted rewards → A_t for each step
│
├── 3. PPO UPDATE (K epochs on the same batch)
│       Clip ratio update on actor
│       MSE loss on critic
│
└── 4. EVALUATE every 50k steps
        Win rate vs RandomPlayer
        Win rate vs MaxDamagePlayer
        Save checkpoint if best so far
```

---

## Why PPO (not DQN) for Pokemon

| | DQN | PPO |
|---|---|---|
| Self-play (Phase 5) | ❌ Stale replay buffer | ✅ On-policy, always fresh |
| Action masking | ❌ Awkward | ✅ Zero illegal probs before sampling |
| Sparse terminal reward | ⚠️ Works but slow | ✅ Advantage reduces variance |
| Continuous obs (64-dim) | ⚠️ Needs extra tricks | ✅ Natural fit for policy networks |

---

## The γ Question — Resolved

> With only terminal rewards, does γ matter?

**Yes.** With a battle of length T, the return at timestep t is:
```
G_t = γ^(T−t) · (±1)
```
γ controls how much the terminal signal decays backward through time.

```
γ = 0.99, 40-move battle:  G_0 = 0.99^40 ≈ 0.67   ✅ signal reaches early moves
γ = 0.95, 40-move battle:  G_0 = 0.95^40 ≈ 0.13   ⚠️  early moves barely learn
γ = 1.00                                            ❌  critic variance explodes
```

**But the real parameter pair is γλ (gamma × GAE lambda):**

PPO uses Generalized Advantage Estimation (GAE):
```
A_t^GAE = Σ (γλ)^l · δ_{t+l}
where δ_t = r_t + γV(s_{t+1}) − V(s_t)  (TD error)
```

With sparse terminal rewards, δ_t ≈ −V(s_t) for every mid-battle step.
The advantage signal decays at rate (γλ)^l, not just γ^l.

**With SB3 defaults (γ=0.99, λ=0.95):**
```
γλ = 0.9405
Over 40 moves: (0.9405)^40 ≈ 0.09
→ Move 1's advantage = ~9% of the terminal signal
→ Agent learns late-battle first, then mid, then early — expected behavior
```

**Decision: keep SB3 defaults. No custom tuning needed for Phase 4.**

Side effect: γ < 1 creates a subtle prior toward winning faster (aggressive play).
This is acceptable — possibly even beneficial for Gen 1 offensive meta.
```
