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

## Open Question (from [REVIEW])

> The discount factor γ < 1 makes agents prefer faster rewards.
> But our reward is always terminal (+1 or −1 at end of battle).
> **Does γ even matter here? What value should we use?**

Hint: think about what γ does to the advantage estimates
when every intermediate reward is 0.
```
