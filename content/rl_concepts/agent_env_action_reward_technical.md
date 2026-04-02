# The Agent, Environment, Action & Reward — Technical Reference

For the developer. Precise API references, exact tensor shapes, and implementation notes.

---

## The Agent

**What it is:** A neural network that decides which move to make each turn.

**Inputs:** A 64-number snapshot of the current battle state (see below).

**Outputs:** A probability distribution over all possible actions. It picks one.

**What it knows:** Only what's in the 64-number observation. It cannot see the
opponent's hidden Pokemon, their unrevealed moves, or anything outside that vector.

**What it does NOT do:** It does not simulate future turns, look ahead, or use
any hand-coded Pokemon knowledge. Everything it learns comes from experience.

**Implementation:** `MaskablePPO` from sb3-contrib with `MlpPolicy`.
Default SB3 MLP: two hidden layers of 64 units each, tanh activations.
Separate actor and critic heads sharing the same trunk.
Trained on GPU (CUDA 12.0). Rollout buffer holds `n_steps × N_ENVS = 2048` transitions.

```
Battle state (64 numbers)
        │
        ▼
┌───────────────────┐
│   Neural Network  │  ← this is the agent
│   (MlpPolicy)     │
│                   │
│  Input layer: 64  │
│  Hidden layers    │
│  Output layer: 10 │
└───────────────────┘
        │
        ▼
 Probability over 10 actions
 e.g. [0.05, 0.05, 0.60, 0.10, 0.05, 0.05, 0.05, 0.02, 0.02, 0.01]
        │
        ▼
 Sample → "use move 3 (Thunderbolt)"
```

---

## The Environment

**What it is:** Everything that is NOT the agent. It receives an action, runs the
battle logic, and returns the next state and reward.

**Concretely:** Two pieces of software working together:

```
┌─────────────────────────────────────────────────────────┐
│  Pokemon Showdown (Node.js server, localhost:8000)      │
│  ─ runs the actual Gen 1 battle simulation              │
│  ─ enforces rules, calculates damage, applies RNG       │
│  ─ manages both players' turns                          │
└─────────────────────────────────────────────────────────┘
                          │  WebSocket
┌─────────────────────────────────────────────────────────┐
│  poke-env + Gen1Env (Python)                            │
│  ─ translates Showdown's protocol into Python objects   │
│  ─ calls embed_battle() to build the 64-number vector   │
│  ─ calls calc_reward() to compute the reward            │
│  ─ presents a Gymnasium-compatible step()/reset() API   │
└─────────────────────────────────────────────────────────┘
```

**What the environment controls that the agent cannot:**
- The opponent's moves (played by RandomPlayer during training)
- RNG: damage rolls, miss chances, crit rates
- Which of the opponent's Pokemon are revealed
- Whether a status condition was applied

**Partial observability:** The environment knows everything about both sides.
The agent only sees what `embed_battle()` puts into the 64-number vector.
The opponent's full team is hidden until each Pokemon is sent out.

**Env stack (code path):**
```
Gen1Env(SinglesEnv)          src/env/gen1_env.py
  └─ SingleAgentWrapper      poke_env.environment.single_agent_wrapper
       └─ SB3Wrapper         src/env/gen1_env.py  ← unwraps dict obs, adds action_masks()
            └─ DummyVecEnv   stable_baselines3.common.vec_env
```

**Observation tensor:** `Box(-1, 1, shape=(64,), dtype=float32)`
Built by `embed_battle(battle)` — called once per turn per agent.

**Action mask tensor:** `array(shape=(10,), dtype=bool)`
Built by `SinglesEnv.get_action_mask(battle)` — called same frequency.
Exposed via `SB3Wrapper.action_masks()` which MaskablePPO calls each step.

---

## An Action

**What it is:** A single integer (0–9) representing one decision per turn.

**The 10 possible actions:**

```
Index   Meaning
─────   ────────────────────────────────────────────────
  0     Switch to team slot 1 (first bench Pokemon)
  1     Switch to team slot 2
  2     Switch to team slot 3
  3     Switch to team slot 4
  4     Switch to team slot 5
  5     Switch to team slot 6  (usually fainted / active)
  6     Use move in slot 1
  7     Use move in slot 2
  8     Use move in slot 3
  9     Use move in slot 4
```

**Action masking:** Not all 10 actions are legal every turn.
- Can't switch to a fainted Pokemon
- Can't use a move with 0 PP
- Can't switch if trapped (Wrap/Bind)
- Must switch if active Pokemon just fainted

Illegal actions are masked to 0 probability before the agent samples.
The agent never picks an illegal action — it never needs to learn "don't do that."

```
Raw output:  [0.05, 0.05, 0.60, 0.10, 0.05, 0.05, 0.05, 0.02, 0.02, 0.01]
Mask:        [   0,    0,    1,    1,    0,    0,    1,    1,    0,    0 ]
             (slots 0,1,4,5 = fainted/active;  slot 8 = no PP left)

After mask:  [0.00, 0.00, 0.60, 0.10, 0.00, 0.00, 0.05, 0.02, 0.00, 0.00]
Renormalize: [0.00, 0.00, 0.78, 0.13, 0.00, 0.00, 0.06, 0.03, 0.00, 0.00]
Pick:        move slot 3 (index 8 → move 3)
```

**One action = one full turn.** The environment handles both sides: the agent
picks its action, the opponent (RandomPlayer) picks theirs, Showdown resolves
the turn, and the new state is returned.

**Action space size:** `Discrete(10)` — constant for Gen 1 singles.
`SinglesEnv.get_action_space_size(gen=1)` returns 10.
(Gen 9 returns more due to Mega/Z-move/Dynamax/Tera slots — not relevant here.)

---

## The Reward

**What it is:** A number the agent receives after each action, telling it how
well it's doing. The agent's goal is to maximise total reward over a battle.

**Our reward function (Phase 2):**

```
+1.0  →  agent wins the battle       (all opponent Pokemon fainted)
−1.0  →  agent loses the battle      (all agent Pokemon fainted)
 0.0  →  battle is still ongoing     (every intermediate turn)
```

**Why sparse (no intermediate rewards)?**

Simpler and cleaner for Phase 2. The agent must figure out on its own which
moves led to winning. Reward shaping (e.g. +0.05 per opponent KO) will be
added in Phase 6 only if learning stalls.

**Credit assignment problem:** With 0 reward for 40 turns and then ±1,
the agent must work backwards to figure out which early decisions mattered.
This is hard. It's why PPO + GAE exists — the critic's value estimates
bridge the gap between early moves and the eventual outcome.

```
Turn:    1    2    3   ...  38   39   40 (KO)
Reward:  0    0    0   ...   0    0   +1

Agent must learn: "My Thunderbolt on turn 3 weakened their Rhydon enough
                   that I could KO it on turn 39. That was a good move."
```

**Implementation:** `Gen1Env.calc_reward(battle)` calls `reward_computing_helper`
with `fainted_value=0, hp_value=0, victory_value=1.0`.
`reward_computing_helper` returns `+victory_value` on win, `-victory_value` on loss.

**What the agent optimises for:**
Winning. Not winning fast, not taking less damage, not showing off.
(γ=0.99 introduces a tiny preference for faster wins — see ppo_explained.md)

**Phase 6 shaping candidates (add only if learning stalls):**
- `+0.05` per opponent KO (speeds up learning signal)
- `-0.05` per own KO (penalises reckless play)
- `+0.01 × hp_advantage` per turn (rewards staying healthy)

---

## Full Loop (one turn)

```
Environment                          Agent
──────────                           ─────
Battle state
  → embed_battle()
  → 64-number vector          →      Observe state
                                     Apply action mask
                                     Sample action (e.g. "use Surf")
  ← action index              ←      Return action

Showdown executes turn
  (agent uses Surf, opponent uses Earthquake, damage applied, RNG resolved)

New battle state
  → calc_reward() → 0.0 (ongoing)
  → embed_battle() → new 64-number vector
                                →    Observe new state + reward
                                     Store (s, a, r, s') in rollout buffer

... repeat for ~40 turns until one side wins ...

calc_reward() → +1.0 or −1.0        Agent receives terminal reward
episode ends, reset() called
new battle begins
```
