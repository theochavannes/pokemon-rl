# Screen Recording Hooks

Moments worth capturing on screen, logged as they happen.

---

## Phase 1 — Infrastructure

### ✅ First successful battle (2026-04-02)
**What happened:**
```
Connecting to Showdown server at ws://localhost:8000 ...
(If this hangs or errors: start the server first with 'node pokemon-showdown start --no-security')
Running 1 battle: RandomPlayer vs RandomPlayer ...
Battle complete. Winner: P2
Setup verified — poke-env <-> Showdown connection is working.
```
**Why it matters:** First proof the full stack works end-to-end — Python agent → WebSocket → Showdown → battle result. The "it's alive" moment.

---

## Phase 2 — Environment
*(pending)*

## Phase 3 — Baselines

### ✅ MaxDamagePlayer vs RandomPlayer benchmark (2026-04-02)
**What happened:**
```
Running 100 battles: MaxDamagePlayer vs RandomPlayer ...

Results (100 battles):
  MaxDamagePlayer wins :  99 / 100  (99.0%)
  RandomPlayer wins    :   1 / 100  (1.0%)

ENVIRONMENT OK — win rate above 80% threshold.
```
**Why it matters:** Confirms the environment is working correctly end-to-end. 99% also sets a brutal bar for the RL agent — beating a type-aware heuristic >60% will require genuinely strategic play. The 1 loss is likely a catastrophic type matchup (e.g. Random got lucky team composition).

## Phase 4 — PPO Training

### ✅ First PPO gradient update (2026-04-02)
**What happened:**
```
Using cuda device
----------------------------
| time/              |     |
|    fps             | 77  |
|    iterations      | 1   |
|    total_timesteps | 128 |
----------------------------
------------------------------------------
| train/                  |              |
|    approx_kl            | 0.0035618674 |
|    clip_fraction        | 0            |
|    clip_range           | 0.2          |
|    entropy_loss         | -1.77        |
|    explained_variance   | -1.34        |
|    learning_rate        | 0.0003       |
|    loss                 | 0.0103       |
|    n_updates            | 10           |
|    policy_gradient_loss | -0.0128      |
|    value_loss           | 0.128        |
Smoke test PASSED
```
**Why it matters:** The agent is alive and learning on GPU. `explained_variance = -1.34` is exactly right for a fresh policy — the critic has no idea what's happening yet. This is the "it's learning" moment for Act 3 of the video. Contrast this number against the same metric after 500k steps.

## Phase 5 — Self-Play
*(pending)*
