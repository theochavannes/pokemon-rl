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
*(pending)*

## Phase 5 — Self-Play
*(pending)*
