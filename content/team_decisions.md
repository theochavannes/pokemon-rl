# Team Decision Log

Running record of architectural decisions, debates, and conclusions.
Updated after every significant team discussion.

---

## 2026-04-02 — Initial Plan Review

**Question:** PPO vs DQN for this project?

**Debate:**
- DQN uses experience replay (off-policy). When the opponent policy changes (self-play Phase 5), old transitions are stale and learned Q-values become wrong.
- PPO is on-policy — discards old data each update. Required for stable self-play.
- MaskablePPO has first-class action masking in sb3-contrib. DQN would need custom surgery.

**Decision:** PPO. Closed.

---

## 2026-04-02 — Observation Space: Partial Observability

**Question:** How to encode opponent's unrevealed team slots?

**Problem:** `-1` for unrevealed HP conflated "not seen yet" with "fainted" (both looked similar).

**Decision:** Add explicit `revealed` binary flag per opponent slot (+6 dims).
- Unrevealed slot: `[-1, 0, 0]` (HP=-1, fainted=0, revealed=0)
- Revealed, alive: `[0.7, 0, 1]`
- Fainted: `[0.0, 1, 1]`

Obs space: **64 dims** (not 58 as originally specced).

---

## 2026-04-02 — VecEnv Strategy

**Question:** SubprocVecEnv from the start or DummyVecEnv first?

**Problem:** SubprocVecEnv forks processes. Each subprocess needs:
- Its own asyncio event loop
- Unique bot usernames (can't have 4 agents all named "PPOAgent")
- poke-env 0.13 supports this, but bugs are hard to diagnose across processes.

**Decision:** `DummyVecEnv` (N=1) in Phase 4 for validation. Scale to `SubprocVecEnv` (N=4) only after single-env training is confirmed working.
Upgrade path is a 2-line change in `train.py` (commented in code).

---

## 2026-04-02 — Discount Factor γ

**Question:** Does γ matter when all rewards are terminal (+1/-1)?

**Analysis:**
- With terminal-only rewards: `G_t = γ^(T−t) · (±1)`
- γ controls how much signal reaches early moves
- γ=0.99, 40-move battle: `G_0 ≈ 0.67` — adequate
- γ=0.95: `G_0 ≈ 0.13` — early moves barely learn
- γ=1.0: critic variance explodes (predicting ±1 from 40 steps out with no anchoring)

**GAE interaction:** Effective decay is `(γλ)^l = (0.9405)^l`. After 40 moves: ~9% signal at move 1. Agent learns late-battle first, then mid, then early. Expected behavior.

**Side effect:** γ<1 slightly rewards winning faster → mild aggressive-play prior. Acceptable.

**Decision:** γ=0.99, λ=0.95 (SB3 defaults). No custom tuning needed.

---

## 2026-04-02 — Opponent Architecture in Training Env

**Question:** Should the opponent player be created inside Gen1Env or passed in?

**Options:**
1. Inside Gen1Env — self-contained, no external lifecycle management
2. Passed in — more flexible but creates shared-state problems with SubprocVecEnv

**Decision:** Opponent created inside `make_env()` factory function with `start_listening=False`.
The `SingleAgentWrapper` injects the battle directly into the opponent — no WebSocket connection needed for the puppet opponent.
Each env index gets unique usernames: `PPOAgent_0`, `RandomOpp_0`, `RandPuppet_0`.

---

## 2026-04-02 — Observation Pipeline

**Finding:** poke-env's `PokeEnv` returns observations as dicts:
`{"observation": array(64,), "action_mask": array(N,)}`

MaskablePPO needs a flat array observation and a separate `action_masks()` method.

**Decision:** Add `SB3Wrapper(gymnasium.Wrapper)` that:
- Unwraps dict obs → flat `array(64,)` for SB3
- Stores last action mask
- Exposes `action_masks() -> bool array` for MaskablePPO

Full stack: `Gen1Env → SingleAgentWrapper → SB3Wrapper → DummyVecEnv → MaskablePPO`

---

## 2026-04-02 — poke-env API Corrections (found during smoke test)

**Issue 1:** `observation_spaces` not set automatically.
`PokeEnv.__init__` sets `action_spaces` but not `observation_spaces`. Subclasses must set it explicitly. Fixed by adding `__init__` to `Gen1Env` that sets `self.observation_spaces = {agent: self.describe_embedding() for agent in self.possible_agents}`.

**Issue 2:** `calc_reward` signature changed in poke-env 0.13.
Original plan used `calc_reward(self, last_battle, current_battle)`. Actual API is `calc_reward(self, battle)` — single argument. Fixed.

**Issue 3:** Stale battle state on reconnect.
If a run crashes mid-battle, the Showdown server keeps the battle alive. Reconnecting with the same username restores the battle, causing `reset_battles()` to fail on the next run. Fixed by appending a 6-char random suffix to all usernames in `make_env()`.

---

## 2026-04-02 — Win Rate Tracking

**Question:** How to evaluate win rate during training without a separate eval loop?

**Decision:** `WinRateCallback` reads terminal episode rewards from SB3's `infos` dict.
SB3 automatically adds `{"episode": {"r": total_reward}}` when an episode ends.
- `r > 0` → win
- `r < 0` → loss

Window of last 100 episodes. Logged to TensorBoard every 10k steps.
Best model saved whenever win rate improves (with minimum 50-episode window).
