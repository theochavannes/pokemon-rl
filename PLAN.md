# Pokemon RBY RL Agent — Implementation Plan

## Project Overview
Build a Reinforcement Learning agent to play competitive Gen 1 (RBY) Pokemon on Pokemon Showdown.

**Stack:**
- Simulator: Pokemon Showdown (Node.js, local server) — in `showdown/`
- Python env: conda `pokemon_rl` env at `C:\Users\theoc\miniconda3\envs\pokemon_rl`
  - Python 3.11
  - poke-env 0.13.0 (gymnasium interface to Showdown)
  - stable-baselines3 + sb3-contrib (MaskablePPO)
  - PyTorch with CUDA (NVIDIA GeForce ~6GB VRAM, CUDA 12.0)
- Battle format: `gen1randombattle` (random teams, simpler to start)
- RL algorithm: PPO with action masking (`MaskablePPO` from sb3-contrib)
- Self-play: Phase 5 — pool of past policy snapshots

**Python interpreter:** `C:/Users/theoc/miniconda3/envs/pokemon_rl/python.exe`
**Project root:** `C:\Users\theoc\code\pokemon_rl\`
**Showdown server:** `C:\Users\theoc\code\pokemon_rl\showdown\` (gitignored, run separately)

---

## How to start the Showdown server
```bash
cd C:\Users\theoc\code\pokemon_rl\showdown
node pokemon-showdown start --no-security
```
The `--no-security` flag disables login requirements for local bots.
Server runs on port 8000 by default. poke-env connects to `ws://localhost:8000/showdown/websocket`.

---

## Phase 1 — Infrastructure (DONE)
- [x] Node.js installed (v24.14.1)
- [x] Pokemon Showdown cloned to `showdown/`
- [x] Showdown npm deps installed (`showdown/node_modules/`)
- [x] conda env `pokemon_rl` created (Python 3.11)
- [x] poke-env 0.13.0 installed
- [x] stable-baselines3 + sb3-contrib installed
- [x] PyTorch with CUDA installed
- [x] git repo initialized

**Remaining Phase 1 task:**
- [x] Write `scripts/verify_setup.py` — start Showdown, connect two RandomPlayers, run 1 battle, confirm it completes

---

## Phase 2 — Gen 1 Gymnasium Environment

### File to create: `src/env/gen1_env.py`

Subclass `poke_env.player.SinglesEnv`. Key methods to implement:

#### `calc_reward(last_battle, current_battle) -> float`
- Return `+1.0` if we won, `-1.0` if we lost, `0.0` otherwise
- Optional shaping (add later): `+0.05` per opponent KO, `-0.05` per own KO

#### `embed_battle(battle) -> ObsType`
Return a numpy float32 array. Suggested features (~56 dims):

```
For own active Pokemon (10 dims):
  - HP fraction (1)
  - Boosts: atk, def, spe, spc, acc, eva — each encoded as boost/6 (6)
  - Status: none/par/slp/brn/frz/psn — one-hot (6) — but only 1 active so just index it (1)
  - Is active? (always 1, placeholder for future use) (1)
  - Fainted? (1)

For own moves (4 moves × 4 dims = 16 dims):
  - Base power normalized /150 (1)
  - Type index normalized /18 (1)  
  - PP fraction (current/max) (1)
  - Damage multiplier vs opponent active (1)

For opponent active Pokemon (7 dims):
  - HP fraction (1)
  - Boosts: atk, def, spe, spc — normalized /6 (4)
  - Status one-hot index (1)
  - Fainted? (1)

For own team (6 slots × 2 dims = 12 dims):
  - HP fraction (1)
  - Fainted? (1)

For opponent team (6 slots × 3 dims = 18 dims):
  - HP fraction if revealed, else -1 (1)
  - Fainted? (1)
  - Revealed? (1)  ← explicit flag; distinguishes "not seen yet" from "fainted"

Speed context (1 dim):
  - Own active speed > opponent active speed? (binary, 1)
```

Total: 64 dims. The `revealed` flag per opponent slot is intentional — partial observability
(not knowing the opponent's team) is a core strategic signal in Gen 1.

#### `describe_embedding() -> ObsType`
Return a gymnasium `Box` space with shape `(64,)`, dtype float32, low=-1, high=1.

#### Action masking
poke-env's `SinglesEnv` provides `action_masks()` automatically — verify it works for Gen 1.

### File to create: `src/agents/heuristic_agent.py`
A `MaxDamagePlayer` that always picks the move with highest expected damage.
Used as a benchmark — the RL agent should beat it >60% before moving to self-play.

---

## Phase 3 — Baseline Agents & Sanity Checks

### File to create: `scripts/benchmark_heuristic.py`
- Start local Showdown server (subprocess)
- Pit `MaxDamagePlayer` vs `RandomPlayer` for 100 battles
- Print win rate — expected >80% for MaxDamage
- This validates the environment is working correctly

---

## Phase 4 — PPO Training

### File to create: `src/train.py`

Key structure:
```python
from poke_env.player import RandomPlayer
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv

# Each subprocess gets unique usernames: PPOAgent_0..3 + RandomOpponent_0..3
# Each subprocess runs its own asyncio event loop
# Train MaskablePPO for ~500k steps
# Evaluate every 50k steps vs RandomPlayer and MaxDamagePlayer
# Save checkpoints to models/
```

**VecEnv strategy:** Use `DummyVecEnv` (N=1) during Phase 2 development for easy debugging.
Switch to `SubprocVecEnv` (N=4) in Phase 4. Each subprocess needs unique bot usernames and
its own asyncio event loop — poke-env 0.13 supports this.

Config to expose (argparse or simple constants at top of file):
- `N_ENVS = 4` — parallel environments (SubprocVecEnv in Phase 4)
- `TOTAL_TIMESTEPS = 500_000`
- `EVAL_FREQ = 50_000`
- `BATTLE_FORMAT = "gen1randombattle"`

### File to create: `src/callbacks.py`
- `WinRateCallback(EvalCallback)` — logs win rate vs fixed opponents
- Saves best model to `models/best_model.zip`

---

## Phase 5 — Self-Play

### File to create: `src/selfplay_train.py`
- Load best model from Phase 4 as starting point
- Every 100k steps, snapshot current policy to `models/snapshots/`
- Keep a pool of last 5 snapshots
- Training opponent = random pick from snapshot pool
- This is the core of getting beyond heuristic-level play

---

## Phase 6 — Evaluation & Iteration

- Watch replays: Showdown saves HTML replays, open in browser
- Expand observation space: encode type chart explicitly (18×18 matrix lookup for current matchup)
- Try LSTM policy to handle partial observability (opponent's hidden moves)
- Eventually: create a Showdown ladder account, test live

---

## Key Gen 1 (RBY) Mechanics to Be Aware Of

1. **1/256 miss** — moves that should be 100% accurate have a tiny miss chance. Showdown implements this.
2. **Freeze is permanent** — frozen Pokemon essentially can't fight. Very strong status.
3. **Sleep Clause** — only 1 opponent Pokemon can be put to sleep at a time (standard competitive rule).
4. **Hyper Beam no-recharge on KO** — if Hyper Beam KOs, no recharge needed.
5. **Crit formula based on Speed** — fast Pokemon (Tauros, Starmie) have very high crit rates.
6. **Special stat** — no Sp.Atk/Sp.Def split. One stat covers both.
7. **Wrap/Bind/Clamp trapping** — these moves trap the opponent for 2-5 turns (controversial in competitive).
8. **Badge boosts** — not relevant in standard competitive but Showdown handles it.
9. **No held items** — Gen 1 has no items in battle.
10. **No team preview** — you don't see opponent's team at start.

---

## File Structure Target

```
pokemon_rl/
├── showdown/          # gitignored — Pokemon Showdown server
├── src/
│   ├── env/
│   │   └── gen1_env.py        # Phase 2 — Gymnasium environment
│   ├── agents/
│   │   └── heuristic_agent.py # Phase 3 — MaxDamagePlayer
│   ├── train.py               # Phase 4 — PPO training
│   ├── selfplay_train.py      # Phase 5 — Self-play training
│   └── callbacks.py           # Phase 4 — WinRate callback
├── scripts/
│   ├── verify_setup.py        # Phase 1 — smoke test
│   └── benchmark_heuristic.py # Phase 3 — baseline benchmark
├── models/                    # gitignored — saved checkpoints
├── logs/                      # gitignored — tensorboard logs
├── PLAN.md                    # this file
└── .gitignore
```

---

## Notes for Future Agent Sessions

- Always activate the conda env: `C:/Users/theoc/miniconda3/envs/pokemon_rl/python.exe`
- Always start Showdown server first before running any poke-env code (in a separate terminal)
- poke-env connects to `ws://localhost:8000/showdown/websocket` by default
- The `--no-security` flag on Showdown is required for local bots (no login)
- Battle format is `gen1randombattle` throughout (random teams, simpler than fixed team OU)
- User codes in Python, prefers direct/concise explanations
