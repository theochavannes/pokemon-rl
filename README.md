# Pokemon RL

Training a reinforcement learning agent to play competitive Gen 1 (RBY) Pokemon using [Pokemon Showdown](https://github.com/smogon/pokemon-showdown) as the battle simulator.

The agent learns through self-play using Proximal Policy Optimization (PPO) with action masking.

---

## Progress

| Phase | Description | Status |
|---|---|---|
| 1 | Infrastructure — Showdown server, Python env, dependencies | ✅ Done |
| 2 | Gen 1 Gymnasium environment (64-dim observation space) | ✅ Done |
| 3 | Baseline agents — MaxDamagePlayer 96/100 vs Random | ✅ Done |
| 4 | PPO training loop | 🔄 In progress |
| 5 | Self-play with policy snapshot pool | ⏳ Pending |
| 6 | Evaluation, replay analysis, iteration | ⏳ Pending |

---

## Stack

- **Simulator** — [Pokemon Showdown](https://github.com/smogon/pokemon-showdown) (local Node.js server, git submodule)
- **RL framework** — [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) + [sb3-contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib) (MaskablePPO)
- **Environment** — [poke-env](https://github.com/hsahovic/poke-env) 0.13 (Gymnasium interface to Showdown)
- **Battle format** — `gen1randombattle` (random teams, no team preview)
- **Python** — 3.11, conda
- **GPU** — NVIDIA GeForce, CUDA 12.0

---

## Setup

**1. Clone with submodule**
```bash
git clone --recurse-submodules https://github.com/theochavannes/pokemon-rl.git
cd pokemon-rl
```

**2. Install Showdown dependencies**
```bash
cd showdown
npm install
cd ..
```

**3. Create conda environment**
```bash
conda create -n pokemon_rl python=3.11
conda activate pokemon_rl
pip install poke-env stable-baselines3 sb3-contrib torch
```

**4. Start the Showdown server** (keep this terminal open)
```bash
cd showdown
node pokemon-showdown start --no-security
```

**5. Verify everything works**
```bash
python scripts/verify_setup.py
```

---

## Running

**Baseline benchmark** (MaxDamagePlayer vs RandomPlayer, 100 battles)
```bash
python scripts/benchmark_heuristic.py
# Expected: MaxDamage wins ~96%
```

**Train the RL agent** *(Phase 4 — coming soon)*
```bash
python src/train.py
```

---

## Project Structure

```
pokemon_rl/
├── showdown/               # Pokemon Showdown server (git submodule)
├── src/
│   ├── env/gen1_env.py     # Gymnasium environment — 64-dim obs, ±1 reward
│   ├── agents/
│   │   └── heuristic_agent.py  # MaxDamagePlayer baseline
│   ├── train.py            # PPO training loop (Phase 4)
│   ├── selfplay_train.py   # Self-play training (Phase 5)
│   └── callbacks.py        # WinRateCallback
├── scripts/
│   ├── verify_setup.py     # Smoke test
│   └── benchmark_heuristic.py
├── replays/
│   └── notable/            # Curated battle replays (open in browser)
├── models/                 # Saved checkpoints (gitignored)
├── logs/                   # TensorBoard logs (gitignored)
└── content/                # YouTube video documentation
```

---

## Observation Space (64 dims)

```
Own active Pokemon     10 dims   HP, 6 boosts, status, active flag, fainted
Own moves               16 dims   4 moves × (base_power, type, PP, effectiveness)
Opponent active          7 dims   HP, 4 boosts, status, fainted
Own bench               12 dims   6 slots × (HP, fainted)
Opponent bench          18 dims   6 slots × (HP, fainted, revealed)
Speed context            1 dim    own_speed > opp_speed?
```

Reward: `+1.0` win · `-1.0` loss · `0.0` ongoing

---

## Key Gen 1 Mechanics

- **1/256 miss** — nominally 100% moves have a tiny miss chance (Showdown replicates this)
- **Freeze is permanent** — one of the strongest status conditions
- **Single Special stat** — no Sp.Atk / Sp.Def split
- **Crit rate from Speed** — fast Pokemon (Tauros, Starmie) crit far more often
- **No held items, no team preview**
- **Sleep Clause** — max 1 opponent Pokemon asleep at a time
