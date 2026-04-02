# Pokemon RL

Training a reinforcement learning agent to play competitive Gen 1 (RBY) Pokemon using [Pokemon Showdown](https://github.com/smogon/pokemon-showdown) as the battle simulator.

The agent learns through a curriculum — first mastering basic play against random and heuristic opponents, then improving through self-play against frozen snapshots of itself.

---

## Progress

| Phase | Description | Status |
|---|---|---|
| 1 | Infrastructure — Showdown server, Python env, dependencies | ✅ Done |
| 2 | Gen 1 Gymnasium environment (64-dim observation space) | ✅ Done |
| 3 | Baseline agents — MaxDamagePlayer 96/100 vs Random | ✅ Done |
| 4A | Curriculum: PPO vs RandomPlayer — peaked at 99% win rate | ✅ Done |
| 4B | Curriculum: PPO vs MaxDamagePlayer | 🔄 In progress |
| 5 | Self-play with frozen policy snapshots | ⏳ Pending |
| 6 | Evaluation, replay analysis, tournament | ⏳ Pending |
| 7 | Competitive RBY format + Gen 1 bug experiments | ⏳ Pending |

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

## Training

The training pipeline is a curriculum — the agent faces progressively stronger opponents and auto-advances when it reaches the win rate target for each phase.

**Phase A → B (RandomPlayer → MaxDamagePlayer)**
```bash
python src/train.py
```
- Phase A: trains vs RandomPlayer until 75% win rate (cap 150k steps)
- Phase B: trains vs MaxDamagePlayer until 65% win rate (cap 200k steps)
- Checkpoints saved every 50k steps to `models/`
- Clean per-step output: `step 20,000 │ vs MaxDamage:  61.0%  [████████████░░░░░░░░]`

**Phase C (Self-play)**
```bash
python src/selfplay_train.py
```
- Loads `phase_B_final.zip` as seed for both sides
- Frozen opponent updates every 50k steps to the latest best model
- Opponent hot-swapped without rebuilding the environment

**Battle simulation — compare any two agents**
```bash
python scripts/battle_sim.py --agent1 best_model --agent2 maxdamage --n 50
python scripts/battle_sim.py --agent1 phase_A_final --agent2 selfplay_final --n 30
python scripts/battle_sim.py --agent1 random --agent2 best_model --n 20
```
Accepts: `"random"`, `"maxdamage"`, or any model name/path.

**TensorBoard**
```bash
tensorboard --logdir logs/
```

---

## Project Structure

```
pokemon_rl/
├── showdown/               # Pokemon Showdown server (git submodule)
├── src/
│   ├── env/gen1_env.py     # Gymnasium environment — 64-dim obs, ±1 reward
│   ├── agents/
│   │   ├── heuristic_agent.py   # MaxDamagePlayer baseline
│   │   └── policy_player.py     # FrozenPolicyPlayer (self-play opponent)
│   ├── train.py            # Curriculum training: Random → MaxDamage
│   ├── selfplay_train.py   # Self-play with frozen snapshots
│   └── callbacks.py        # WinRateCallback — logging, milestones, early stop
├── scripts/
│   ├── verify_setup.py
│   ├── benchmark_heuristic.py
│   ├── battle_sim.py       # Simulate any two agents head-to-head
│   └── tournament.py       # Round-robin tournament across checkpoints (Phase 6)
├── replays/
│   └── notable/            # Curated battle replays (open in browser)
├── models/                 # Saved checkpoints (gitignored)
├── logs/                   # TensorBoard logs (gitignored)
└── content/                # Documentation and YouTube video notes
    ├── agent_knowledge.md  # What the agent sees and doesn't see
    ├── hooks.md            # Video milestones and content angles
    └── rl_concepts/        # RL theory docs for technical and general audiences
```

---

## Observation Space (64 dims)

```
Own active Pokemon     10 dims   HP, 6 boosts, status, active flag, fainted
Own moves              16 dims   4 moves × (base_power, type, PP, effectiveness)
Opponent active         7 dims   HP, 4 boosts, status, fainted
Own bench              12 dims   6 slots × (HP, fainted)
Opponent bench         18 dims   6 slots × (HP, fainted, revealed)
Speed context           1 dim    own_speed > opp_speed?
```

The agent learns from **move features, not slot positions** — base power, type, PP fraction, and type effectiveness vs the current opponent. In random battles where teams change every match, the agent must generalize from features rather than memorizing move slots.

Reward: `+1.0` win · `-1.0` loss · `0.0` ongoing (sparse terminal reward)

---

## Key Gen 1 Mechanics

- **1/256 miss** — nominally 100% moves have a tiny miss chance (Showdown replicates this)
- **Freeze is permanent** — one of the strongest status conditions
- **Single Special stat** — no Sp.Atk / Sp.Def split
- **Crit rate from Speed** — fast Pokemon (Tauros, Starmie) crit far more often
- **No held items, no team preview**
- **Sleep Clause** — max 1 opponent Pokemon asleep at a time
- **Ghost/Psychic bug** — Ghost moves are coded as immune to Psychic (should be 2×); experiment planned for Phase 7
