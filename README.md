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

The training pipeline is a three-phase curriculum. Each phase auto-advances when the win rate target is reached.

### Phases

| Phase | Script | Opponent | Win rate target | What the agent learns |
|---|---|---|---|---|
| A | `train.py` | RandomPlayer | 75% | Basic move selection, type advantages |
| B | `train.py` | MaxDamagePlayer | 65% | Defensive switching, not just attacking |
| C | `selfplay_train.py` | Frozen self-snapshots | — | Prediction, long-term strategy |

Phases A and B run sequentially from one command. Phase C is launched separately after B completes.

---

### Run management

Every training session creates an isolated folder: `runs/run_001/`, `runs/run_002/`, etc. Experiments never overwrite each other.

**Default — auto-resume an interrupted run:**
```bash
python src/train.py
```
If a previous run was killed mid-training, this resumes it from the last checkpoint automatically.

**Force a brand new experiment** (new hyperparameters, fresh start):
```bash
python src/train.py --new-run
```
Creates the next `runs/run_NNN/` and starts from scratch. The old run is untouched.

**Self-play after curriculum:**
```bash
python src/selfplay_train.py              # auto-detects latest completed curriculum run
python src/selfplay_train.py --run run_001  # continue a specific run
python src/selfplay_train.py --new-run    # fresh self-play experiment
```

Each run folder contains:
```
runs/run_001/
  models/          checkpoints (gitignored)
  logs/            TensorBoard data (gitignored)
  replays/         battle replays for this run (gitignored)
  training_log.md  win rate table per 10k steps (committed)
  run_info.json    hyperparameters + status (committed)
```

---

### Battle simulation
```bash
python scripts/battle_sim.py --agent1 best_model --agent2 maxdamage --n 50
python scripts/battle_sim.py --agent1 runs/run_001/models/phase_A_final --agent2 runs/run_002/models/phase_B_final --n 30
python scripts/battle_sim.py --agent1 random --agent2 best_model --n 20
```
`--agent1/--agent2` accept: `"random"`, `"maxdamage"`, or any path to a model checkpoint.

**TensorBoard**
```bash
tensorboard --logdir runs/run_001/logs/
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

## Observation Space (139 dims)

```
Own active        16 dims   HP, 6 boosts (incl accuracy/evasion), status, active, fainted,
                            type_1, type_2, base_atk, base_def, base_spa, base_spe*
Own moves         20 dims   4 moves × 5 features (base_power, type, PP, effectiveness, accuracy)
Own bench         36 dims   6 slots × (HP, fainted, type_1, type_2, status, base_spe*)
Opponent active   15 dims   HP, 6 boosts (incl accuracy/evasion), status, fainted,
                            type_1, type_2, base_atk, base_def, base_spa, base_spe*
Opponent bench    30 dims   6 slots × (HP, fainted, revealed, type_1, type_2)
Opp revealed mvs  20 dims   up to 4 seen opponent moves × 5 features (padded)
Trapping           2 dims   own_trapped, own_maybe_trapped (Wrap/Bind flags)
```
`* base_spe is paralysis-adjusted — PAR quarters speed in Gen 1 independently of stage boosts`

Key design decisions:
- **Base stats included** — all Pokemon are level 100 with maxed DVs in randombattle, so stats are fully deterministic from species. Agent reasons about damage and speed tiers exactly as a human would
- **Paralysis speed correction** — `boosts["spe"]` only captures stage changes; PAR's 0.25× speed penalty is applied directly to the speed value so the agent sees the real effective speed
- **Move accuracy included** — distinguishes Thunder (70%) from Thunderbolt (100%), Fire Blast from Flamethrower
- **Both sides get all 6 boosts** — accuracy + evasion included; Double Team / Sand-Attack are real Gen 1 strategies
- **Own bench status** — agent knows if its switch-in is asleep or paralyzed before committing
- **Trapping flags** — Wrap/Bind/Fire Spin lock the opponent; agent knows when it cannot switch
- **Up to 4 opponent revealed moves** — memory of what the opponent has used, with threat level vs current active
- **Unrevealed opponent slots** = `(-1, 0, 0, 0, 0)` — distinct signal from fainted `(0, 1, 0, 0, 0)`

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
