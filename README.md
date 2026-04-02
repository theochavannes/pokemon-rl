# Pokemon RL

Training a reinforcement learning agent to play competitive Gen 1 (RBY) Pokemon using [Pokemon Showdown](https://github.com/smogon/pokemon-showdown) as the battle simulator.

The agent learns through mixed-opponent training with adaptive difficulty — facing all opponent archetypes simultaneously, starting mostly random and annealing to full-strength as performance improves.

---

## Progress

| Phase | Description | Status |
|---|---|---|
| 1 | Infrastructure — Showdown server, Python env, dependencies | Done |
| 2 | Gen 1 Gymnasium environment (153-dim observation space) | Done |
| 3 | Baseline agents — MaxDamage, TypeMatchup, Stall, AggressiveSwitcher | Done |
| 4 | Mixed-opponent training with epsilon annealing | In progress |
| 5 | Self-play with frozen policy snapshots | Pending |
| 6 | Evaluation, replay analysis, tournament | Pending |

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

### 4-phase curriculum

The agent progresses through increasingly difficult opponents. Each phase auto-advances when the win rate target is sustained at full difficulty.

| Phase | Opponent | How it works | Target |
|---|---|---|---|
| A | Random | Uniform random actions (baseline) | 85% |
| B | RandomAttacker | Random moves 85%, random switch 15% (realistic) | 80% |
| C | SoftmaxDamage | Moves sampled proportional to damage^(1/temp). Temperature anneals from 2.0 (soft) to 0.1 (near-argmax MaxDamage) based on win rate | 70% at temp=0.1 |
| D | Mixed+Self | All heuristic opponents + frozen self-play. Smooth difficulty via temperature/epsilon annealing | 60% |

```bash
python src/train.py --new-run    # start fresh
python src/train.py              # resume latest run
```

**Heuristic opponents** (used in Phase D mixed pool):
| Opponent | Strategy |
|---|---|
| MaxDamage | Always picks highest base_power x type_effectiveness move |
| TypeMatchup | Best type move, switches out of bad matchups |
| Stall | Status moves first (TWave, Toxic), then damage |
| AggressiveSwitcher | Switches to type-counter aggressively |

**Reward shaping** (linearly decays to 0 over 5000 battles):
- `hp_value=0.5` — reward for dealing damage, penalty for taking it
- `fainted_value=0.5` — reward for KOs, penalty for own Pokemon fainting
- `status_value=0.1` — reward for inflicting status conditions
- `victory_value=3.0` — winning always dominates

**Self-play** (after curriculum):
```bash
python src/selfplay_train.py
```

---

### Run management

Every training session creates an isolated folder: `runs/run_001/`, `runs/run_002/`, etc.

```
runs/run_001/
  models/          checkpoints (gitignored)
  logs/            TensorBoard data (gitignored)
  replays/         battle replays per phase (gitignored)
    notable/       milestone replay snapshots
  training_log.md  win rate table (committed)
  run_info.json    hyperparameters + status (committed)
```

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
│   ├── env/gen1_env.py     # Gymnasium environment — 153-dim obs, shaped reward
│   ├── agents/
│   │   ├── heuristic_agent.py   # 4 heuristic opponents + epsilon wrappers
│   │   └── policy_player.py     # FrozenPolicyPlayer (self-play opponent)
│   ├── train.py            # Mixed-opponent training with epsilon annealing
│   ├── selfplay_train.py   # Self-play with frozen snapshots
│   ├── callbacks.py        # WinRateCallback — logging, milestones, epsilon decay
│   ├── obs_transfer.py     # Obs-space transfer learning for model compatibility
│   └── run_manager.py      # Run isolation (runs/run_NNN/)
├── scripts/
│   ├── verify_setup.py
│   ├── benchmark_heuristic.py
│   └── battle_sim.py       # Simulate any two agents head-to-head
├── runs/                   # Training runs (gitignored except logs)
└── content/                # Documentation and YouTube video notes
```

---

## Observation Space (153 dims)

```
Own active        17 dims   HP, 6 boosts (incl accuracy/evasion), status, active, fainted,
                            type_1, type_2, level, base_atk, base_def, base_spa, base_spe*
Own moves         20 dims   4 moves x 5 features (base_power, type, PP, effectiveness, accuracy)
Own bench         42 dims   6 slots x (HP, fainted, type_1, type_2, status, level, base_spe*)
Opponent active   16 dims   HP, 6 boosts (incl accuracy/evasion), status, fainted,
                            type_1, type_2, level, base_atk, base_def, base_spa, base_spe*
Opponent bench    36 dims   6 slots x (HP, fainted, revealed, type_1, type_2, level)
Opp revealed mvs  20 dims   up to 4 seen opponent moves x 5 features (padded)
Trapping           2 dims   own_trapped, own_maybe_trapped (Wrap/Bind flags)
```
`* base_spe is paralysis-adjusted — PAR quarters speed in Gen 1 independently of stage boosts`

Key design decisions:
- **Level included** — gen1randombattle assigns varying levels (58-100) for balance; the agent sees actual level to reason about effective stats
- **Base stats + level** — actual combat stats scale with level; providing both lets the network learn the interaction
- **Paralysis speed correction** — PAR's 0.25x speed penalty applied directly to speed value
- **Move accuracy** — distinguishes Thunder (70%) from Thunderbolt (100%)
- **Own bench status + level** — informed switching decisions
- **Opponent revealed moves** — memory of what the opponent has used
- **Unrevealed opponent slots** = `(-1, 0, 0, 0, 0, 0)` — distinct from fainted `(0, 1, 0, 0, 0, 0)`

---

## Key Gen 1 Mechanics

- **1/256 miss** — nominally 100% moves have a tiny miss chance
- **Freeze is permanent** — one of the strongest status conditions
- **Single Special stat** — no Sp.Atk / Sp.Def split
- **Crit rate from Speed** — fast Pokemon (Tauros, Starmie) crit far more often
- **No held items, no team preview**
- **Sleep Clause** — max 1 opponent Pokemon asleep at a time
