# Pokemon RL

Training a reinforcement learning agent to play competitive Gen 1 (RBY) Pokemon using [Pokemon Showdown](https://github.com/smogon/pokemon-showdown) as the battle simulator.

![Python 3.11](https://img.shields.io/badge/python-3.11-blue) ![Tests](https://github.com/theochavannes/pokemon-rl/actions/workflows/test.yml/badge.svg)

---

## Results

After 57 training runs with a 4-phase curriculum against mixed opponents:

| Metric | Value |
|---|---|
| Peak win rate vs OU-only heuristics | **65%** (run_057, step 355K) |
| vs MaxDamage | ~68% |
| vs TypeMatchup | ~61% |
| vs Random | ~91% |
| Training runs completed | 57 |
| Observation space | 1,739 features |
| Species pool | 39 OU-only Gen 1 species |

---

## Quick Start

```bash
git clone --recurse-submodules https://github.com/theochavannes/pokemon-rl.git
cd pokemon-rl
pip install -e .
cd showdown && npm install && node pokemon-showdown start --no-security &
python scripts/evaluate.py --model models/best_model_v1.zip
```

The best trained checkpoint (`models/best_model_v1.zip`, 11.6 MB) is committed to the repo.

---

## Architecture

```
Pokemon Showdown (Node.js, local WebSocket)
         |
poke-env 0.13 (Gymnasium interface)
         |
Gen1Env (src/env/gen1_env.py)
  - 1739-dim float32 observation
  - action masking (invalid moves/switches blocked)
  - shaped reward (fainted=0.5, victory=1.0, decays over 5K battles)
         |
MaskablePPO (sb3-contrib) — 256x128 MLP policy
         |
4-env DummyVecEnv mixed league
  env 0: SelfPlay (frozen snapshot)
  env 1: MaxDamagePlayer
  env 2: TypeMatchupPlayer
  env 3: SoftmaxDamagePlayer (temperature-annealed)
```

---

## Key Technical Contributions

- **Observation-space transfer learning** (`src/obs_transfer.py`) — zero-pads prior checkpoints to a larger obs dimension, enabling incremental feature additions without full retraining.
- **KO-probability features** — per-move `prob_ko`, `expected_dmg_fraction`, and `recharge_trap` computed using the full Gen 1 damage formula (crit rate from Speed, type chart, base power, HP fractions).
- **Mean+min phase gate** — curriculum advances only when `mean(heuristic WRs) >= 0.65` AND `min >= 0.50` for 2 consecutive evals, preventing premature phase transitions driven by a single easy opponent.
- **OU-only species pool** — filtered `gen1randombattle` to 39 OU-eligible Gen 1 species, removing unevolved and non-competitive Pokemon that inflate win rates without teaching transferable play.

---

## Observation Space (1,739 dims)

All Pokemon are forced to level 100 in Showdown for training consistency.

```
Own active          28 dims   HP, 6 boosts (+accuracy/evasion), status, active,
                              fainted, type x2, 6 base stats*, 3 KO features/move
Own moves           28 dims   4 moves x 7 features (power, type, PP, effectiveness,
                              accuracy, prob_ko, expected_dmg_fraction)
Own bench          240 dims   6 slots x 40 (HP, fainted, types, status, base stats,
                              4 moves x 7 features w/ KO features)
Opponent active     27 dims   HP, 6 boosts, status, fainted, type x2, 6 base stats*
Opponent bench     240 dims   6 slots x 40 (HP, fainted, revealed, types, base stats,
                              up to 4 revealed moves x 7 features)
Role features      168 dims   Competitive role tags: sweeper, wall, status-setter,
                              revenge-killer, pivot (own bench + opponent bench)
Opp revealed mvs    28 dims   up to 4 seen opponent active moves x 7 features
Trapping            2 dims    own_trapped, own_maybe_trapped (Wrap/Bind flags)
```

`* base_spe is paralysis-adjusted — PAR quarters speed in Gen 1 independently of stage boosts`

Key design decisions:
- **Full bench info** — all 4 moves per bench Pokemon with effectiveness vs current opponent enables informed switching
- **KO features** — `prob_ko` and `expected_dmg_fraction` expose the kill-or-not decision explicitly in obs
- **Role tags** — sweeper/wall/pivot labels extracted from base stat profiles (see `src/env/gen1_env.py`)
- **Opponent memory** — unrevealed slots start at -1.0, distinct from fainted (0.0)

---

## Training Curriculum

The agent trains against all heuristic opponents simultaneously from the start (mixed league), with difficulty annealed via temperature and epsilon parameters. Phase advances are gated on sustained win-rate criteria.

| Phase | Target | Description |
|---|---|---|
| A | Critic warmup | 100 rollouts with frozen actor, no phase gate |
| B | WR >= 0.50 | Full training begins against mixed opponents |
| C | mean >= 0.65, min >= 0.50 | Tighten temperature annealing |
| D | Plateau | Self-play pool activated |

**Mixed league (4 envs):**
- `MaxDamagePlayer` — always picks highest base_power × type_effectiveness
- `TypeMatchupPlayer` — best typed move, switches out of bad matchups
- `SoftmaxDamagePlayer` — damage-proportional sampling, temperature annealed 2.0 → 0.1
- `FrozenPolicyPlayer` — frozen snapshot of agent itself (self-play)

```bash
python src/train.py --new-run    # start a new run
python src/train.py              # resume latest run
```

---

## What I'd Do Next

The agent is stuck in a "never switch" local optimum (~1% voluntary switch rate across all 57 runs). The PPO gradient from terminal rewards alone is too weak to bootstrap switching behaviour.

- **Behavioural cloning pretraining** — generate expert demonstrations from `TypeMatchupPlayer`, pretrain a BC checkpoint, then fine-tune with PPO. This directly seeds the switch-behaviour prior.
- **Richer reward signal for switching** — shaped reward for switching into a type advantage, rather than relying on the delayed terminal signal.
- **Larger network** — current 256×128 MLP may lack capacity to represent the full Gen 1 matchup graph. A 512×256 or attention-based architecture could help.

---

## Project Structure

```
pokemon_rl/
├── showdown/               # Pokemon Showdown server (git submodule)
├── src/
│   ├── env/gen1_env.py     # Gymnasium environment — 1739-dim obs, shaped reward
│   ├── agents/
│   │   ├── heuristic_agent.py   # MaxDamage, TypeMatchup, Softmax + epsilon wrappers
│   │   └── policy_player.py     # FrozenPolicyPlayer (self-play opponent)
│   ├── train.py            # 4-phase curriculum training loop
│   ├── selfplay_train.py   # Self-play with frozen snapshots
│   ├── callbacks.py        # WinRateCallback — logging, milestones, annealing
│   ├── obs_transfer.py     # Obs-space transfer learning for checkpoint compatibility
│   └── run_manager.py      # Run isolation (runs/run_NNN/)
├── scripts/
│   ├── evaluate.py         # Benchmark a checkpoint vs heuristic opponents
│   ├── verify_setup.py     # End-to-end smoke test
│   ├── benchmark_heuristic.py
│   └── battle_sim.py       # Simulate any two agents head-to-head
├── models/
│   └── best_model_v1.zip   # Best checkpoint (65% WR, obs_dim=1739)
├── tests/
│   └── test_core.py        # Unit tests (pytest)
└── PLAN.md                 # 6-phase implementation roadmap
```

---

## Gen 1 Mechanics

- **1/256 miss** — nominally 100% moves have a tiny miss chance
- **Freeze is permanent** — one of the strongest status conditions
- **Single Special stat** — no Sp.Atk / Sp.Def split; one stat covers both
- **Crit rate from Speed** — fast Pokemon (Tauros, Starmie) crit far more often
- **No held items, no team preview** — pure positioning and read-based play
