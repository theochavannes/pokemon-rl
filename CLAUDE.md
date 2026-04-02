# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Git Commits

Never include "Co-Authored-By: Claude" or any mention of Claude in commit messages. Commits must always appear as authored solely by the user.

## Multi-Agent Team

This project uses a 5-persona dev team ([ML], [SYS], [REVIEW], [MEDIA], [PM]). See `AGENTS.md` for roles, invocation guide, and decisions log. To activate: *"Load the agent team from AGENTS.md. Context: [brief summary]."*

## Project Overview

Reinforcement learning agent for competitive Gen 1 (RBY) Pokemon battles using Pokemon Showdown as the simulator and PPO (MaskablePPO) as the training algorithm.

## Tech Stack

- **Game server**: Pokemon Showdown (Node.js/TypeScript) in `showdown/`
- **Python env**: `C:/Users/theoc/miniconda3/envs/pokemon_rl/python.exe` (Python 3.11)
- **Key Python libs**: poke-env 0.13.0, stable-baselines3 + sb3-contrib (MaskablePPO), PyTorch CUDA 12.0
- **Battle format**: `gen1randombattle` throughout

## Development Workflow

On a fresh clone, initialize the submodule first:

```bash
git submodule update --init
cd showdown && npm install
```

Always start the Showdown server before running any Python/poke-env code:

```bash
cd showdown
node pokemon-showdown start --no-security
# WebSocket available at ws://localhost:8000/showdown/websocket
```

## Showdown Server Commands

```bash
# Build (compile TypeScript → dist/)
cd showdown && node build

# Lint
npm run lint
npm run fix        # auto-fix

# Test (runs lint first)
npm test
npx mocha -g "text"   # single test by name
```

## Architecture

```
showdown/          # Pokemon Showdown server (separate git repo, gitignored)
src/
  env/             # Gymnasium environment wrapper (gen1_env.py)
  agents/          # Heuristic baseline agents
  train.py         # PPO training loop
  selfplay_train.py
  callbacks.py     # WinRateCallback
scripts/
  verify_setup.py
  benchmark_heuristic.py
models/            # Checkpoints (gitignored)
logs/              # TensorBoard logs (gitignored)
PLAN.md            # 6-phase implementation roadmap
```

The Python side uses poke-env to connect to the local Showdown WebSocket server. poke-env exposes a Gymnasium-compatible interface; `src/env/gen1_env.py` wraps it with a ~58-dim float32 observation space and ±1 win/loss rewards with action masking.

## Gen 1 Mechanics to Be Aware Of

- 1/256 miss chance on nominally 100%-accurate moves
- Freeze is permanent; Sleep Clause limits opponent to 1 asleep at a time
- Single Special stat (no Sp.Atk/Sp.Def split)
- Crit rate derived from Speed stat
- No held items; no team preview
