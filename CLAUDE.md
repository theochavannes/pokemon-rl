# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Git Commits

Never include "Co-Authored-By: Claude" or any mention of Claude in commit messages. Commits must always appear as authored solely by the user.

## Multi-Agent Team

This project uses a 5-persona dev team ([ML], [SYS], [REVIEW], [MEDIA], [PM]) plus specialists ([POKE-ENV], [GYM], [RBY], [SB3], [SE], [RL]). See `AGENTS.md` for full roles and decisions log.

**[MEDIA] is ALWAYS ACTIVE — this is a hard requirement, not a suggestion.**

After EVERY significant interaction (bug found, decision made, training results analyzed, approach changed, expert consulted), Claude MUST update the content files before responding to the user. This is not optional. The user should NEVER have to ask for content updates.

Specifically, after each significant moment, append to:
- `content/hooks.md` — video-worthy moments, narrative beats, before/after comparisons
- `content/team_decisions.md` — architectural decisions, what was tried, what failed and why

[MEDIA] uses their expertise to decide what's relevant for a YouTube video about building a Pokemon RL agent. They think like a content creator: what would be interesting to show? What's the story arc? What visuals would work? What's surprising or counterintuitive?

Examples of moments [MEDIA] MUST capture without being asked:
- A bug is found → note the bug, the symptom, the fix, the before/after
- Training results are checked → note the key metrics, what they mean, what changed
- An expert is consulted → note the recommendation and why it matters
- An approach fails → note what was tried, why it failed, what was learned
- A new technique is introduced → note what it is, why it's better, how it works

DO NOT wait for the user to say "update content files." DO IT PROACTIVELY.

**[PM]** always ends their turn with current phase status and explicit next step.

To activate the team in a new session: *"Load the agent team from AGENTS.md. Context: [brief summary]."*

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
  agents/          # Heuristic baseline agents + epsilon wrappers
  train.py         # 4-phase curriculum training loop
  selfplay_train.py
  callbacks.py     # WinRateCallback (logging, milestones, epsilon/temp annealing)
  obs_transfer.py  # Obs-space transfer learning for model compatibility
  run_manager.py   # Run isolation (runs/run_NNN/)
scripts/
  verify_setup.py
  benchmark_heuristic.py
  battle_sim.py    # Simulate any two agents head-to-head
  tournament.py
tests/
  test_core.py     # Unit tests (23 tests, run with pytest)
runs/              # Training runs (gitignored except logs)
PLAN.md            # 6-phase implementation roadmap
```

The Python side uses poke-env to connect to the local Showdown WebSocket server. poke-env exposes a Gymnasium-compatible interface; `src/env/gen1_env.py` wraps it with a 421-dim float32 observation space, shaped rewards (fainted=0.5, hp=0.5, status=0.1, victory=3.0), and action masking. All Pokemon are forced to level 100 in Showdown for training consistency.

Env stack: `Gen1Env → SingleAgentWrapper → SB3Wrapper → Monitor → DummyVecEnv → MaskablePPO`

## Gen 1 Mechanics to Be Aware Of

- 1/256 miss chance on nominally 100%-accurate moves
- Freeze is permanent; Sleep Clause limits opponent to 1 asleep at a time
- Single Special stat (no Sp.Atk/Sp.Def split)
- Crit rate derived from Speed stat
- No held items; no team preview
