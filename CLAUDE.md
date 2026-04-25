# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Git Commits

Never include "Co-Authored-By: Claude" or any mention of Claude in commit messages. Commits must always appear as authored solely by the user.

## Autonomous sessions

At the start of any autonomous multi-step session, run `echo "<task description>" > .claude/task.md`. Delete it with `rm -f .claude/task.md` only when all tasks are complete. The `enforce-stop.sh` hook blocks session exit while this file exists.

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
- **Python env**: Python 3.11 (conda recommended: `conda create -n pokemon_rl python=3.11`)
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

The Python side uses poke-env to connect to the local Showdown WebSocket server. poke-env exposes a Gymnasium-compatible interface; `src/env/gen1_env.py` wraps it with a 1739-dim float32 observation space, shaped rewards (fainted=0.5, victory=1.0), and action masking. All Pokemon are forced to level 100 in Showdown for training consistency.

## Observation Space Changes — IMPORTANT

`obs_transfer.py` zero-pads new dimensions assuming they are APPENDED at the end of the vector. If you change the obs space by inserting features in the middle (e.g., expanding per-move features from 5→10), existing checkpoints become INCOMPATIBLE — the feature positions shift and the model's learned weights map to wrong inputs. In that case, **do NOT use obs_transfer** — retrain BC from scratch with the new obs shape instead. Only use obs_transfer when strictly appending new dims at the end.

Env stack: `Gen1Env → SingleAgentWrapper → SB3Wrapper → Monitor → DummyVecEnv → MaskablePPO`

## Complex Decisions

When facing complex architectural decisions (reward shaping, observation space changes, training curriculum design), break the problem into steps, consider at least 2 alternatives, and compare trade-offs before recommending an approach.

## Git Workflow — Ship Every Task

When work is complete (code changes pass tests), ALWAYS follow this sequence without being asked:

1. **Branch**: `git checkout -b <descriptive-branch-name>`
2. **Commit**: Stage relevant files with focused commits. Use `git commit -m "$(cat <<'EOF' ... EOF)"` for multi-line messages. Never use `--no-verify`.
3. **Push**: `git push -u origin <branch>`
4. **PR**: `gh pr create --title "..." --body "$(cat <<'EOF' ... EOF)"` — include a Summary section and Test plan. No AI attribution.
5. **Wait for CodeRabbit**: `gh pr checks <N>` until CodeRabbit shows pass. Read inline comments with `gh api repos/<owner>/<repo>/pulls/<N>/comments`.
6. **Address feedback**: Fix all valid issues from one CodeRabbit review round in a **single commit** titled `Address CodeRabbit feedback on PR #<N>` before pushing. This cuts re-review quota burn and matches open-source convention. Ignore pure nitpicks on docs/notes files.
7. **Merge**: `gh pr merge <N> --squash --delete-branch`. Squash collapses the PR's commits into one clean commit on master. Full per-commit history remains visible in the closed PR on GitHub. Granular local commits become free — commit as often as you want while working, it all collapses at merge.
8. **Sync**: `git checkout master && git pull`

This is not optional. Every task that changes tracked files ends with a merged PR on master.

**Note on content-only PRs**: When updating `content/hooks.md` or `content/team_decisions.md` to document a decision made in a code PR, prefer bundling the content update into that same PR rather than opening a follow-up PR. This lands the doc with the code it describes (better traceability) and saves a CodeRabbit review cycle. Standalone content-only PRs are fine when the content isn't tied to a specific code change.

**CodeRabbit config**: `.coderabbit.yaml` at repo root excludes `content/`, `notes/`, `showdown/`, `runs/`, `data/`, and root-level planning `*.md` from review to preserve quota for code.

## Gen 1 Mechanics to Be Aware Of

- 1/256 miss chance on nominally 100%-accurate moves
- Freeze is permanent; Sleep Clause limits opponent to 1 asleep at a time
- Single Special stat (no Sp.Atk/Sp.Def split)
- Crit rate derived from Speed stat
- No held items; no team preview
