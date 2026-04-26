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

# Hard Rules

These apply to every project. Read before acting.

- **No AI attribution.** Never add `Co-Authored-By: Claude` or any mention of AI to commits or PR bodies.
- **Never `--no-verify`.** If a pre-commit hook fails, fix the underlying issue. Same for CI.
- **Never commit directly to `master`/`main`.** All changes ship via feature branch → PR → squash-merge.
- **Read files before acting.** If a path or intent is ambiguous, read more context to resolve it. If still unclear, convene subagents to deliberate — don't block on user input.
- **Two alternatives minimum.** For any non-trivial design choice, state at least one alternative and why it's worse before recommending. Skip only for mechanical edits.
- **Diagnose before fixing.** When behaviour looks wrong, find *why* before changing code — surface fixes often mask the real issue.
- **Task marker.** At the start of any autonomous multi-step session, run `echo "<task description>" > .claude/task.md`. Delete it with `rm -f .claude/task.md` only when all tasks are complete. The `enforce-stop.sh` hook blocks session exit while the file exists.
- **Skills before generic agents.** Before spawning a generic agent for any named task (literature review, code review, debugging, etc.), check the skill table in this file's CLAUDE.md. If a skill covers it, use it. Generic agents are for tasks with no matching skill.
- **Workflow is a checklist, not advisory.** For non-trivial tasks, each phase in the documented workflow is mandatory in order. Trivial-task exceptions documented in the workflow (e.g. skip to Phase 3) may be applied explicitly. Skipping or deferring any other step requires explicit user sign-off with a stated reason. Do not silently defer steps to "later" and do not add steps that aren't in the documented workflow without flagging them as improvised.
- **Session checkpoints.** For autonomous work spanning more than ~30 minutes, write `.claude/checkpoint.md` at each phase boundary: phase name, what's done, what's blocked, what's next. Recovery from a crash must start from this file, not from manual reconstruction.
- **Parallel session ownership.** Before running multiple concurrent sessions, define non-overlapping file ownership. Sessions must not write to each other's files. Prefer a single session unless work is provably independent; if parallelising, shard explicitly and document what each session owns.

# Base Claude Guidelines

Source: [forrestchang/andrej-karpathy-skills](https://github.com/forrestchang/andrej-karpathy-skills)
Use: Copy or `cat` into any project's CLAUDE.md under a "Guidelines" section.

---

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```text
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

# Development Workflow

Source: [obra/superpowers](https://github.com/obra/superpowers)
Use: Include in CLAUDE.md for any non-trivial project. Shapes how Claude approaches multi-step work.

---

## The 6-Phase Workflow

### Phase 1 — Clarify
Ask clarifying questions before writing any code.
- What is the desired outcome?
- What are the constraints?
- What does "done" look like?

### Phase 2 — Plan
Break into small, verifiable tasks (aim for 2–5 minutes each).
```text
Task 1: [description] → verify: [how to confirm it's done]
Task 2: [description] → verify: [how to confirm it's done]
```
Present the plan and get sign-off before implementing.

### Phase 3 — Implement
- Work task by task
- Keep changes small and focused
- Involve subagents for parallel independent work when possible

### Phase 4 — Test (TDD)
- RED: Write a failing test first
- GREEN: Write minimum code to pass
- REFACTOR: Clean up without breaking

### Phase 5 — Review
Check implementation against original plan:
- Did we solve what was asked?
- Any scope creep?
- Any unnecessary complexity?

### Phase 6 — Complete
- Verify all tests pass
- Handle merge/commit decisions explicitly

## When to Use This

- Any task that takes more than 15 minutes
- Any task that touches more than 2 files
- Any task where requirements are ambiguous

For trivial tasks (single-file edits, clear requirements), skip to Phase 3.

## Phases are mandatory

The 6 phases are a **required sequence**, not a menu. Do not reorder, skip, or defer phases without explicit user approval and a stated reason. Improvised steps outside this sequence must be flagged as such before executing — not executed and reported after.

<!-- git-workflow updated by incorporate.sh -->
# Git & PR Workflow

Every task that changes tracked files ends with a **merged PR on master**. Not optional.

**Creating a PR is not done. Merging it is done.** After opening a PR, you must complete the full loop below before considering the task complete or moving on to any other work.

1. **Branch** — `git checkout -b <descriptive-name>` (e.g., `add-validation`, not `fix-1`)
2. **Commit** — focused commits, imperative subject under ~70 chars. Multi-line via heredoc:
   ```bash
   git commit -m "$(cat <<'EOF'
   Subject line

   Body if needed.
   EOF
   )"
   ```
   Prefer new commits over `--amend` for anything already pushed.
3. **Push** — `git push -u origin <branch>`
4. **PR** — `gh pr create --title "..." --body "$(cat <<'EOF' ... EOF)"` with `## Summary` (bullets) and `## Test plan` (checklist). No AI attribution.
5. **Wait for CodeRabbit** — poll `gh pr checks <N>` until the CodeRabbit review check completes. Do not proceed until it finishes.
6. **Read every CodeRabbit comment** — `gh api repos/<owner>/<repo>/pulls/<N>/comments` and `gh api repos/<owner>/<repo>/pulls/<N>/reviews`. Read all of them; do not skim.
7. **Address feedback in one commit** — title `Address CodeRabbit feedback on PR #<N>`. For each finding: fix it if valid, reply dismissing it if wrong, explicitly skip pure style nitpicks on docs/notes. Batch all fixes into one commit.
8. **Merge** — `gh pr merge <N> --squash --delete-branch`
9. **Sync** — `git checkout master && git pull`

**Content-only updates**: fold doc/decision-log edits into the code PR they describe. Cleaner history, saves a CodeRabbit cycle.

**Decision log**: record architectural choices in `decisions.md`. Only put settled design decisions there — not plans, timelines, future directions, or to-do items (those go in `notes/`). Don't re-litigate decisions already there — amend the entry if the call changes.

## Anti-patterns

- Stale branches — merge or delete
- Committing directly to master
- Premature abstraction — three similar lines beats a speculative helper
- Starting implementation before the plan is approved

# Skills

Invoke via `/<skill-name>` in Claude Code or the Skill tool.

**Rule: prefer specialized skills over generic agents.** If a skill in this table covers the task, use it. Do not improvise a generic agent when a purpose-built skill exists.


| Skill | When to invoke |
|-------|---------------|
| `commit` | Before every git commit |
| `code-reviewer` | After finishing an implementation |
| `classifying-review-findings` | Alongside `code-reviewer` — triage severity of every finding |
| `simplify` | After implementing — checks for unnecessary complexity |
| `concise-planning` | Before starting any multi-step task |
| `using-superpowers` | At session start for autonomous/long-running work |
| `tdd-workflow` | When doing test-driven development |
| `debugging-toolkit-smart-debug` | When stuck on a bug |
| `reviewing-claude-config` | When modifying CLAUDE.md, settings.json, skill files, or agents |
| `retrospecting` | At end of any significant session — offer to run before stopping |

# Development Workflow

Source: [obra/superpowers](https://github.com/obra/superpowers)
Use: Include in CLAUDE.md for any non-trivial project. Shapes how Claude approaches multi-step work.

---

## The 6-Phase Workflow

### Phase 1 — Clarify
Ask clarifying questions before writing any code.
- What is the desired outcome?
- What are the constraints?
- What does "done" look like?

### Phase 2 — Plan
Break into small, verifiable tasks (aim for 2–5 minutes each).
```text
Task 1: [description] → verify: [how to confirm it's done]
Task 2: [description] → verify: [how to confirm it's done]
```
Present the plan and get sign-off before implementing.

### Phase 3 — Implement
- Work task by task
- Keep changes small and focused
- Involve subagents for parallel independent work when possible

### Phase 4 — Test (TDD)
- RED: Write a failing test first
- GREEN: Write minimum code to pass
- REFACTOR: Clean up without breaking

### Phase 5 — Review
Check implementation against original plan:
- Did we solve what was asked?
- Any scope creep?
- Any unnecessary complexity?

### Phase 6 — Complete
- Verify all tests pass
- Handle merge/commit decisions explicitly

## When to Use This

- Any task that takes more than 15 minutes
- Any task that touches more than 2 files
- Any task where requirements are ambiguous

For trivial tasks (single-file edits, clear requirements), skip to Phase 3.

## Phases are mandatory

The 6 phases are a **required sequence**, not a menu. Do not reorder, skip, or defer phases without explicit user approval and a stated reason. Improvised steps outside this sequence must be flagged as such before executing — not executed and reported after.

<!-- git-workflow updated by incorporate.sh -->
# Git & PR Workflow

Every task that changes tracked files ends with a **merged PR on master**. Not optional.

**Creating a PR is not done. Merging it is done.** After opening a PR, you must complete the full loop below before considering the task complete or moving on to any other work.

1. **Branch** — `git checkout -b <descriptive-name>` (e.g., `add-validation`, not `fix-1`)
2. **Commit** — focused commits, imperative subject under ~70 chars. Multi-line via heredoc:
   ```bash
   git commit -m "$(cat <<'EOF'
   Subject line

   Body if needed.
   EOF
   )"
   ```
   Prefer new commits over `--amend` for anything already pushed.
3. **Push** — `git push -u origin <branch>`
4. **PR** — `gh pr create --title "..." --body "$(cat <<'EOF' ... EOF)"` with `## Summary` (bullets) and `## Test plan` (checklist). No AI attribution.
5. **Wait for CodeRabbit** — poll `gh pr checks <N>` until the CodeRabbit review check completes. If CodeRabbit is absent or quota-limited, run `bash .claude/hooks/gemini-pr-review.sh <N>` as fallback and treat its output as the review. Do not proceed until one review (CodeRabbit or Gemini) is complete.
6. **Read every review comment** — For CodeRabbit: `gh api repos/<owner>/<repo>/pulls/<N>/comments` and `gh api repos/<owner>/<repo>/pulls/<N>/reviews`. For Gemini fallback: findings are printed to stdout by the script — read them there. Do not skim either source.
7. **Address feedback in one commit** — title `Address review feedback on PR #<N>`. For each finding: fix it if valid, reply dismissing it if wrong, explicitly skip pure style nitpicks on docs/notes. Batch all fixes into one commit.
8. **Merge** — `gh pr merge <N> --squash --delete-branch`
9. **Sync** — `git checkout master && git pull`

**Content-only updates**: fold doc/decision-log edits into the code PR they describe. Cleaner history, saves a CodeRabbit cycle.

**Decision log**: record architectural choices in `decisions.md`. Only put settled design decisions there — not plans, timelines, future directions, or to-do items (those go in `notes/`). Don't re-litigate decisions already there — amend the entry if the call changes.

## Anti-patterns

- Stale branches — merge or delete
- Committing directly to master
- Premature abstraction — three similar lines beats a speculative helper
- Starting implementation before the plan is approved

# Development Workflow

Source: [obra/superpowers](https://github.com/obra/superpowers)
Use: Include in CLAUDE.md for any non-trivial project. Shapes how Claude approaches multi-step work.

---

## The 6-Phase Workflow

### Phase 1 — Clarify
Ask clarifying questions before writing any code.
- What is the desired outcome?
- What are the constraints?
- What does "done" look like?

### Phase 2 — Plan
Break into small, verifiable tasks (aim for 2–5 minutes each).
```text
Task 1: [description] → verify: [how to confirm it's done]
Task 2: [description] → verify: [how to confirm it's done]
```
Present the plan and get sign-off before implementing.

### Phase 3 — Implement
- Work task by task
- Keep changes small and focused
- Involve subagents for parallel independent work when possible

### Phase 4 — Test (TDD)
- RED: Write a failing test first
- GREEN: Write minimum code to pass
- REFACTOR: Clean up without breaking

### Phase 5 — Review
Check implementation against original plan:
- Did we solve what was asked?
- Any scope creep?
- Any unnecessary complexity?

### Phase 6 — Complete
- Verify all tests pass
- Handle merge/commit decisions explicitly

## When to Use This

- Any task that takes more than 15 minutes
- Any task that touches more than 2 files
- Any task where requirements are ambiguous

For trivial tasks (single-file edits, clear requirements), skip to Phase 3.

## Phases are mandatory

The 6 phases are a **required sequence**, not a menu. Do not reorder, skip, or defer phases without explicit user approval and a stated reason. Improvised steps outside this sequence must be flagged as such before executing — not executed and reported after.

# Development Workflow

Source: [obra/superpowers](https://github.com/obra/superpowers)
Use: Include in CLAUDE.md for any non-trivial project. Shapes how Claude approaches multi-step work.

---

## The 6-Phase Workflow

### Phase 1 — Clarify
Ask clarifying questions before writing any code.
- What is the desired outcome?
- What are the constraints?
- What does "done" look like?

### Phase 2 — Plan
Break into small, verifiable tasks (aim for 2–5 minutes each).
```text
Task 1: [description] → verify: [how to confirm it's done]
Task 2: [description] → verify: [how to confirm it's done]
```
Present the plan and get sign-off before implementing.

### Phase 3 — Implement
- Work task by task
- Keep changes small and focused
- Involve subagents for parallel independent work when possible

### Phase 4 — Test (TDD)
- RED: Write a failing test first
- GREEN: Write minimum code to pass
- REFACTOR: Clean up without breaking

### Phase 5 — Review
Check implementation against original plan:
- Did we solve what was asked?
- Any scope creep?
- Any unnecessary complexity?

### Phase 6 — Complete
- Verify all tests pass
- Handle merge/commit decisions explicitly

## When to Use This

- Any task that takes more than 15 minutes
- Any task that touches more than 2 files
- Any task where requirements are ambiguous

For trivial tasks (single-file edits, clear requirements), skip to Phase 3.

## Phases are mandatory

The 6 phases are a **required sequence**, not a menu. Do not reorder, skip, or defer phases without explicit user approval and a stated reason. Improvised steps outside this sequence must be flagged as such before executing — not executed and reported after.
