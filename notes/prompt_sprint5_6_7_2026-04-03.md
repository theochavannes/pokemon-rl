# Sprint 5-6-7 Execution Prompt (2026-04-03)

First, prevent the laptop from sleeping (keep the existing _prevent_sleep() mechanism active throughout this entire session). Do not allow sleep until I explicitly say otherwise.

Load the agent team from AGENTS.md. Context: We just completed Sprint 3A (mixed league, obs space 928 dims, NN [256,128], SmartHeuristic with status moves). We now need to implement Sprints 5, 6, and 7 sequentially.

## Sprint Execution (repeat for each sprint in order)

Execute each sprint one at a time, in this order:
1. C:\Users\theoc\code\pokemon_rl\notes\sprint5_obs_completeness.md
2. C:\Users\theoc\code\pokemon_rl\notes\sprint6_nn_architecture.md
3. C:\Users\theoc\code\pokemon_rl\notes\sprint7_training_refinement.md

For EACH sprint, follow these steps exactly:

### Step 1: Branch
- Make sure you are on master and have pulled latest
- Create a new branch with a descriptive name (e.g., sprint-5-obs-completeness)

### Step 2: Implement
- Read the sprint spec file thoroughly
- Implement ALL steps described in the spec
- Run tests after each significant change (python -m pytest tests/ -v)
- Fix any test failures before moving on
- Update CLAUDE.md if the obs space dimension changes

### Step 3: Team Review
- Have the full team ([ML], [SE], [REVIEW], [RL], [SMOGON], [RBY], [POKE-ENV], [SB3], [GYM], [TEST]) review the implementation
- [REVIEW] must challenge assumptions and check edge cases
- [TEST] must verify test coverage
- [SMOGON] and [RBY] must verify Gen 1 mechanics are correctly implemented
- Fix any issues found during review before proceeding

### Step 4: Commit, Push, PR
- Commit all changes with a descriptive message (NO Co-Authored-By or Claude mentions)
- Push to the branch
- Create a PR with a clear summary and test plan

### Step 5: CI + CodeRabbit
- Wait for CI (lint + tests) to pass. If they fail, fix and push again
- Wait for CodeRabbit to complete its review (it may take a minute). If it pauses due to too many commits, trigger it with a comment or proceed
- Read ALL CodeRabbit feedback. Fix every issue unless you have a specific technical reason not to (document why you skip any)
- Commit and push fixes

### Step 6: Merge + Sync
- Merge the PR (squash merge is fine)
- Switch to master and pull
- Verify master has the changes

### Step 7: Move to next sprint
- Only after the previous sprint is fully merged, start the next one
- Each sprint may depend on the previous one's obs space changes

## After All 3 Sprints Are Merged

Run the full BC + training pipeline sequentially. Make sure the Showdown server is running first (start it if not):

```bash
cd showdown && node pokemon-showdown start --no-security &
sleep 3
```

Then run each command one at a time, waiting for each to finish before starting the next:

```bash
python scripts/generate_bc_data.py --teacher smart --opponent mixed --n-battles 5000
python scripts/behavioral_cloning.py
python src/train.py --new-run
```

The training run should be left running in the background. Report the first few eval lines so I can see if it's working (Elo, win rate, Vol.Switch%, per-opponent breakdown).

## Important Rules
- Do NOT use obs_transfer.py for mid-vector obs space changes — retrain BC from scratch instead
- All status encodings must respect Gen 1 mechanics (no Electric paralysis immunity, Body Slam CAN paralyze Normal types, Toxic != Poison)
- Update content/team_decisions.md and content/hooks.md after each sprint with [MEDIA] notes
- If a sprint spec step turns out to be infeasible (e.g., poke-env doesn't expose a needed field), skip it and document why
- Keep the training from run_043 or later running if possible — don't kill it unless necessary for Showdown server access
