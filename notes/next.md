# Next Iteration

Load the agent team from AGENTS.md. Then execute the loop below.

## Phase 0: Read Current State

Read these files to understand where we are:

- `content/team_decisions.md` — full history of what was tried, what worked, what failed
- `src/train.py` — current training config and pipeline
- `src/env/gen1_env.py` — observation space, reward function
- `src/callbacks.py` — metrics and monitoring

Find the most recent training run (`runs/run_NNN/`). Read its `training_log.md`
and `run_info.json`. Identify:

1. What config was used (hyperparams, architecture, curriculum)
2. Key metrics — check ALL of these, not just win rate:
   - **Win rate** (overall + per-opponent breakdown)
   - **ExplVar** (value function quality — is it positive? trending up?)
   - **Temperature / epsilon** (opponent difficulty — is it annealing or stuck?)
   - **BestMv%** (BC knowledge retention — eroding or stable?)
   - **Vol.Sw%** (is the agent learning new behaviors?)
   - **Elo** (directional skill trend)
3. Whether training is improving, plateauing, or degrading
4. What the current bottleneck is

**Compare with the previous run(s).** At the same step count, is this run
better, worse, or identical on each metric? If a change was made between
runs, did it have the intended effect?

Write a 1-paragraph status summary to `notes/diagnostic_findings.md` (append,
don't overwrite).

## Phase 1: Diagnose the Bottleneck

Use `/systematic-debugging` methodology. Do NOT propose fixes yet.

Based on the current metrics, answer:

1. **What is the agent failing to do?** (e.g., not improving WR, losing to
   specific opponents, forgetting BC knowledge, value function broken)
2. **What does the data say?** Cite specific numbers from the training log.
   Not "ExplVar is low" but "ExplVar=0.07 at step 90K, was 0.21 at step 60K."
   Do this for EVERY key metric, not just the one that looks worst.
3. **What changed vs the last working state?** (if there was one)
4. **What are the top 3 hypotheses?** For each, state:
   - What evidence supports it
   - What evidence would disprove it
   - A concrete experiment (< 50K steps or a benchmark) to test it

Save findings to `notes/diagnostic_findings.md`.

### HARD GATE: Do not proceed until you have at least one hypothesis with a
concrete experiment that can confirm/disprove it in under 30 minutes.

## Phase 2: Quick Experiments

Design the smallest experiment that tests the top hypothesis. Rules:

- Max 50K steps per experiment (use 1-2 envs for speed if needed)
- Use `PYTHONUNBUFFERED=1` so output isn't trapped by buffering
- Measure the specific metric that confirms/disproves the hypothesis
- Run max 3 experiments. If all 3 fail to explain the problem, reassess
  hypotheses before running more.
- Benchmarks (scripts/benchmark_league.py) are fast and informative — use
  them when appropriate.

Write results to `notes/diagnostic_findings.md`.

### HARD GATE: Based on experiment results, commit to ONE approach.
If experiments were inconclusive, say so and propose what to try next session.
Do not build a fix on unverified theory.

## Phase 2.5: Validate the Fix Mechanism

Before implementing the full fix, answer these questions:

1. **What is the specific mechanism by which this fix improves the bottleneck
   metric?** Not "larger buffer helps" but "ExplVar is 0.12 because X, and
   this change addresses X by doing Y, which should raise it to Z."
2. **What would a mini-experiment (50K steps) show if the fix works?** Define
   the expected metric change. If you can't predict what success looks like,
   you don't understand the mechanism well enough.
3. **Run the mini-experiment.** If the target metric doesn't move in the
   expected direction within 50K steps, the fix is wrong — go back to Phase 1.

### HARD GATE: The mini-experiment must show measurable improvement in the
target metric before proceeding to full implementation. "It needs more steps"
is not an acceptable answer — if the mechanism works, early signal should be
visible.

## Phase 3: Multi-Agent Design Review

Have the team weigh in on the proposed fix (3-5 sentences per agent max):

- [ML] proposes the fix with RL theory justification. Must explain the
  MECHANISM — what specifically changes in the math/gradient/optimization.
- [REVIEW] challenges: What assumption is this fix built on? What's the
  simplest experiment that would disprove it? Has a similar fix failed before
  in this project? (Check team_decisions.md for past failures.)
- [SE] checks: will this break existing infrastructure? Resume logic?
  Checkpoint compatibility?
- [SB3] checks: does SB3 support what we're proposing? Any gotchas?

Output: ranked list of approaches with [effort, risk, expected impact].

[REVIEW] must cite at least one past failure from team_decisions.md and explain
why this fix avoids the same mistake. If they can't, that's a red flag.

## Phase 4: Implement and Ship

1. Make the code changes
2. Run tests (`pytest tests/ -q`)
3. Follow the git workflow in CLAUDE.md:
   branch -> commit -> push -> PR -> CodeRabbit -> address feedback -> merge -> sync
4. If starting a training run:
   - Define **kill criteria** before starting: "If metric X hasn't improved
     by step Y, abort." Be specific.
   - Do NOT pipe long-running training through head/tail — redirect to a log
     file: `python src/train.py > runs/run_NNN_console.log 2>&1`
   - Check early results (first 50K steps) against the mini-experiment from
     Phase 2.5. If they diverge, investigate immediately.

## Phase 5: Content Updates

[MEDIA] updates `content/hooks.md` and `content/team_decisions.md` with:
- What was the bottleneck?
- What experiment revealed the answer?
- What was the fix?
- What's the narrative beat for the video?
- If the fix FAILED: document why — failed approaches are valuable content.

## Rules

- Use `C:/Users/theoc/miniconda3/envs/pokemon_rl/python.exe` for all Python
- Start Showdown before any battles: `cd showdown && node pokemon-showdown start --no-security`
- Use `PYTHONUNBUFFERED=1` prefix for any Python experiment scripts
- No VecNormalize, no two-tower architecture, no Co-Authored-By
- Commit after each logical step, not at the end
- If any test fails, fix it before proceeding
- Every task that changes tracked files ends with a merged PR on master

## Anti-Patterns (from project history)

These have failed repeatedly. Do not repeat them without new evidence:

- **Config-only fixes for structural problems.** Tuning hyperparams (vf_coef,
  n_steps, batch_size) has never broken through a plateau in this project.
  If the bottleneck is structural (e.g., value function can't learn from
  stochastic matchups), config changes won't help.
- **Building elaborate plans on unverified hypotheses.** Every plan built
  without a diagnostic experiment first has failed. 100% failure rate.
- **"It needs more steps."** If a fix works, early signal is visible within
  50K steps. If ExplVar/WR/BestMv% hasn't moved by 50K, the fix is wrong.
- **Fixing symptoms instead of root causes.** "ExplVar is low" is a symptom.
  "The value function overfits each rollout because n_epochs=10" is a root
  cause. Fixes must target root causes.
