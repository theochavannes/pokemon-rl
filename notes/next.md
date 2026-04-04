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
2. What the key metrics look like (ExplVar, win rate, BestMv%, per-opponent WR)
3. Whether training is improving, plateauing, or degrading
4. What the current bottleneck is

Write a 1-paragraph status summary to `notes/diagnostic_findings.md` (append,
don't overwrite).

## Phase 1: Diagnose the Bottleneck

Use `/systematic-debugging` methodology. Do NOT propose fixes yet.

Based on the current metrics, answer:

1. **What is the agent failing to do?** (e.g., not improving WR, losing to
   specific opponents, forgetting BC knowledge, value function broken)
2. **What does the data say?** Cite specific numbers from the training log.
   Not "ExplVar is low" but "ExplVar=0.07 at step 90K, was 0.21 at step 60K."
3. **What changed vs the last working state?** (if there was one)
4. **What are the top 3 hypotheses?** For each, state what evidence would
   confirm or disprove it.

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

Write results to `notes/diagnostic_findings.md`.

### HARD GATE: Based on experiment results, commit to ONE approach.
If experiments were inconclusive, say so and propose what to try next session.
Do not build a fix on unverified theory.

## Phase 3: Multi-Agent Design Review

Have the team weigh in on the proposed fix (3-5 sentences per agent max):

- [ML] proposes the fix with RL theory justification
- [REVIEW] challenges: is there a simpler thing to try first?
- [SE] checks: will this break existing infrastructure?
- [SB3] checks: does SB3 support what we're proposing?

Output: ranked list of approaches with [effort, risk, expected impact].

## Phase 4: Implement and Ship

1. Make the code changes
2. Run tests (`pytest tests/ -q`)
3. Follow the git workflow in CLAUDE.md:
   branch -> commit -> push -> PR -> CodeRabbit -> address feedback -> merge -> sync
4. Start a training run if the changes warrant it (run in background)

## Phase 5: Content Updates

[MEDIA] updates `content/hooks.md` and `content/team_decisions.md` with:
- What was the bottleneck?
- What experiment revealed the answer?
- What was the fix?
- What's the narrative beat for the video?

## Rules

- Use `C:/Users/theoc/miniconda3/envs/pokemon_rl/python.exe` for all Python
- Start Showdown before any battles: `cd showdown && node pokemon-showdown start --no-security`
- Use `PYTHONUNBUFFERED=1` prefix for any Python experiment scripts
- No VecNormalize, no two-tower architecture, no Co-Authored-By
- Commit after each logical step, not at the end
- If any test fails, fix it before proceeding
- Every task that changes tracked files ends with a merged PR on master
