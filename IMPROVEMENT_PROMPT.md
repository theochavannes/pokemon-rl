# Pokemon RL Codebase Improvement Prompt

Use this prompt with Claude Code or any capable LLM to perform a comprehensive codebase improvement.

---

## Context

This is a reinforcement learning project training a Pokemon Gen 1 battle agent using MaskablePPO (stable-baselines3) + poke-env + Pokemon Showdown. The agent has a 435-dim observation space, 10 discrete actions (6 switches + 4 moves), and trains through a 4-phase curriculum against increasingly difficult heuristic opponents.

The codebase works but has accumulated technical debt, bugs, inconsistencies, and half-finished features from rapid iteration. It needs a thorough cleanup and improvement pass.

## Instructions

You are 10 specialized agents working together on this codebase. Each agent has a specific role, works independently on their domain, but coordinates through a shared review at the end. **Iterate until every agent signs off.**

### Agent Roles

**Agent 1 — [BUG-HUNTER] Critical Bug Fixer**
Find and fix all bugs. Known bugs to verify and fix:
- `src/train.py`: `shaping_decay_battles` parameter is never passed to WinRateCallback (uses hardcoded default 5000)
- `src/selfplay_train.py`: `env` parameter missing from WinRateCallback call (shaping decay won't work in selfplay)
- `src/train.py` line 265: `reset_num_timesteps` logic — `model is None` is always False here because model is assigned earlier in the loop
- `src/env/gen1_env.py`: Module docstring says 64-dim obs, SB3Wrapper docstring says 64-dim — should be 435
- Run the code mentally end-to-end for each training phase (A, B, C, D) and trace every variable to find logic errors

**Agent 2 — [OBS-AUDITOR] Observation Space Verifier**
- Verify the 435-dim observation count by tracing `embed_battle()` line by line
- Check that every feature is correctly normalized to [-1, 1]
- Verify no NaN/Inf can propagate (what happens if a Pokemon has no moves? no type?)
- Check that bench Pokemon move effectiveness is computed against the right target (own bench vs opponent active, opponent bench vs own active)
- Add runtime validation: `assert np.all(np.isfinite(result))` before returning
- Remove the `level` feature from observation since all Pokemon are now forced to level 100 (it's always 1.0, wasting a dimension per Pokemon)

**Agent 3 — [REWARD-ENGINEER] Reward System Auditor**
- Trace the complete reward pipeline: `calc_reward()` → `reward_computing_helper()` → Monitor → callback
- Verify shaping decay works end-to-end: Gen1Env.shaping_factor is mutated by callback._decay_shaping() through the env wrapper chain (Monitor → SB3Wrapper → SingleAgentWrapper → Gen1Env)
- Check if the wrapper chain `inner = e; while hasattr(inner, "env"): inner = inner.env` actually reaches Gen1Env
- Verify win detection threshold `r >= 0.5` is correct with the current shaping values (fainted=0.5, hp=0.5, status=0.1, victory=3.0). Can a loss ever have r >= 0.5?
- Ensure the ValueError catch (reward=0.0, terminated=True) doesn't corrupt episode statistics in the Monitor wrapper

**Agent 4 — [CURRICULUM-ARCHITECT] Training Pipeline Reviewer**
- Review the 4-phase curriculum flow for correctness
- Verify phase graduation logic: epsilon/temperature must reach end value AND win rate target sustained
- Check that `global_episodes` counter correctly accumulates across phases for shaping decay
- Verify opponent objects are accessible through DummyVecEnv wrapper chain for epsilon/temperature mutation
- Check selfplay frozen model initialization: what happens if best_model.zip doesn't exist yet in Phase D?
- Verify the softmax temperature annealing formula and rate limiting work correctly

**Agent 5 — [CODE-QUALITY] Clean Code Enforcer**
- Remove all dead code, unused imports, commented-out code
- Fix all inconsistencies between docstrings and actual behavior
- Standardize naming conventions (e.g., `epsilon_start` sometimes means temperature)
- Remove duplicate logic
- Ensure all magic numbers have named constants or are passed as parameters
- Fix the `opponent_epsilon` parameter being reused for temperature (confusing API)
- Make `make_env()` less bloated — consider a factory pattern or opponent registry

**Agent 6 — [TEST-WRITER] Test Coverage**
- Write unit tests for:
  - `embed_battle()`: mock a battle object, verify output shape and value ranges
  - `obs_transfer.py`: verify weight expansion works correctly (old dims → new dims)
  - `_expected_damage()`: verify damage calculation with known type matchups
  - `SoftmaxDamagePlayer`: verify temperature affects move distribution correctly
  - `_EpsilonMixin`: verify epsilon blend works, verify swap_model works
  - Reward shaping decay: verify shaping_factor decreases correctly over episodes
- Write integration smoke test: create env, run 1 episode, verify obs shape, reward range, action mask

**Agent 7 — [PERF-OPTIMIZER] Performance Reviewer**
- Profile the training loop bottleneck: is it Showdown WebSocket I/O, observation encoding, or PPO updates?
- Check if `embed_battle()` does unnecessary work (e.g., recomputing type charts every call)
- Verify DummyVecEnv is the right choice (or if SubprocVecEnv + shared memory for epsilon would be faster)
- Check if `_move_features()` is called too many times per step (once per own move + once per bench move + once per opponent move)
- Suggest concrete optimizations with expected speedup

**Agent 8 — [RL-THEORIST] Algorithm & Hyperparameter Reviewer**
- Review PPO hyperparameters: n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, lr=3e-4
- Is [64,64] network large enough for 435-dim input? Should we use [128,128]?
- Review the logit bias approach (+2.0 moves, -2.0 switches) — is the magnitude correct?
- Is ent_coef=0 correct or should we add entropy regularization?
- Review the reward shaping values (fainted=0.5, hp=0.5, status=0.1, victory=3.0) — is the balance right?
- Should gamma be lower for shaped rewards (since intermediate rewards change the effective horizon)?
- Is the 5000-battle shaping decay schedule appropriate for the training length?

**Agent 9 — [DOCS-WRITER] Documentation Updater**
- Update README.md: obs space is 435 dims (not 153), all level 100, correct reward values
- Update CLAUDE.md: architecture section is outdated (still says 64-dim, mentions old env stack)
- Update all docstrings in gen1_env.py to match actual dimensions and behavior
- Update content/team_decisions.md with the latest decisions (softmax opponent, logit bias, 435-dim obs, level 100)
- Verify content/hooks.md narrative is current and compelling

**Agent 10 — [INTEGRATION-TESTER] End-to-End Validator**
- Run `python src/train.py --new-run` mentally through each code path
- Verify the Showdown server modification (level 100 force) is correctly applied
- Verify `selfplay_train.py --run run_NNN` works with the latest model format
- Check that `obs_transfer.py` handles 153→435 dim expansion correctly
- Verify all opponent types can be instantiated without errors
- Check that the `_EpsilonMixin.swap_model()` correctly loads and uses the frozen policy
- Verify `battle_sim.py` works with the current observation space

## Process

1. Each agent works independently on their domain
2. Agents 1-4 (critical) go first
3. Agents 5-8 (quality) go second
4. Agents 9-10 (docs/validation) go last
5. After all agents complete, do a **cross-review**: each agent reviews the changes made by the other agents for conflicts
6. **Iterate** until all agents sign off with no remaining issues
7. Commit with a detailed message listing every change made

## Constraints

- Do NOT add unnecessary abstractions or over-engineer
- Do NOT change the PPO algorithm or training approach (those are deliberate choices)
- Do NOT modify the Showdown submodule further (level 100 change is already made)
- Prefer fixing existing code over rewriting from scratch
- Every change must have a clear reason — no cosmetic-only changes
- Test after each significant change to verify nothing breaks
