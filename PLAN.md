# Pokemon RBY RL Agent — Implementation Roadmap

## Project Overview
Build a Reinforcement Learning agent to play competitive Gen 1 (RBY) Pokemon on Pokemon Showdown.

**Stack:**
- Simulator: Pokemon Showdown (Node.js, local server) — in `showdown/`
- Python 3.11, poke-env 0.13.0, stable-baselines3 + sb3-contrib (MaskablePPO), PyTorch CUDA
- Battle format: `gen1randombattle` (OU-only species pool, random teams)
- RL algorithm: PPO with action masking (`MaskablePPO` from sb3-contrib)

---

## Phase 1 — Infrastructure ✅ COMPLETE

- [x] Pokemon Showdown server (git submodule, local WebSocket)
- [x] Python environment with poke-env, stable-baselines3, PyTorch CUDA
- [x] `scripts/verify_setup.py` — end-to-end smoke test
- [x] `pyproject.toml`, Ruff, pre-commit hooks, CI, CodeRabbit

---

## Phase 2 — Gen 1 Gymnasium Environment ✅ COMPLETE

- [x] `src/env/gen1_env.py` — 1739-dim observation space, action masking, shaped rewards
- [x] Reward: fainted=0.5, victory=1.0, shaping decay over 5000 battles
- [x] Env stack: `Gen1Env → SingleAgentWrapper → SB3Wrapper → Monitor → DummyVecEnv`
- [x] OU-only species pool (39 species, filtered from `gen1randombattle`)
- [x] KO-probability features: `prob_ko`, `expected_dmg_fraction`, `recharge_trap` per move

---

## Phase 3 — Curriculum Training ✅ COMPLETE (Phase A reached, gate not yet crossed)

- [x] `src/train.py` — 4-phase curriculum with critic warmup, temperature annealing, mean+min gate
- [x] `src/callbacks.py` — WinRateCallback with per-env stats, milestone saves, Elo tracking
- [x] `src/obs_transfer.py` — zero-pad obs expansion for checkpoint compatibility
- [x] Mixed league: SelfPlay (env 0) + MaxDamage + TypeMatchup + SoftmaxDamage (envs 1-3)
- [x] Phase gate: `mean(heuristics) >= 0.65 AND min(heuristics) >= 0.50` (2 consecutive evals)
- [x] 57 training runs completed
- **Peak result:** 65% WR vs OU-only heuristic opponents (run_057, step 355K)

**Plateau analysis (see `notes/run_057_postmortem.md`):**
The agent is stuck in a "never switch" local optimum (~1% vol_switch). The PPO gradient from
terminal rewards alone is too weak to pull it out. Behavioural cloning from TypeMatchupPlayer
is the most promising path to bootstrap switch behaviour.

---

## Phase 4 — Behavioural Cloning Bootstrap (NOT STARTED)

Goal: use BC from TypeMatchupPlayer to seed the switch-behaviour prior, then fine-tune with PPO.

- [ ] Run `scripts/generate_bc_data.py` with TypeMatchupPlayer as expert
- [ ] Train BC model via `scripts/behavioral_cloning.py`
- [ ] Fine-tune with PPO from BC checkpoint (obs_transfer compatible)
- [ ] Gate: sustained mean heuristic WR >= 0.65

---

## Phase 5 — Self-Play League (BLOCKED on Phase 4)

- [ ] `src/selfplay_train.py` — frozen policy pool, Elo-tracked snapshots
- [ ] Gate: self-play Elo improvement plateau

---

## Phase 6 — Evaluation & Polish (BLOCKED on Phase 5)

- [ ] `scripts/evaluate.py` — benchmark vs all heuristics + self-play pool
- [ ] Watch replays: Showdown HTML replays
- [ ] Potential: Showdown ladder account

---

## Key Gen 1 Mechanics

1. **1/256 miss** — moves nominally 100% accurate have a tiny miss chance
2. **Freeze is permanent** — very strong status
3. **Sleep Clause** — only 1 opponent Pokemon can be asleep at a time
4. **Hyper Beam no-recharge on KO** — if it KOs, no recharge needed
5. **Crit rate from Speed** — fast Pokemon (Tauros, Starmie) have very high crit rates
6. **Single Special stat** — no Sp.Atk/Sp.Def split
7. **No held items, no team preview**
