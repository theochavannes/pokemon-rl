# Video Content Hooks

Curated moments and narrative beats for the YouTube video, maintained by [MEDIA].

---

## Act 1: Setup & First Signs of Life

### The Stack
- Pokemon Showdown (local server) + poke-env + MaskablePPO
- Gen 1 random battles — no team building, pure battle skill
- 421-dimensional observation space: HP, types, moves, boosts, status, base stats, opponent info (all level 100)

### MaxDamage Baseline: 99% vs Random
Our heuristic bot (always picks highest-damage move) crushes RandomPlayer 99/100. This is the bar the RL agent must beat.

### First PPO Gradient Update
```
explained_variance = -1.34  (the agent has no clue what's happening)
entropy_loss = -1.77        (picks actions randomly)
```
**Hook:** Show this next to the same metrics after training. The contrast tells the story.

---

## Act 2: Learning to Beat Random (Phase A)

### Key Milestones (consistent across runs 010-013)
| Step | Win Rate | Switch% | Avg Turns | What changed |
|------|----------|---------|-----------|-------------|
| 0 | ~50% | 60% | 70+ | Random baseline (6 switches, 4 moves = 60% switch by chance) |
| 10k | 53% | 50% | 67 | First sign of learning — attacking more than switching |
| 35k | 73% | 46% | 61 | Consistent improvement, games getting shorter |
| 55k | 80% | 43% | 53 | Dominant — finishing games faster |
| 70k | 90% | 41% | 49 | Mastery — crushing Random, ready to advance |

**Narrative beat:** The agent discovered that attacking > switching, and learned to finish games.

---

## Act 3: The MaxDamage Wall (the struggle)

### The Cliff — Repeated Across Every Run
| Run | Approach | Phase A peak | Phase B result |
|-----|----------|-------------|----------------|
| run_010 | Epsilon-greedy, step triggers | 90% | Collapsed to 2%, recovered to 30-50%, stuck |
| run_011 | Epsilon-greedy, smooth | 87% | Graduated too fast (only 4 evals) |
| run_012 | Epsilon = f(win_rate), rate-limited | 90% | Collapsed to 2%, stuck at 20-35% |
| run_013 | 10x reward shaping | 88% | Held 80% longer, still declined to 27-45% |

**Hook:** "The agent learned to crush random opponents, then hit a wall it couldn't climb."

### Why It's Stuck — Expert Panel Diagnosis (2026-04-02)
Assembled panel: [RL], [SB3], [POKE-ENV], [RBY], [SE] + two external consultants (Principal RL Engineers, DeepMind/OpenAI backgrounds, former MIT professors).

Key findings:
1. **Reward signal 50x too weak** — per-step shaping was 0.01-0.05, drowned by terminal +/-1.0. Agent couldn't tell which moves were better. Fixed: 10x increase.
2. **Epsilon-greedy creates bimodal opponent** — coin flip between fully random and fully strategic per turn. Not a coherent mid-skill player. Fixed: replaced with softmax/Boltzmann sampling.
3. **StallPlayer was broken** — string vs enum comparison meant it never used status moves. Was a MaxDamage clone. Fixed.
4. **ValueError catch injected fake losses** — poke-env desyncs returned -1.0 reward, poisoning training. Fixed: neutral 0.0.

**Visual:** The debugging journey itself is content — "we assembled RL experts and they found the agent was basically training blind."

### The Softmax Solution (2026-04-02)
**Before (epsilon-greedy):** Per-turn coin flip — either 100% random OR 100% max-damage. Agent faces a schizophrenic opponent.

**After (softmax/Boltzmann):** Moves sampled proportional to damage^(1/temperature). High temp = soft preferences. Low temp = near-deterministic. Every turn is coherent — opponent always somewhat prefers stronger moves.

Formula: `prob(move_i) = damage_i^(1/temp) / sum(damage_j^(1/temp))`

Temperature anneals from 2.0 (soft) to 0.1 (near-argmax) based on win rate over last 500 games.

**Why it's better:** The agent faces a consistent "mid-skill" player that gradually gets sharper, not a random/expert coin flip.

### Reward Shaping Decay (2026-04-02)
Shaping starts at full strength (learn basics from intermediate rewards) and linearly decays to 0 over 5000 battles (eventually optimize purely for winning). Global across all phases — never resets.

---

## Act 4: The Fix (Pending Results)

### Current Curriculum (run_014+)
| Phase | Opponent | Difficulty |
|-------|----------|-----------|
| A | Random (60% switch) | Fixed, easy |
| B | RandomAttacker (85% attack) | Fixed, medium |
| C | SoftmaxDamage (temp 2.0 -> 0.1) | Smooth ramp to MaxDamage |
| D | Mixed + frozen self-play | All opponents, endgame |

### The Hidden Bug — Temperature Annealing Was Dead
**Hook:** "We built an entire softmax opponent system and it turns out the temperature never changed."

The callback that anneals the softmax temperature checked `hasattr(opp, "epsilon")` as a guard. SoftmaxDamagePlayer uses `temperature`, not `epsilon`. The entire Phase C annealing was silently broken — the opponent stayed at the initial soft temperature forever. Phase C could only exit by hitting max_steps (400k), never by actually graduating at full difficulty.

**Visual:** Show the code diff. One line changed, entire training pipeline unlocked.

### What to Capture
- Save run_013 Phase B plateau (20-45%) as "before softmax"
- Run with softmax opponent as "after"
- Overlay training curves for the comparison

### The Professionalization Moment — Tooling Up
**Hook:** "We stopped coding the agent and started building the workshop."

After the 10-agent audit found 5 bugs (including the dead temperature annealing), we realized the project had zero safety nets — no pre-commit hooks, no proper CI, no AI code review, no dependency management. Just a single Pylint check with half its rules disabled.

**What changed:**
- Replaced Pylint with Ruff (10-100x faster, catches more)
- Added pre-commit hooks (auto-format, block large files — critical when one `git add .` could commit a 200MB model)
- Modern CI pipeline (lint then test on every PR)
- CodeRabbit AI code review (caught 5 issues on the first PR it reviewed)
- Dependabot for automatic security updates
- Claude Code hooks (every AI edit gets auto-formatted)

**Visual:** Split screen — messy git status with untracked replays and IDE files vs clean status after cleanup. Show CodeRabbit's first review catching real issues.

**Why it's good content:** Solo developer doing ML realizes they need the same tooling discipline as a team. The audience learns why these tools matter for ANY project, not just RL.

---

## Future Content Angles

### Gen 1 Bug Experiments
- Ghost/Psychic immunity bug: patch it, retrain, see if Gengar becomes viable
- Focus Energy bug: actually reduces crit rate instead of boosting it
- Hook: "What would competitive Pokemon look like without a 30-year-old bug?"

### Self-Play Arc
- Agent trains against frozen copies of itself
- Does it discover known Gen 1 meta strategies? (Tauros dominance, Wrap abuse, Sleep + sweep)
- Hook: "I gave an AI 500,000 battles to figure out Pokemon. Here's what it invented."

### ✅ Win rate crossed 55% — step 17204 (2026-04-02)
- Avg battle length: 64.2 turns
- Switch rate: 47.8% of actions
- Best move used: move_1 (15.8% of move actions)
- **Why it matters for the video:** The agent crossed 55% win rate vs RandomPlayer at step 17,204. It is switching more than attacking — may be learning defensive play.

### ✅ Win rate crossed 60% — step 17204 (2026-04-02)
- Avg battle length: 64.2 turns
- Switch rate: 47.8% of actions
- Best move used: move_1 (15.8% of move actions)
- **Why it matters for the video:** The agent crossed 60% win rate vs RandomPlayer at step 17,204. It is switching more than attacking — may be learning defensive play.

### ✅ Win rate crossed 65% — step 26804 (2026-04-02)
- Avg battle length: 63.5 turns
- Switch rate: 46.2% of actions
- Best move used: move_1 (16.0% of move actions)
- **Why it matters for the video:** The agent crossed 65% win rate vs RandomPlayer at step 26,804. It is switching more than attacking — may be learning defensive play.

### ✅ Win rate crossed 70% — step 26804 (2026-04-02)
- Avg battle length: 63.5 turns
- Switch rate: 46.2% of actions
- Best move used: move_1 (16.0% of move actions)
- **Why it matters for the video:** The agent crossed 70% win rate vs RandomPlayer at step 26,804. It is switching more than attacking — may be learning defensive play.

### ✅ Win rate crossed 75% — step 35464 (2026-04-02)
- Avg battle length: 59.8 turns
- Switch rate: 44.6% of actions
- Best move used: move_1 (16.1% of move actions)
- **Why it matters for the video:** The agent crossed 75% win rate vs RandomPlayer at step 35,464. It is switching more than attacking — may be learning defensive play.

### ✅ Win rate crossed 80% — step 49440 (2026-04-02)
- Avg battle length: 53.1 turns
- Switch rate: 39.8% of actions
- Best move used: move_1 (21.4% of move actions)
- **Why it matters for the video:** The agent crossed 80% win rate vs RandomPlayer at step 49,440. It heavily prefers attacking over switching — aggressive style.

### ✅ Win rate crossed 85% — step 49440 (2026-04-02)
- Avg battle length: 53.1 turns
- Switch rate: 39.8% of actions
- Best move used: move_1 (21.4% of move actions)
- **Why it matters for the video:** The agent crossed 85% win rate vs RandomPlayer at step 49,440. It heavily prefers attacking over switching — aggressive style.

### ✅ Win rate crossed 55% — step 100324 (2026-04-02)
- Avg battle length: 43.6 turns
- Switch rate: 30.7% of actions
- Best move used: move_1 (18.9% of move actions)
- **Why it matters for the video:** The agent crossed 55% win rate vs RandomPlayer at step 100,324. It heavily prefers attacking over switching — aggressive style.

### ✅ Win rate crossed 60% — step 106668 (2026-04-02)
- Avg battle length: 43.0 turns
- Switch rate: 30.3% of actions
- Best move used: move_1 (19.4% of move actions)
- **Why it matters for the video:** The agent crossed 60% win rate vs RandomPlayer at step 106,668. It heavily prefers attacking over switching — aggressive style.

### ✅ Win rate crossed 65% — step 144256 (2026-04-02)
- Avg battle length: 41.4 turns
- Switch rate: 28.8% of actions
- Best move used: move_1 (21.3% of move actions)
- **Why it matters for the video:** The agent crossed 65% win rate vs RandomPlayer at step 144,256. It heavily prefers attacking over switching — aggressive style.

### ✅ Win rate crossed 70% — step 146320 (2026-04-02)
- Avg battle length: 40.2 turns
- Switch rate: 28.7% of actions
- Best move used: move_1 (21.3% of move actions)
- **Why it matters for the video:** The agent crossed 70% win rate vs RandomPlayer at step 146,320. It heavily prefers attacking over switching — aggressive style.

### The Observability Moment (2026-04-02)
**Hook:** "We realized the training was a black box — if the terminal crashed, we lost everything."

The project had zero logging infrastructure. Every training output was a bare `print()` that disappeared when the window closed. Worse, poke-env's internal logger was spamming hundreds of duplicate warnings per run about specific Pokemon (Vaporeon kept appearing). The signal was drowning in noise.

**What changed:** Added proper Python logging with file output (`training.log` per run), a duplicate message filter that suppresses repeated warnings after 3 occurrences, and silenced poke-env's chatty per-battle messages. Now every eval, milestone, curriculum change, and shaping decay is recorded to disk with timestamps.

**Visual:** Show a split — left side: terminal with hundreds of identical Vaporeon warnings scrolling past. Right side: clean log file with structured eval lines and "suppressed 47 duplicate messages."

**Why it matters for the video:** This is the "building the workshop" arc continuing. You can't debug ML if you can't see what happened. Especially relevant when runs take hours and you come back to find something went wrong.

### ✅ Win rate crossed 75% — step 194604 (2026-04-02)
- Avg battle length: 38.5 turns
- Switch rate: 27.4% of actions
- Best move used: move_1 (21.5% of move actions)
- **Why it matters for the video:** The agent crossed 75% win rate vs RandomPlayer at step 194,604. It heavily prefers attacking over switching — aggressive style.

---

## Act 5: The Variance Problem (2026-04-03)

### The 90% Ceiling
**Hook:** "The agent hit 90% win rate in the first 2000 steps — then spent 100,000 more steps learning absolutely nothing."

Evidence: run_030 training log shows win rate at step 2K = 0.88, at step 98K = 0.89. Completely flat. Meanwhile run_027 graduated at 98% in 5K steps — pure luck of the evaluation window. The 95% target was being hit or missed by RNG, not by learning.

**Visual:** Overlay run_027 (instant graduation) vs run_030 (100K steps, stuck). Same code, same hyperparameters, wildly different outcomes. The randomness IS the story.

### Expert Panel #2: "How do you teach an AI when the exam is rigged?"
**Hook:** "We brought in RL professors, Meta/DeepMind engineers, and a top competitive Pokemon player to solve one question: how do you give an AI credit for skill when half the outcome is luck?"

The key insight came from the competitive player: **type charts don't predict Gen 1 matchups.** What matters is which Pokemon you rolled (tier quality), speed tiers, and sleep move access. The professors said the solution is a "privileged critic" — an evaluator that sees both full teams (info the agent never gets) and tells the training loop "this game was supposed to be hard/easy."

**Visual:** Split screen — agent's view (one Pokemon visible on each side) vs evaluator's view (both full teams with tier ratings). "The agent is playing blind. The evaluator sees the whole board."

### The Forced Switch Bug
**Hook:** "Our metrics said the agent was voluntarily switching 18% of the time and never being forced to switch. That's impossible — Pokemon forces switches when your active faints."

Root cause: the callback was reading the action mask AFTER the step executed, not before. After a forced switch completes, the new state has moves available → classified as voluntary. Every forced switch was being miscounted.

**Visual:** Code diff — one line change (`self._get_action_masks()` → `self.locals["action_masks"]`).

---

## Act 6: The League (Sprint 3A — 2026-04-03)

### The Architecture Pivot
**Hook:** "We threw out the entire training curriculum and started over."

The sequential 4-phase curriculum (Random → RandomAttacker → SoftmaxDamage → Mixed) was fundamentally flawed: each phase trained the agent for one opponent, then the next phase's new opponent erased what it learned. Classic catastrophic forgetting.

**Before:** 4 phases, each with one opponent type. Agent masters each, then forgets.
**After:** 1 phase, 4 opponents simultaneously. Every rollout includes self-play, MaxDamage, TypeMatchup, and SoftmaxDamage. The agent can never forget because every opponent is always present.

**Visual:** Side-by-side training curves — old curriculum showing the "cliff" at each phase transition vs new mixed league with (hopefully) smooth improvement.

### The Conservative Clipping Trick
**Hook:** "One number change to prevent the AI from forgetting everything it learned."

PPO's `clip_range` went from 0.2 to 0.1. This limits how much the policy can change per gradient update — an implicit trust region that keeps the agent close to its BC warm-start. Used in InstructGPT/ChatGPT's RLHF training for the same reason: prevent the fine-tuned model from drifting too far from its supervised starting point.

**Why it's good content:** Connects our Pokemon project to state-of-the-art LLM training. "The same trick OpenAI uses to keep ChatGPT from going off the rails."

### The League Roster
**Visual:** Show all 4 opponents in a "league bracket" format:
- **Self-Play (Env 0):** A frozen copy of the agent itself, updated every 50K steps. Starts as the BC warm-start (a MaxDamage clone). Gets progressively smarter as training continues.
- **MaxDamagePlayer (Env 1):** Always picks the highest-damage move. Pure aggression.
- **TypeMatchupPlayer (Env 2):** Switches to type advantages. Forces the agent to handle smart switching.
- **SoftmaxDamagePlayer (Env 3):** Probabilistic damage selection, temperature anneals from random-ish (2.0) to near-deterministic (0.1) based on the agent's win rate.

### The Silent Forfeit Catastrophe
**Hook:** "The agent was forfeiting every single game and the training just kept going like nothing was wrong."

First mixed_league run (run_036): 25% win rate, 9-turn average, 0% voluntary switches. Looked bad but not suspicious. Then we opened the replays — **every game ended with the agent forfeiting**, many on turn 1 before any moves were played.

**Root cause:** poke-env validates opponent moves with `strict=True`. When the opponent's chosen order doesn't match valid orders (battle state timing), it raises ValueError. Our error handler caught it and called reset() — which forfeits the current game.

**The kicker:** Env 0 (self-play via FrozenPolicyPlayer) worked perfectly because FrozenPolicyPlayer uses `strict=False`. The heuristic opponents (MaxDamage, TypeMatchup, SoftmaxDamage) all forfeited every game. The training ran for 24K steps of pure garbage data and never flagged a problem.

**Fix:** One line — `strict=False` in the Gen1Env constructor. Invalid orders now gracefully fall back to random instead of crashing.

**Visual:** Side-by-side replay comparison — "working" game (env 0, 40+ turns) vs "broken" game (env 1, forfeit on turn 1). Same code, same agent, different outcome based on one boolean flag buried in poke-env internals.

### What to Watch For
- Does BestMv% stay above 70%? (BC knowledge preserved)
- Does Vol.Switch% start increasing? (TypeMatchup forcing switching decisions)
- Does the agent beat all heuristics at >50% within 500K steps?
- Does the self-play opponent matter, or does the agent just learn from heuristics?

---

## Act 7: The Switching Problem (2026-04-03)

### PPO Cannot Learn Switching
**Hook:** "We spent 200,000 steps trying to teach an AI to switch Pokemon. It never switched once."

After fixing the forfeit bug, the mixed league ran properly — real games, 27-turn averages, all opponents active. But across two full runs (run_037 and run_038), voluntary switch rate stayed at exactly 0.0%. The agent NEVER voluntarily switched a single time.

**Why:** The BC warm-start was trained on MaxDamagePlayer, which never switches. The agent inherited a -1.37 logit bias against switching. We tried:
- Halving the bias (-1.37 → -0.69): no effect
- clip_range 0.1 (conservative): biases literally didn't move in 68K steps
- clip_range 0.2 (standard): biases drifted but BestMv% eroded without gaining switching

**The deeper problem:** Switching produces ZERO immediate reward. The reward function gives +0.5 for KOs and +1.0 for winning. When you switch to resist a hit, the reward that turn is 0.0. The benefit shows up 3-10 turns later when you eventually KO. No RL algorithm can learn a behavior that produces no measurable signal.

### The Realization
**Hook:** "You can't learn a skill no one ever showed you."

Every successful game AI (AlphaStar, OpenAI Five) was pre-trained on expert data that included the full behavioral repertoire. Our BC was trained on MaxDamagePlayer — an expert at move selection, but an expert that NEVER switches. The agent perfectly learned "never switch" alongside "pick good moves."

**Visual:** Show the action_net bias values — switches at -1.37, moves at +0.08. A 1.46 logit gap that PPO couldn't close in 200K steps. "The weights were screaming 'don't switch' and the reward function had nothing to say about it."

### The Fix: Teach Before Training
Build a smarter teacher that combines MaxDamage's move selection with competitive switching strategy. Re-train BC on data that includes strategic switching. Then PPO has a starting point that actually includes the skill we want it to refine.

### The Deeper Problem: The AI Doesn't Know What Its Moves Do
**Hook:** "We realized the AI was playing Pokemon without knowing what its moves actually do."

The observation space gives each move 5 features: base_power, type, PP, effectiveness, accuracy. That's it. Thunder Wave (the most important move in Gen 1) looks like "0 damage, Electric type, 100% accuracy." Swords Dance looks like "0 damage, Normal type, 100% accuracy." Recover looks like "0 damage, Normal type, 100% accuracy." They're all identical to the neural network — and all look worse than any attacking move.

The agent has ZERO reason to ever use a status move. It can't tell the difference between Thunder Wave (cripples opponent permanently), Swords Dance (doubles attack power), Recover (heals 50% HP), and Splash (does literally nothing). They all look like "0bp move = bad."

**Visual:** Show the 5 move features side by side for Thunder Wave, Swords Dance, Recover, and Splash. All four produce nearly identical observation vectors. "The AI thinks these four moves are the same."

This is like teaching someone chess without telling them that knights can jump, bishops move diagonally, and pawns can promote. You can learn some basic strategy, but you'll never play well.
