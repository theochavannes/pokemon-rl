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

### The Brain Too Small for the Job
**Hook:** "We were trying to teach a brain the size of a walnut to play chess."

The neural network had 63,000 parameters trying to process 704 features. The first layer compressed 704 inputs through 64 neurons — an 11:1 ratio. For comparison, image classifiers compress at 3:1. We upgraded to 300K parameters with [256, 128] layers — 5x the capacity. Combined with encoding what moves actually DO (secondary effects, recoil, self-destruct), the agent can finally see the full game.

**Visual:** Side-by-side network diagrams. Old: 704->64->64->10 (tiny funnel). New: 928->256->128->10 (proper pyramid). "Same game, 5x the brain."

---

## Act 8: The Full Picture (Sprint 5 — 2026-04-03)

### The Agent Learns What Moves Actually Do
**Hook:** "We went from 14 features per move to 19 — and the 5 new ones change everything."

The obs space went from 928 to 1222 dimensions. But it's not about the numbers — it's about WHAT the agent can now see:

1. **"This move paralyzes" vs "this move freezes"**: Before, Body Slam and Blizzard both just had a "secondary_chance" number. Now the agent knows Body Slam paralyzes (0.8) and Blizzard freezes (1.0). Freeze is PERMANENT in Gen 1 — the agent can learn why Blizzard is the best move in the game.

2. **"Don't bother, they're immune"**: Toxic on a Poison type? Thunder Wave on a Ground type? The agent now sees a status_immune flag = 1.0. No more wasted turns.

3. **"They're already statused"**: Thunder Wave on an already-paralyzed opponent? target_statused = 1.0. The agent can learn to stop doing useless things.

4. **"This move traps you"**: Wrap/Bind/Clamp/Fire Spin get a trapping flag. In Gen 1, these prevent ALL actions for 2-5 turns — one of the most broken mechanics in the game.

5. **Volatile status awareness**: Substitute up? Reflect halving damage? Confused? Leech Seeded? The agent can now see the battlefield, not just HP bars.

**Visual:** Before/after comparison of Thunder Wave's observation vector. Before: [0, 0.72, 0.43, 0, 1.0, 1.0, 0.8, 0, 0, 0, 0, 0, 0, 0]. After: [0, 0.72, 0.43, 0, 1.0, 0, 0, 0.8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0/1]. The zeros that used to make it look useless now carry meaning.

**Why it matters for the video:** This is the "giving the AI eyes" moment. All the previous training failures weren't because the AI was stupid — it was because it was playing blind. Now it can see the full game.

### The Split Brain (Sprint 6)
**Hook:** "We split the AI's brain in two — one half thinks about its team, the other half thinks about the opponent."

Before: a flat 1222→256→128 network. Every neuron processes all 1222 features at once — your own HP, the opponent's moves, whether Reflect is up, all mixed together. The network has to learn what to pay attention to from scratch.

After: a two-tower architecture. One tower processes your own team info (606 dims), the other processes opponent info (601 dims). Each tower compresses independently to 128 features. Then a merge layer combines them with 15 global features to make decisions.

**Why it's better:** The network no longer has to learn that "my HP" and "their HP" are fundamentally different kinds of information. Each tower specializes — the own-tower learns "what can I do?" and the opp-tower learns "what are they doing?" The merge layer learns "given what I can do and what they're doing, what should I pick?"

**Visual:** Network diagram — before (single funnel 1222→256→128) vs after (two funnels merging). Show the total parameter count: ~300K → ~500K. "Same game, split brain."

### The League of Past Selves (Sprint 7)
**Hook:** "Every 50,000 training steps, the AI saves a snapshot of itself. Then it plays against a random past version — including the version that couldn't even beat a random player."

This is fictitious self-play, the same technique used in AlphaStar and OpenAI Five. Instead of always fighting the latest version of itself (which can lead to circular strategies), the agent faces random historical versions. This forces robust play that works against many strategies, not just the current one.

**Visual:** Timeline showing snapshots accumulating: "Step 50K: first snapshot. Step 100K: pool of 2. Step 500K: pool of 10 past selves." Show the random selection — "this game, you're playing against yourself from 200K steps ago."

### The Two-Tower Disaster (Sprint 5-6-7 Post-Mortem)
**Hook:** "We assembled a 16-expert panel, implemented every recommendation, and the AI got WORSE."

Sprint 5 added 19 new obs features (status immunity, trapping, volatile status). Sprint 6 added a two-tower neural network architecture. Sprint 7 added more per-move features. Total: 928 to 1559 dimensions with a brain that processes own-team and opponent separately.

**The result:** run_045 peaked at 39% win rate and collapsed to 16% over 600K steps. run_043 (before the sprints) hit 62%.

**Root cause:** The two-tower architecture killed the BC warm-start. Old BC: 99% accuracy imitating MaxDamagePlayer with a flat MLP. New BC: 46.6% accuracy with two-tower. The loss literally plateaued at epoch 1 — the model couldn't learn AT ALL with the complex architecture.

**The deeper lesson:** Good architecture + bad initialization = worse than simple architecture + good initialization. Every successful game AI (AlphaStar, OpenAI Five) introduces complexity gradually. We tried to do it all at once.

**Visual:** Two training curves overlaid. run_043 (flat MLP, 928 dims): steady climb to 62%. run_045 (two-tower, 1559 dims): brief plateau at 39%, then free-fall to 16%. "More features, bigger brain, worse results."

**Why it's great content:** This is the "hubris" moment in the story. The team got confident after Sprint 3A's success and tried to do too much at once. The fix reveals the right approach: crawl, then walk, then run.

### The Polish Pass (Sprint 7)
**Hook:** "We added 6 more features per move that the pros said matter. Hyper Beam skips your next turn. Dream Eater only works on sleeping targets. Blizzard has 5 PP — use it wisely."

The obs space went from 1222 to 1559 dimensions. But each new feature encodes something a human player considers automatically:
- Can I afford to use Hyper Beam? (must_recharge)
- Is Dream Eater even useful right now? (requires_sleep)
- How many times can I use this move? (pp_max)
- What turn is it? (turn_phase — early game vs endgame changes everything)

---

## Act 9: The Diagnosis (2026-04-04)

### The Full Team Review That Changed Everything
**Hook:** "We spent a day reading every line of code, every training log, every failed experiment. And we found out the diagnosis we'd been working from was wrong."

After 48 runs and 0 progress, the GM demanded answers. The full team (ML, SYS, REVIEW, SE, RL, SB3, GYM, POKE-ENV) convened for a deep forensic review of the codebase.

**The wrong diagnosis we'd been operating on:** "The actor and critic share a features extractor. BC only trains the actor path. The critic's features never get trained." This sounded right. It was in the planning doc. Three runs were designed around fixing it.

**The truth:** For MlpPolicy, the "shared features extractor" is `nn.Flatten()` — a parameter-free reshape operation. **There are no shared learned parameters.** The actor has its own 1559→256→128→10 network. The critic has its own 1559→256→128→1 network. Completely separate. The "fix" everyone was working toward would have changed literally nothing.

**Visual:** Architecture diagram showing the two completely separate networks with the word "SHARED" crossed out over the Flatten layer. "The thing we thought was the problem... wasn't even a thing."

### The Real Problem: Chicken-and-Egg
**Hook:** "PPO needs a good value function to know which direction to improve. But the value function needs a good policy to learn from. Neither can go first."

PPO computes advantages using: A_t = r_t + γV(s_{t+1}) - V(s_t). When V is random, A_t is random. Random advantages → random policy gradient → BC knowledge erodes. The policy changes → new state distribution → V has to relearn. Neither component can stabilize.

**Visual:** Circular dependency diagram. V → A → π → states → V. "A circle where every node is wrong."

### The Actor Freeze Solution
**Hook:** "What if we just... stopped the policy from changing?"

The fix is almost comically simple: set `requires_grad=False` on the actor parameters. The policy stays at BC quality. The state distribution is fixed. The value function can learn as a normal supervised regression problem. When ExplVar goes positive, unfreeze the actor. Now PPO has reliable advantages and can actually improve.

**Visual:** Before — the circular dependency with all arrows red. After — the arrow from π back to V is cut (frozen). The value function can now learn without the ground shifting beneath it.

### The Hidden Monitoring Bug
**Hook:** "We also found out that every metric we've been tracking for the last 8 runs was wrong."

The `_move_damages_from_obs` function in callbacks.py reads move features with a stride of 14 (the original per-move feature count). The actual stride is 26. Moves 1-3 have been reading garbage observation indices. BestMv% and DmgEff — our two key behavioral metrics — have been unreliable since Sprint 5.

**Visual:** Show the observation layout with arrows pointing to what the code THINKS it's reading vs what it's ACTUALLY reading. "This reads base_power. This reads... trapping flag. Off by 12 indices."

**Why it matters for the video:** This is the "our instruments were broken" reveal. You can't navigate if your compass is lying to you. Every decision based on BestMv% trends was based on bad data.

### What to Watch For in Training
- **During actor freeze:** ExplVar climbing from negative toward positive. Win rate staying stable (BC quality preserved). "The critic is learning while the player holds still."
- **After unfreeze:** Win rate increasing beyond BC baseline. ExplVar staying positive. The moment the lines cross is the climax: "The AI is now better than its teacher."
- **The contrast shot:** Runs 046/048 (no freeze, ExplVar negative, BC erodes) vs the new run (freeze phase, ExplVar climbs, unfreeze, win rate improves). Same codebase, same data, one PyTorch flag different.

---

## Act 11: The Deeper Diagnosis (2026-04-05)

### "The Diagnosis of the Diagnosis Was Wrong"
**Hook:** "We built a 500-line plan around a theory we never tested. Then we tested it. It was wrong — in a way that made us find the REAL answer."

The previous session diagnosed "chicken-and-egg value bootstrap" as the root cause and proposed actor freezing. Before executing, the prompt for this session demanded: PROVE the hypothesis with evidence. Don't build another plan on guesswork.

**The smoking gun:** Run 043 had POSITIVE ExplVar the entire time (0.01-0.26). The claim "ExplVar stays negative across all runs" — the foundation of the previous plan — was factually wrong for run 043. The value function DID learn. PPO STILL didn't improve.

**Visual:** Side-by-side ExplVar charts. Run 043: consistently positive (green). Run 048: mostly negative (red). Both: win rate stuck at 50%. "The value function was working. The agent wasn't."

### Three Problems, Not One
**Hook:** "It wasn't a bug. It was three bugs standing on each other's shoulders in a trenchcoat."

1. **The value network was 4x too big.** Run 043 used [64,64] (104K params). Run 048 used [256,128] (432K params). Starting from random init, the bigger network couldn't converge. "Like teaching a new brain — the bigger the brain, the longer it takes to wake up."

2. **PPO was eroding the BC knowledge.** BestMv% declined in EVERY run at ~0.04%/K steps. PPO's noisy advantages were slowly overwriting the expert knowledge. "Like a student taking random advice that slowly makes them forget what they actually know."

3. **Even with a working value function, the signal was too weak.** Run 043 proved this: ExplVar positive, win rate flat. The advantage estimates explain only 10-25% of return variance. The rest is team composition RNG. "Even with a working compass, the wind is too strong."

**Visual:** Three-layer diagram. Layer 1 (value init): "cold start." Layer 2 (BC erosion): "knowledge drain." Layer 3 (signal): "needle in a haystack." "You can't solve layer 3 without solving layers 1 and 2 first."

### The Weight Copy Insight
**Hook:** "The fix for the value function wasn't to train it harder. It was to give it a cheat sheet."

After BC training, the policy network already learned to process 1559 dims of Pokemon battle state into 128 useful features. Those features know what makes a move good, what types match up, when a Pokemon is in trouble. The value function needs those exact same features — it just needs to learn "is this a good or bad game state?" on top of them.

So: copy the policy_net weights to the value_net. 432K parameters of useful features, instantly initialized. The value head (128→1) is 129 parameters that need to learn from scratch — trivial.

**Visual:** Animated weight transfer. Left network (policy) lights up blue — "trained by BC." Weights flow to right network (value). "Same features, new purpose. 432,000 parameters instantly useful."

### The Validation-First Approach
**Hook:** "This time, we're testing before we build."

Two 30-minute experiments before committing to the full plan:
- Experiment A: Train from scratch (no BC). Does ExplVar go positive? If no, the obs space is broken.
- Experiment B: Use run 043's hyperparams (LR=1e-4) with [256,128]. Does ExplVar match run 043? If yes, the learning rate was the whole problem.

**Visual:** Decision tree. "If A and B both fail → fundamental obs space issue. If A passes, B fails → BC handoff problem. If both pass → we were overthinking it."

**Why it's great content:** This is the "scientific method" moment. After 48 runs of hypothesis → build → fail, the team finally stops and says "let's just test the hypothesis first." The audience sees ML research done right.

### The Warmstart Was Already Perfect
**Hook:** "The value function we've been trying to fix for days was already at 99% accuracy. The problem was that PPO kept breaking it."

The critic warmstart script produced a value function with ExplVar 0.99 — it perfectly predicted game returns. But after 23K steps of PPO, the same value function had ExplVar -0.26. The weight changes between the two were barely measurable (mean absolute diff: 0.02). PPO's 10 training epochs per rollout were overfitting the value network to tiny batches, destroying its ability to generalize.

**Visual:** ExplVar gauge: starts at 0.99 (green). After PPO step 1: 0.95. Step 2: 0.80. Step 3: 0.40. Step 10K: -0.26 (red). "A nearly perfect value function, destroyed in minutes."

### 7 Experiments, 1 Answer: n_epochs=3
**Hook:** "We ran 7 experiments in parallel. Six failed. One worked. The difference was a single number."

Tested every combination: high LR, low LR, frozen actor, unfrozen, MC returns, bootstrap. Only configs with n_epochs=3 (instead of 10) maintained positive ExplVar. The value network was memorizing each rollout's 8192 data points in 10 passes, then forgetting how to generalize.

**Visual:** Bar chart of 7 experiments. Six red bars (negative ExplVar). One green bar (C2: freeze + n_epochs=3, ExplVar +0.185). Circle the "3" in n_epochs=3. "This one number."

### The Training Run That Actually Works (run_052)
**Hook:** "For the first time in 52 runs, the value function stayed positive after unfreeze."

Warmup phase: actor frozen for 50K steps. ExplVar climbed from -0.03 to +0.17. BestMv% held perfectly at 57-58% (BC knowledge preserved). Then the actor unfroze. ExplVar: still positive at +0.07. BestMv%: 58% and stable. Win rate: 45-67%.

Compare with run 048: ExplVar immediately negative, BestMv% declining from step 1.

**Visual:** Split training curve. Run 048 (red): ExplVar dips negative immediately, BestMv% declines. Run 052 (green): ExplVar stays positive through warmup and post-unfreeze. "The same architecture, the same data, three numbers changed."

---

## Act 10: Process Discipline (2026-04-04)

### The Team That Kept Building Plans on Wrong Assumptions
**Hook:** "We've had 48 failed training runs. The last plan was 500 lines long and built on a diagnosis that was wrong. What if the process itself is the bug?"

After the GM demanded accountability, the team realized a pattern: every plan hypothesized a root cause, immediately built an elaborate fix around it, then failed because the hypothesis was never tested. The shared-features-extractor diagnosis? Wrong. The critic warmstart approach? Built on the wrong assumption. The two-tower architecture? Solved a problem that wasn't the bottleneck.

**The fix:** Added 5 process discipline tools (antigravity skills) to AGENTS.md:
- `/systematic-debugging` — no fixes without verified root cause
- `/phase-gated-debugging` — 5-phase gate, can't edit code until root cause confirmed
- `/planning-with-files` — save findings every 2 actions, never lose context
- `/closed-loop-delivery` — task incomplete until metrics prove it works
- `/evaluation` — regression tests catch "ExplVar went negative" automatically

**New hard rule:** Before any plan that proposes code changes, the team must complete a debugging pass. Plans on unverified hypotheses have a 100% failure rate so far.

**Visual:** Timeline of failed diagnoses — "shared features extractor" (wrong), "critic warmstart insufficient" (symptom not cause), "two-tower will fix it" (made it worse). Each one with a big red X. Then the new rule: "PROVE IT FIRST."

**Why it's great content:** Meta-learning moment. The team isn't just training an AI — they're learning how to debug AI. The audience sees that ML research is 90% diagnosis and 10% implementation.
