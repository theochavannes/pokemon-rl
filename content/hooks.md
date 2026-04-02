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

### ✅ Win rate crossed 75% — step 194604 (2026-04-02)
- Avg battle length: 38.5 turns
- Switch rate: 27.4% of actions
- Best move used: move_1 (21.5% of move actions)
- **Why it matters for the video:** The agent crossed 75% win rate vs RandomPlayer at step 194,604. It heavily prefers attacking over switching — aggressive style.
