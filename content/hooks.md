# Screen Recording Hooks

Moments worth capturing on screen, logged as they happen.

---

## Phase 1 — Infrastructure

### ✅ First successful battle (2026-04-02)
**What happened:**
```
Connecting to Showdown server at ws://localhost:8000 ...
(If this hangs or errors: start the server first with 'node pokemon-showdown start --no-security')
Running 1 battle: RandomPlayer vs RandomPlayer ...
Battle complete. Winner: P2
Setup verified — poke-env <-> Showdown connection is working.
```
**Why it matters:** First proof the full stack works end-to-end — Python agent → WebSocket → Showdown → battle result. The "it's alive" moment.

---

## Phase 2 — Environment
*(pending)*

## Phase 3 — Baselines

### ✅ MaxDamagePlayer vs RandomPlayer benchmark (2026-04-02)
**What happened:**
```
Running 100 battles: MaxDamagePlayer vs RandomPlayer ...

Results (100 battles):
  MaxDamagePlayer wins :  99 / 100  (99.0%)
  RandomPlayer wins    :   1 / 100  (1.0%)

ENVIRONMENT OK — win rate above 80% threshold.
```
**Why it matters:** Confirms the environment is working correctly end-to-end. 99% also sets a brutal bar for the RL agent — beating a type-aware heuristic >60% will require genuinely strategic play. The 1 loss is likely a catastrophic type matchup (e.g. Random got lucky team composition).

## Phase 4 — PPO Training

### ✅ First PPO gradient update (2026-04-02)
**What happened:**
```
Using cuda device
----------------------------
| time/              |     |
|    fps             | 77  |
|    iterations      | 1   |
|    total_timesteps | 128 |
----------------------------
------------------------------------------
| train/                  |              |
|    approx_kl            | 0.0035618674 |
|    clip_fraction        | 0            |
|    clip_range           | 0.2          |
|    entropy_loss         | -1.77        |
|    explained_variance   | -1.34        |
|    learning_rate        | 0.0003       |
|    loss                 | 0.0103       |
|    n_updates            | 10           |
|    policy_gradient_loss | -0.0128      |
|    value_loss           | 0.128        |
Smoke test PASSED
```
**Why it matters:** The agent is alive and learning on GPU. `explained_variance = -1.34` is exactly right for a fresh policy — the critic has no idea what's happening yet. This is the "it's learning" moment for Act 3 of the video. Contrast this number against the same metric after 500k steps.

## Phase 5 — Self-Play
*(pending)*

### ✅ Win rate crossed 10% — step 10000 (2026-04-02)
- Avg battle length: 65.8 turns
- Switch rate: 46.8% of actions
- Best move used: move_1 (17.6% of move actions)
- **Why it matters for the video:** The agent crossed 10% win rate vs RandomPlayer at step 10,000. It is switching more than attacking — may be learning defensive play.

### ✅ Win rate crossed 25% — step 10000 (2026-04-02)
- Avg battle length: 65.8 turns
- Switch rate: 46.8% of actions
- Best move used: move_1 (17.6% of move actions)
- **Why it matters for the video:** The agent crossed 25% win rate vs RandomPlayer at step 10,000. It is switching more than attacking — may be learning defensive play.

### ✅ Win rate crossed 40% — step 10000 (2026-04-02)
- Avg battle length: 65.8 turns
- Switch rate: 46.8% of actions
- Best move used: move_1 (17.6% of move actions)
- **Why it matters for the video:** The agent crossed 40% win rate vs RandomPlayer at step 10,000. It is switching more than attacking — may be learning defensive play.

### ✅ Win rate crossed 50% — step 10000 (2026-04-02)
- Avg battle length: 65.8 turns
- Switch rate: 46.8% of actions
- Best move used: move_1 (17.6% of move actions)
- **Why it matters for the video:** The agent crossed 50% win rate vs RandomPlayer at step 10,000. It is switching more than attacking — may be learning defensive play.

### ✅ Win rate crossed 60% — step 20000 (2026-04-02)
- Avg battle length: 55.4 turns
- Switch rate: 42.2% of actions
- Best move used: move_1 (19.6% of move actions)
- **Why it matters for the video:** The agent crossed 60% win rate vs RandomPlayer at step 20,000. It is switching more than attacking — may be learning defensive play.

### ✅ Win rate crossed 75% — step 20000 (2026-04-02)
- Avg battle length: 55.4 turns
- Switch rate: 42.2% of actions
- Best move used: move_1 (19.6% of move actions)
- **Why it matters for the video:** The agent crossed 75% win rate vs RandomPlayer at step 20,000. It is switching more than attacking — may be learning defensive play.

---

### [MEDIA] Future Video Angles — logged 2026-04-02

#### Angle 1: How Close Is It to the Real Competitive RBY Meta?
Once we have a trained agent in gen1ou format (not randombattle), compare its team choices, move priorities, and win conditions against documented competitive RBY meta:
- Dominant Gen 1 OU Pokemon: Tauros, Chansey, Starmie, Alakazam, Exeggutor, Snorlax, Jynx, Slowbro, Lapras, Gengar
- Top strategies: Wrap/Clamp/Bind abuse, Sleep + sweep, Amnesia sweepers, Body Slam paralysis spread
- Does the agent rediscover these strategies from scratch, or find something different?
- Hook: "I gave an AI 500,000 battles to learn Pokemon. Here's what it discovered."

#### Angle 2: The Ghost/Psychic Bug — What If Gamefreak Fixed It?
Gen 1 shipped with a critical type chart bug: Ghost-type moves are IMMUNE to Psychic (0× instead of 2×). This makes Psychic the most broken type in the game — Starmie, Alakazam, Jynx, Slowbro, Exeggutor are all dominant partly because Ghost can't threaten them.

Other known Gen 1 bugs to investigate:
- Focus Energy: supposed to 4× crit rate, actually reduces it to 1/256
- 1/256 miss: all 100%-accurate moves have a 1/256 miss chance
- Hyper Beam: no recharge if it KOs
- Badge boosts: Gym badges permanently boost stats in battle
- Special stat: Amnesia doubles Special Attack AND Special Defense simultaneously

Experiment: patch Showdown to fix Ghost→Psychic (0× → 2×) and retrain the agent. Does Gengar (Ghost/Poison) become viable? Does the Psychic-type stranglehold break?
Hook: "What would competitive Pokemon look like if Gamefreak hadn't shipped a 30-year-old bug?"

**Why this is compelling for video:** The agent has no preconceptions — it will discover the optimal meta purely from battle outcomes. If we fix the bug, it will show us the counterfactual history of Pokemon competitive play.
