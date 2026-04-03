# Expert Panel Discussion #3: Why The Agent Can't Learn — Full Conversation

**Date:** 2026-04-03
**Context:** run_032 — Phase A graduated at 96%->91% in 7K steps, Phase B stuck at 50-55% for 200K steps, Phase C collapsed to 0-3% for 400K steps, Phase D declining from 66% to 22%.

## Participants

**Internal team:** [ML], [RL], [SE], [REVIEW], [SYS], [GYM], [SB3], [RBY], [PM], [MEDIA]
**External:**
- [Smogon1] — Top-5 Smogon RBY ladder, 3x tournament winner
- [Smogon2] — RBY teambuilder/analyst, Smogon tiering council
- [RL-Debug] — Principal Research Scientist, specializes in diagnosing RL training failures (ex-DeepMind)
- [RL-Curriculum] — Associate Professor, UC Berkeley, published on curriculum learning and catastrophic forgetting

---

## Part 1: Diagnosing The Collapse

**[PM]:** Here's the data from run_032. Phase A graduated instantly. Phase B stuck at 50-60% for 200K steps against RandAttacker. Phase C collapsed to 0-3% against SoftmaxDmg and stayed there for 400K steps — avg turns 7-11, meaning the agent dies in 3-4 real turns. Phase D started at 66% (opponents 95% random) and declined to 22% as opponents strengthened. Something is fundamentally broken. Floor is open — I want at least 30 concrete ideas, ranked.

**[RL-Debug]:** Let me start with the Phase C data because it's the most diagnostic. The agent is at 0-3% win rate for 400,000 steps. That's not "slow learning" — that's **zero gradient signal**. Let me explain why.

PPO uses advantages `A(s,a) = R(s,a) - V(s)`. If the agent loses every game, the value function learns `V(s) ~ -3.0` (the loss penalty) for every state. Then advantages for every action become approximately zero: `A ~ -3.0 - (-3.0) = 0`. The policy gradient is:

```
grad(J) ~ E[A(s,a) * grad(log pi(a|s))] ~ 0
```

**When you always lose, PPO literally cannot distinguish good actions from bad ones.** There's no variance in outcomes to learn from. This is the classic **reward sparsity death spiral** — losing constantly provides exactly as much learning signal as winning constantly.

**[RL-Curriculum]:** And look at the phase transition. Phase A -> Phase B: win rate drops from 91% to 58%. Phase B -> Phase C: drops from ~60% to 3%. These are **catastrophic cliffs**. The agent has no time to adapt — it goes from comfortable to annihilated in a single step.

This is catastrophic forgetting in reverse. The policy that worked against Random is not just insufficient against SoftmaxDmg — it's actively counterproductive. The skills don't transfer because the optimal strategy changes fundamentally between opponents.

**[Smogon1]:** Let me explain WHY the strategy changes so drastically between opponents.

Against Random: the opponent switches randomly 60% of the time. You win by just attacking. Any attack beats a random switch. The "optimal" strategy is literally "press any attack button."

Against SoftmaxDmg: the opponent **always attacks with high-damage moves**. Now it's a DPS race. If you pick the wrong move (low damage, bad typing), you lose because they're picking the RIGHT move. This requires the agent to actually evaluate type effectiveness, base power, and STAB — none of which mattered against Random.

**The skills learned against Random are USELESS against SoftmaxDmg.** It's like learning to beat a toddler at chess and then playing Magnus Carlsen.

**[Smogon2]:** And it's worse than that. Against SoftmaxDmg at temperature 2.0 (the starting difficulty), the opponent is "soft but competent" — it favors high-damage moves but still has some randomness. But even at temp 2.0, it will almost always pick a super-effective move over a resisted one. The agent needs to understand type matchups from turn 1.

Look at the avg turns: **7-11 steps per game in Phase C**. That's 3-5 actual turns (each player moves once per turn, but steps include forced switches after faints). The agent is getting swept in 3-4 attacks. It's not even surviving long enough to learn what's happening.

**[RL-Debug]:** That's the critical feedback loop. Short episodes = few transitions per episode = poor credit assignment = no learning = continued short episodes. The agent needs to survive longer to collect data about what works, but it can't survive longer without already knowing what works.

**[ML]:** Let me highlight something from the data. In Phase C, the vol.switch% starts at 1.4% and slowly climbs to 16.6% over the 400K steps. The agent IS slowly learning that it should switch more. But it never translates into wins because it's switching randomly — not to the right Pokemon.

**[GYM]:** I want to flag something about the observation space. The agent has 421 dimensions of information including type effectiveness per move, base power, and bench Pokemon stats. But a [64, 64] MLP with 421 inputs has to learn which of those 421 features matter for which of 10 actions. That's a lot of structure to discover from scratch, especially when episodes are 7 steps long.

**[SB3]:** And the default MlpPolicy in SB3 uses a shared network for policy and value function with a single output head split. With 421 inputs and [64, 64] hidden layers, we have:
- Layer 1: 421 x 64 = 26,944 weights
- Layer 2: 64 x 64 = 4,096 weights
- Policy head: 64 x 10 = 640 weights
- Value head: 64 x 1 = 64 weights

The first layer has to do ALL the heavy lifting — compressing 421 features into 64 neurons. That's a 6.5:1 compression in one step. It might not have enough capacity to represent the relationships between move features and action choices.

**[REVIEW]:** OK, I've been listening. Let me challenge some assumptions.

First: we said "bigger networks were tested and found worse." The decision log shows [128,128] and [256,256] were tried. But were they tested under the SAME conditions we have now (proper reward shaping, softmax opponents, action masking)? Or were they tested back when we had broken reward signals and epsilon-greedy opponents? Because the conclusion "bigger nets don't help" might be from a time when nothing would have helped.

Second: we're assuming the agent SHOULD be able to beat SoftmaxDmg. What does MaxDamagePlayer (our heuristic) actually achieve against SoftmaxDmg at temp 2.0? If MaxDamage only gets 60% against SoftmaxDmg, then the 70% target is unreasonable.

**[ML]:** Good questions. The bigger networks were tested back in run_001-010, when reward shaping was broken and we were using epsilon-greedy opponents. We should re-test.

**[Smogon1]:** On the baseline question — MaxDamagePlayer should absolutely crush SoftmaxDmg at temp 2.0. MaxDamage always picks the optimal damage move. SoftmaxDmg at temp 2.0 mostly picks good moves but sometimes picks suboptimal ones. It's like a slightly drunk version of MaxDamage. MaxDamage should win 70-80%.

**[SE]:** We should actually run that benchmark. Without it we don't know what's achievable.

**[RL-Curriculum]:** Let me now outline the structural problems I see, and then we can brainstorm solutions.

**Problem 1: Phase transitions are cliffs, not ramps.** The agent goes from 90% to 3% in one step. No curriculum in the literature works this way. Effective curricula make the task 5-10% harder each time, not 90% harder.

**Problem 2: The agent learns Phase A policy without learning Phase B/C skills.** Against Random, "attack with anything" works. The agent never needs to learn type effectiveness, switching, or damage calculation. These skills aren't "harder versions" of what it learned — they're entirely different skills.

**Problem 3: Short episodes kill credit assignment.** In 7-step episodes, PPO's GAE can barely function. With gamma=0.99 and lambda=0.95, the effective horizon is ~20 steps. But if episodes are 7 steps long, most of the return is just the terminal reward. The per-step advantages are dominated by "did I win or lose" rather than "was this specific action good."

**Problem 4: The value function can't predict outcomes.** Explained variance was 0.157 — barely above random. If V(s) is wrong, advantages are noise, and the policy gradient is noise.

**[RL-Debug]:** Let me add **Problem 5: There is no exploration mechanism.** PPO's only exploration comes from entropy in the policy. With ent_coef=0.01, that's minimal. The agent at 0% win rate has no reason to try different strategies because it never stumbles into a win that would reinforce new behavior. This is exactly the **deceptive gradient** problem — the gradient says "everything is equally bad, change nothing."

**[RBY]:** And **Problem 6: Gen 1 battle mechanics create a high variance floor.** Critical hits in Gen 1 are based on speed (fast Pokemon get 20%+ crit rates). One crit can swing a game. Sleep is permanent. Freeze is permanent. These mechanics mean that even good play loses to RNG regularly.

---

## Part 2: Brainstorming Solutions

**[PM]:** OK, now let's brainstorm solutions. I want every participant to contribute. Let's aim for 30+ ideas.

**[RL-Curriculum]:**

**Idea 1: Reduce observation space drastically for early phases.** 421 dims is enormous for a [64,64] MLP. The network can't extract useful features from this. Start with just the active Pokemon matchup (~30 dims: own HP, own 4 moves w/ effectiveness, opp HP, opp type). Add bench info in later phases only after the agent learns basic move selection.

**Idea 2: Eliminate phase transitions entirely — use a single continuous curriculum.** Instead of discrete phases with new opponents, have ONE opponent that slowly morphs from random to strategic. The SoftmaxDamage temperature starts at 100.0 (basically random) and anneals to 0.1 over the entire training run. No distribution shift.

**Idea 3: Massively increase the step budget.** 200K steps for Phase B is nothing. At 4 envs x 45 turns/game, that's ~4,400 games. With a noisy reward signal and 421-dim observation, the network needs far more samples to learn. Try 2M steps per phase minimum.

**[RL-Debug]:**

**Idea 4: Use a per-turn survival bonus.** Add +0.01 per turn alive. Right now, dying in 3 turns and dying in 30 turns give nearly the same reward. The agent has no incentive to stay alive longer, which would naturally give it more data to learn from.

**Idea 5: Hindsight action relabeling.** After each loss, identify which moves would have been better (by type effectiveness) and create synthetic positive-reward transitions. This gives gradient signal even from losses.

**Idea 6: Intrinsic curiosity / count-based exploration.** Add reward for visiting new game states (new Pokemon matchups, different switch patterns). This breaks the "everything is equally bad" deadlock.

**Idea 7: Increase ent_coef dramatically during collapse.** When win rate drops below 10%, jack entropy up to 0.1 or even 0.5. Force the agent to explore wildly. When wins start coming, anneal entropy back down.

**[Smogon1]:**

**Idea 8: Teach type effectiveness BEFORE battles.** Create a supervised pre-training task: given a move type and opponent types, predict effectiveness multiplier. Pre-train the first layer of the network on this. The agent enters battle already knowing Fire > Grass.

**Idea 9: Start with 1v1, not 6v6.** Reduce the team size for initial training. Start with 1v1 (single Pokemon, no switching). The agent only needs to learn which move to pick. Once it masters 1v1, expand to 2v2, then 3v3, then 6v6. Each step adds one new skill (switching).

**Idea 10: Use only a subset of Pokemon initially.** Start training with only 10 well-known Pokemon (Tauros, Starmie, Alakazam, Snorlax, etc.). Once the agent learns these matchups, gradually introduce more Pokemon. This reduces the combinatorial explosion.

**[Smogon2]:**

**Idea 11: Add "perfect information" mode for early training.** Show the agent the opponent's full team from turn 1 (cheat mode). Once it learns to play with full info, gradually hide information. Learning strategy is easier when you can see everything.

**Idea 12: Make effectiveness features more prominent in the observation.** Effectiveness is one of 5 features per move, buried at position 3 of each move block, among 421 total features. A [64,64] MLP might not be able to isolate it.

**Idea 13: Add a "best move" signal to the observation.** Explicitly encode which move slot has the highest expected damage. One-hot vector [0,0,1,0] meaning "move 3 is your best option." This makes the learning problem trivially easy for the first layer.

**[GYM]:**

**Idea 14: Separate policy and value networks (no weight sharing).** Currently the MlpPolicy shares the first layers. The value function needs to learn "am I winning?" while the policy needs to learn "what should I do?" — these need different representations.

**Idea 15: Use a larger first hidden layer to avoid information bottleneck.** Change from [64, 64] to [256, 64] or [128, 64]. The first layer needs to compress 421 features; give it enough capacity.

**[SB3]:**

**Idea 16: Lower gamma for early phases.** With gamma=0.99, the effective horizon is ~100 steps. But Phase C episodes are 7 steps long. Set gamma=0.95 or even 0.9 for short-episode phases, then increase it as episodes get longer.

**Idea 17: Tune n_steps / n_epochs for short episodes.** With n_steps=2048 and 7-step episodes, each rollout contains ~290 complete episodes. Consider increasing n_epochs to squeeze more learning from each batch.

**Idea 18: Use learning rate warmup / scheduling.** Start with a lower learning rate (1e-4) and increase to 3e-4 as training stabilizes.

**[RL-Curriculum]:**

**Idea 19: Progressive neural network expansion.** Start with a [32, 32] network for simple opponents. When the agent masters them, copy weights into a larger [64, 64] network with extra neurons initialized to zero. This is "net2net" style expansion.

**Idea 20: Behavioral cloning warmup.** Before RL training, pre-train the policy via imitation learning on MaxDamagePlayer's actions. The agent starts with a "pick high-damage moves" policy instead of random.

**Idea 21: Population-based training (PBT).** Run 8-16 agents in parallel with different hyperparameters. Periodically copy the best agent's weights to the worst ones and mutate hyperparameters.

**[SE]:**

**Idea 22: Fix the Phase B target.** 95% against RandAttacker is unreasonable. Lower to 70-75%.

**Idea 23: Add gradient clipping diagnostics.** Log gradient norms per layer. If gradients are vanishing or exploding during Phase C, we'll see it.

**Idea 24: Save and analyze losing replays.** Take 50 Phase C losses and manually inspect them. Are there patterns? Is the agent always losing to the same Pokemon?

**[RL-Debug]:**

**Idea 25: Widen clip_range during recovery.** When win rate < 10%, increase clip_range from 0.2 to 0.4. Allows larger policy updates to escape the "always lose" basin.

**Idea 26: Periodic policy resets with value function retention.** If stuck at 0% for N steps, reinitialize the policy network (with logit bias) but keep the value function.

**Idea 27: Off-policy correction with replay buffer.** Store the top 10% of episodes (by return). Mix on-policy and replayed high-return episodes. Ensures the agent always sees SOME winning trajectories.

**[Smogon1]:**

**Idea 28: Create a "RandomDamage" opponent.** Between Random and SoftmaxDmg — picks a random MOVE (not switch) every turn. Tests if the agent can learn "pick the better move" without the opponent also picking smart moves.

**Idea 29: Use separate networks per game phase.** Early game (6v6) and late game (2v1) require different strategies. Switch sub-networks based on remaining Pokemon count.

**[Smogon2]:**

**Idea 30: Encode speed comparison explicitly.** In Gen 1, moving first is huge. Add: "am I faster? (own_speed / (own_speed + opp_speed))". Collapses a complex inference into one signal.

**Idea 31: Encode "can I one-shot?" per move.** For each move, add: `estimated_damage / opp_remaining_hp`. Turns action selection from a numerical estimation into a threshold check.

**[ML]:**

**Idea 32: Feature attention / feature grouping.** Group 421 features into semantic blocks (own active, own moves, own bench, opp active, opp bench). Process each group with a small network before combining.

**Idea 33: Action-dependent value function (Q-learning).** Instead of V(s), use Q(s,a). Directly tells the agent "this move is worth X in this state."

**[RBY]:**

**Idea 34: Add "turns of sleep remaining" counter.** Sleep is the most broken mechanic in Gen 1. Currently we just have a float status encoding.

**Idea 35: Encode known opponent moves as threat assessment.** If I've seen the opponent use Thunderbolt, encode that my Water-type is threatened.

---

## Part 3: Voting and Ranking

**[PM]:** Let me ask each expert to vote on their top 5 by impact and feasibility.

**[RL-Debug] top 5:** #9 (1v1 first), #20 (behavioral cloning), #4 (survival reward), #2 (continuous curriculum), #7 (dynamic entropy)

**[RL-Curriculum] top 5:** #2 (continuous curriculum), #9 (1v1 first), #20 (behavioral cloning), #19 (progressive net expansion), #1 (reduced obs)

**[Smogon1] top 5:** #9 (1v1 first), #8 (teach type effectiveness), #10 (subset of Pokemon), #28 (RandomDamage opponent), #20 (behavioral cloning)

**[Smogon2] top 5:** #13 (best move signal), #31 (can-I-KO), #12 (prominent effectiveness), #9 (1v1 first), #30 (speed comparison)

**[ML] top 5:** #20 (behavioral cloning), #15 (larger first layer), #14 (separate policy/value), #32 (feature grouping), #9 (1v1 first)

**[RL] top 5:** #20 (behavioral cloning), #9 (1v1 first), #2 (continuous curriculum), #16 (lower gamma), #14 (separate nets)

**[SE] top 5:** #22 (fix Phase B target), #24 (analyze replays), #9 (1v1 first), #20 (behavioral cloning), #23 (gradient diagnostics)

**[REVIEW] top 5:** #24 (analyze replays first), #22 (fix targets), #28 (RandomDamage opponent), #2 (start softmax at temp 100), #9 (1v1 first)

**[GYM] top 5:** #15 (wider first layer), #14 (separate nets), #9 (1v1 first), #13 (best move signal), #16 (lower gamma)

**[SB3] top 5:** #16 (lower gamma), #17 (tune rollout), #15 (wider first layer), #18 (LR schedule), #14 (separate nets)

---

**[Smogon1]:** Before the ranking — I want to stress that #9 (1v1 first) is the single most important recommendation. In competitive Pokemon, people learn 1v1 matchups before they learn team play. Switching is the hardest skill and introduces massive combinatorial complexity. A 6v6 random battle with 421-dim observations and 10 possible actions is an absurdly hard problem for a [64,64] MLP to solve from scratch. Start with 1v1. It makes everything else 10x easier.

**[RL-Debug]:** I agree. And #20 (behavioral cloning) is almost as important. The agent currently starts with zero knowledge except "attack > switch" (the logit bias). Pre-training on MaxDamagePlayer actions gives it a real starting policy. It'll enter Phase C already knowing "pick the high-damage move" — which is 80% of what SoftmaxDmg does. The RL then only needs to learn the remaining 20% (when to switch, which move when it's close).

---

## Part 4: Final Consensus Ranking

| Rank | # | Idea | Impact | Effort |
|------|---|------|--------|--------|
| 1 | 20 | Behavioral cloning warm-start from MaxDamagePlayer | Very High | Low |
| 2 | 2 | Single continuous curriculum (no phase transitions) | Very High | Low |
| 3 | 9 | Start with 1v1, expand to 6v6 | Very High | Medium |
| 4 | 7/8 | Train move selection in isolation / per-move feedback | High | Low-Medium |
| 5 | 22 | Fix Phase B target (95% -> 75%) | High | Trivial |
| 6 | 15 | Wider first hidden layer [256, 64] | High | Trivial |
| 7 | 14 | Separate policy and value networks | High | Low |
| 8 | 31 | Add "can I KO?" signal per move | High | Low |
| 9 | 30 | Speed comparison feature | Medium-High | Trivial |
| 10 | 16 | Lower gamma for short-episode phases | Medium | Trivial |
| 11 | 28 | Add RandomDamage opponent | Medium-High | Low |
| 12 | 13 | Add "best move" signal to obs | Medium | Low |
| 13 | 7 | Dynamic entropy based on win rate | Medium | Low |
| 14 | 8 | Supervised pre-training on type effectiveness | Medium | Medium |
| 15 | 24 | Inspect losing replays | Medium | Trivial |
| 16 | 4 | Per-turn survival reward | Medium | Low |
| 17 | 10 | Subset of Pokemon initially | Medium | Medium |
| 18 | 12 | Separate larger value function network | Medium | Low |
| 19 | 1 | Reduced obs space for early phases | Medium | Medium |
| 20 | 18 | Reset optimizer on phase transitions | Low-Medium | Trivial |
| 21 | 32 | Feature grouping / lightweight attention | Medium | Medium |
| 22 | 19 | Progressive network growing | Medium | High |
| 23 | 25 | Wider clip range during recovery | Low-Medium | Trivial |
| 24 | 26 | Periodic policy reset, keep value function | Low-Medium | Low |
| 25 | 17 | Reward clipping/normalization | Low-Medium | Low |
| 26 | 23 | Gradient norm diagnostics | Low | Trivial |
| 27 | 33 | Q-learning / action-dependent value | Low-Medium | High |
| 28 | 6 | Intrinsic curiosity | Low | High |
| 29 | 5 | Hindsight action relabeling | Low | High |
| 30 | 27 | Off-policy replay buffer | Low | High |
| 31 | 21 | Population-based training | Low | Very High |
| 32 | 34 | Sleep turns counter | Low | Low |
| 33 | 35 | Threat assessment encoding | Low | Medium |
| 34 | 29 | Phase-dependent sub-networks | Low | High |
| 35 | 11 | Perfect information mode | Low | High |

---

## Part 5: Recommended Implementation Plan

**[PM]:** The panel recommends this order:

### Sprint 1: Quick wins (before next training run)
- Fix Phase B target (95% -> 75%)
- Add KO estimation + speed comparison features to observation
- Add per-move feedback reward for move selection
- Separate policy and value networks
- Lower gamma to 0.95

### Sprint 2: Architecture changes
- Behavioral cloning warm-start from MaxDamagePlayer games
- Single continuous curriculum (eliminate phase transitions)
- Wider first hidden layer [256, 64]

### Sprint 3: Structural changes
- 1v1 training mode
- Add RandomDamage opponent
- Dynamic entropy scaling

### Future
- Progressive network growing
- Feature grouping / attention
- LSTM/GRU policy
