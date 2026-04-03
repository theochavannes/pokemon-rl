# Expert Panel Discussion #4: What Comes After Behavioral Cloning?

**Date:** 2026-04-03
**Context:** BC warm-start works (99% accuracy imitating MaxDamagePlayer). Agent starts at ~70-80% vs weak opponents. Reward simplified to faint differential + win/loss. LR lowered to 1e-4. Vol.Switch still 0% — agent never switches voluntarily. Question: what should the agent train against now?

## Participants

**Internal:** [ML], [RL], [SE], [REVIEW], [SYS], [GYM], [SB3], [RBY], [PM], [MEDIA]
**External:**
- [DeepRL] — Principal Research Scientist, DeepMind (AlphaStar, OpenAI Five)
- [Prof-RL] — Professor at CMU, self-play dynamics and population-based training
- [Smogon1] — Top-5 Smogon RBY ladder, 3x tournament winner
- [Smogon2] — RBY teambuilder/analyst, Smogon tiering council

---

## Key Insights from Discussion

### Self-play alone is insufficient
Two MaxDamage clones playing each other never discover switching. Self-play only becomes valuable once the agent already switches, because then it needs to predict the opponent's switches. Diverse opponents are needed to force the full skill set.

### AlphaStar's lesson
Pure self-play produces "cheese strategies" — extreme play that loses to simple counters. AlphaStar solved this with a "league" of diverse opponents including frozen snapshots, exploiter agents, and heuristic baselines.

### The switching gap is critical
Vol.Switch at 0% means the agent has zero defensive skills. Against any opponent that uses type advantages, the agent will hit a wall. Opponents that punish staying in (TypeMatchup, AggressiveSwitcher) provide the evolutionary pressure for switching.

---

## All 35 Ideas (see full conversation for details)

1. League training (AlphaStar-style)
2. Prioritized opponent sampling
3. Exploiter agents
4. Population-based self-play
5. Fictitious self-play (FSP)
6. PFSP (Prioritized Fictitious Self-Play)
7. Nash averaging
8. Switching teacher opponent
9. Download external bots
10. Fixed matchup drills
11. Strategy-specific opponents (SleepLead, WrapTrapper, etc.)
12. Matchup-aware training (only update on fair matchups)
13. Mixed opponent pool (4 envs, 4 opponents)
14. Tournament gating for checkpoints
15. Switch reward bonus
16. Switch observation features
17. Two-headed attack/switch policy
18. KL penalty against BC policy
19. BC safety net (revert on regression)
20. Asymmetric self-play
21. Automated evaluation gauntlet
22. Replay analysis pipeline
23. Curriculum over opponent skills
24. Hindsight self-play
25. BC variants (MaxDmg, TypeMatch, Stall)
26. Elo-based matchmaking
27. Multi-objective reward
28. Commentary-based learning
29. Pre-computed optimal switches
30. Metagame simulation
31. Train on human replays
32. Sleep clause exploitation training
33. Endgame training
34. Progressive complexity
35. BC regression test

---

## Consensus Ranking (top 15)

| Rank | # | Idea | Impact | Effort | Votes |
|------|---|------|--------|--------|-------|
| 1 | 13 | Mixed opponent pool (4 envs, 4 opponents) | Very High | Trivial | 10/10 |
| 2 | 34 | Progressive complexity (validate each step) | Very High | Low | 6/10 |
| 3 | 18 | KL penalty against BC policy | High | Low | 4/10 |
| 4 | 21 | Automated evaluation gauntlet | High | Low | 3/10 |
| 5 | 8 | Switching teacher opponent | High | Low | 3/10 |
| 6 | 23 | Curriculum over opponent skills | High | Medium | 3/10 |
| 7 | 35 | BC regression test | Medium-High | Trivial | 3/10 |
| 8 | 5 | Fictitious self-play (opponent history) | High | Medium | 3/10 |
| 9 | 15 | Switch reward bonus | Medium-High | Trivial | 2/10 |
| 10 | 25 | BC variants (train on different heuristics) | High | Medium | 2/10 |
| 11 | 26 | Elo-based matchmaking | High | Medium | 2/10 |
| 12 | 16 | Switch observation features | Medium-High | Low | 2/10 |
| 13 | 11 | Strategy-specific opponents | Medium-High | Medium | 2/10 |
| 14 | 19 | BC safety net (revert on regression) | Medium | Low | 2/10 |
| 15 | 2 | Prioritized opponent sampling | High | Medium | 1/10 |

---

## Recommended Implementation Plan

### Sprint 3A — Mixed Opponent Pool (implement now)
- Env 0: Self-play (frozen BC model initially)
- Env 1: MaxDamagePlayer
- Env 2: TypeMatchupPlayer
- Env 3: SoftmaxDamagePlayer (temp annealing)
- No phase transitions. One continuous training run.
- Add KL penalty against BC policy (beta=0.1)
- Add BC regression test every 50K steps

### Sprint 3B — Switching Pressure (if vol.switch still 0% after 100K)
- Add switching teacher opponent
- Small switch reward bonus
- Add switch observation features

### Sprint 3C — Scaling Up (once agent beats all heuristics >60%)
- Fictitious self-play with checkpoint history
- Elo-based matchmaking
- Automated evaluation gauntlet
