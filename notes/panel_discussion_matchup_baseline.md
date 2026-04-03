# Expert Panel Discussion: Privileged Matchup Evaluator for Reward Scaling

**Date:** 2026-04-03
**Context:** Agent plateaus at ~90% vs Random in Phase A. Training doesn't improve beyond initial logit bias. Root cause suspected: team RNG variance drowns learning signal.

**Participants:**
- **[ML]** Machine Learning Engineer
- **[RL]** Staff ML Researcher — RL Theory
- **[SE]** Staff Engineer
- **[REVIEW]** Principal Engineer (Devil's Advocate)
- **[RBY]** Gen 1 Competitive Expert
- **[MEDIA]** Content Director
- **[PM]** Technical PM
- **External: [DeepRL]** — Principal RL Research Scientist, DeepMind background
- **External: [Meta-RL]** — Staff ML Engineer, Meta FAIR, specializes in game-playing agents
- **External: [Prof-RL]** — Professor of RL, UC Berkeley, published on variance reduction in policy gradients
- **External: [Smogon-RBY]** — Top-10 Smogon RBY ladder player, tournament winner

---

**[PM]:** The proposal is to build a privileged evaluator that sees both full teams at battle start and estimates win probability. This scales the terminal reward so the agent gets more credit for winning hard matchups and more blame for losing easy ones. The goal is to reduce variance from team RNG, which we believe is capping learning in Phase A. Floor is open.

**[Prof-RL]:** This is a well-studied idea. What you're describing is an **episode-level control variate** — you're subtracting a baseline from the return to reduce variance without changing the expected gradient. The classic form is:

```
adjusted_return = R - b(context)
```

where `b` is your matchup estimator and `context` is the privileged team info. If `b` is accurate, this can dramatically reduce variance. The theory is clean — any baseline that doesn't depend on the agent's actions leaves the policy gradient unbiased.

**[DeepRL]:** Agreed on the theory. But I want to flag a practical distinction. The proposal isn't just a baseline subtraction — it's *multiplicative* reward scaling ("more reward if we win while expected to lose"). Those are different things with different properties.

**Additive baseline** (subtract expected return): unbiased, variance reduction only, well-understood.

**Multiplicative scaling** (scale reward by surprise): changes the optimization objective. You're no longer maximizing win rate — you're maximizing "surprising wins." An agent could learn to play conservatively in favorable matchups (low reward anyway) and gamble in unfavorable ones (high reward if it pays off). That might not be what you want.

**[Prof-RL]:** That's an important distinction. I'd strongly recommend the **additive form**. Subtract the expected return, don't multiply. The math is:

```
reward_adjusted = victory_value - matchup_baseline(team_a, team_b)
```

This preserves the policy gradient's unbiasedness while reducing variance. The agent still tries to win every game — it just gets cleaner gradient signals.

**[RL]:** I want to connect this to what PPO already does. The GAE advantage is `A(s,a) = R - V(s)`, which is already subtracting a baseline. The problem Theo identified is that V(s) at turn 1 can't see the full teams — it only sees two active Pokemon. So V(s) is a poor baseline for the team-matchup component of variance.

An episode-level privileged baseline would capture exactly the variance V(s) misses. They're complementary, not redundant.

**[Meta-RL]:** At FAIR we did something similar for Diplomacy — the agent's critic had access to private information that the policy didn't. It's called an **asymmetric actor-critic**. The actor sees only legal observations; the critic sees privileged state. This is cleaner than hacking the reward because it integrates naturally with PPO's value function.

Concretely: you'd modify the value network to take a separate input — the full team features — while the policy network stays on the 421-dim observation. The value function learns "this matchup is worth +0.3 on average" and PPO's advantage automatically adjusts.

**[REVIEW]:** Hold on. Both approaches — reward baseline and asymmetric critic — need an accurate matchup estimator. How good does it need to be? And what happens when it's wrong?

**[Prof-RL]:** A bad baseline increases variance rather than reducing it. If your estimator says "easy matchup, you should win" but it's actually hard, the agent gets a large negative signal for a reasonable loss. Over many episodes this washes out (the gradient is still unbiased), but convergence slows. The estimator doesn't need to be perfect, but it needs to be better than nothing — i.e., its predictions should correlate with actual outcomes.

**[Smogon-RBY]:** Let me give you the competitive perspective on what makes a team good in Gen 1 randoms.

First — **the matchup is NOT just type charts.** In RBY, a few things dominate:

1. **Speed tiers matter enormously.** If I have Tauros, Starmie, Jolteon and you have Slowbro, Exeggutor, Rhydon — I'm probably faster across the board. Speed = who attacks first = who gets the KO before taking damage. A heuristic that just counts type matchups would completely miss this.

2. **Sleep is the most broken mechanic.** If one team has a reliable sleep lead (Exeggutor with Sleep Powder, Jynx with Lovely Kiss) and the other doesn't, that's a massive advantage. Sleep is permanent in Gen 1 — that Pokemon is gone. A single sleep move can swing the whole game.

3. **Coverage moves matter more than STAB.** Alakazam doesn't win because of Psychic STAB — it wins because the things that resist Psychic in Gen 1 are... nothing, really, since the Ghost immunity is bugged. What matters is whether your team has the right coverage to break through the opponent's defensive Pokemon.

4. **Chansey/Snorlax are structural.** If one side has a special wall (Chansey) and the other is all special attackers, the Chansey side has a huge edge. The heuristic needs to understand which Pokemon are walls and which are threats.

5. **In random battles specifically**, the biggest factor is often just **who gets the better individual Pokemon.** Tauros, Snorlax, Chansey, Starmie, Exeggutor, Alakazam — these are tier 1. If you roll three of them and your opponent rolls none, you're heavily favored regardless of type matchups.

So if you're building a matchup estimator, tier lists and speed tiers matter more than type charts.

**[ML]:** That's really valuable. So a heuristic estimator would need: tier ratings per Pokemon, speed comparisons, sleep move detection, and special vs physical balance. Not just type effectiveness.

**[Smogon-RBY]:** Right. And honestly? I'd start even simpler. Just **sum the tier ratings** of both teams. In Gen 1 randoms, team quality correlates heavily with how many OU-tier Pokemon you rolled. You could build a lookup table of ~150 Gen 1 Pokemon → tier score (1-5 scale) and sum each team. That alone would capture maybe 60% of matchup variance.

**[Meta-RL]:** I like that. Start with a dead-simple heuristic baseline — sum of tier scores — and see if it reduces variance enough to help learning. If it does, you can refine the estimator later. If it doesn't help even with a reasonable estimator, the problem is elsewhere.

**[SE]:** Implementation-wise, where does this baseline get computed? The `reward_computing_helper` runs inside `Gen1Env.calc_reward()`, which only sees the battle object. The privileged team info is available in the battle object (`battle.team` and `battle.opponent_team`) but `opponent_team` only contains revealed Pokemon during play.

Wait — actually, in poke-env, at the very start of a gen1randombattle, do we see the opponent's full team?

**[RBY]:** No. Gen 1 random battles have no team preview. You see your 6 and their lead. Their bench is revealed only as they switch in.

**[Smogon-RBY]:** Correct. And this is actually a problem for any privileged evaluator. You need Showdown server-side data to see the opponent's full team before the battle starts. The agent never sees it through poke-env.

**[SE]:** So we'd need to either: (a) hook into Showdown's battle object to extract both teams at battle start, or (b) compute the baseline retroactively — once the battle is over, we've seen all opponent Pokemon that were sent out (usually all 6), and we can compute the matchup score then.

**[REVIEW]:** Option (b) has a flaw: if you won in 3 KOs and they only sent out 3 Pokemon, you don't know the other 3. The baseline would be based on partial information.

**[Meta-RL]:** Option (b) is fine statistically. You use whatever info is available at episode end. Even a partial-information baseline reduces variance if it's correlated with difficulty. And most games do reveal 5-6 Pokemon per side.

**[DeepRL]:** I want to propose an alternative that sidesteps the estimator entirely: **reward normalization with a running mean/std**. Instead of predicting the expected return per matchup, you just normalize all returns globally:

```
normalized_return = (R - running_mean) / running_std
```

This is much simpler, requires no privileged info, and automatically adjusts as the agent improves. It doesn't capture per-matchup variance as precisely, but it handles the "some games are worth more reward than others" problem. SB3 already supports this with `normalize_reward=True` in VecNormalize.

**[Prof-RL]:** That's a reasonable first step. Per-episode normalization captures the macro variance (am I generally winning or losing?) but doesn't capture the matchup-specific variance (this particular game was hard/easy). The privileged baseline is strictly more powerful in theory, but reward normalization is free.

**[RL]:** Are we already using VecNormalize?

**[ML]:** No. The env stack is `Gen1Env → SingleAgentWrapper → SB3Wrapper → Monitor → DummyVecEnv`. No VecNormalize.

**[DeepRL]:** That might be low-hanging fruit then. Before building a privileged evaluator, try wrapping the env in `VecNormalize(env, norm_obs=True, norm_reward=True)`. Observation normalization alone often helps significantly with MLPs.

**[REVIEW]:** Summary so far — we have three proposals on the table:

1. **Privileged matchup baseline (original idea)**: Episode-level reward adjustment based on team quality. Additive, not multiplicative. Needs an estimator (start with tier-score heuristic). Requires seeing opponent team (either via Showdown hook or retroactively).

2. **Asymmetric actor-critic**: Privileged info feeds the value function, not the reward. Cleaner theoretically. But requires modifying SB3's PPO architecture — custom policy class.

3. **VecNormalize (reward + obs normalization)**: Zero-effort baseline. Doesn't capture per-matchup variance but normalizes the overall reward scale. Try this first.

**[Smogon-RBY]:** I want to add one more thought. In competitive play, the best players don't just evaluate matchup at team preview — they **re-evaluate every turn** as information is revealed. "Oh, their Starmie has Thunder Wave instead of Thunderbolt — that completely changes how I play this." A static start-of-game baseline misses this, but it's still valuable for the team-RNG component.

**[Meta-RL]:** Good point. The static baseline handles "I got a bad team" variance. The per-turn re-evaluation is what the value function should learn. They target different variance sources.

**[MEDIA]:** This is a great content moment. "We brought in RL professors, competitive Pokemon players, and engineers from DeepMind and Meta — and they argued about whether a 30-year-old Game Boy game needs the same reward shaping techniques used to train Diplomacy AI." The split between "what the agent sees" vs "what a privileged evaluator knows" is very visual — you can show the two perspectives side by side. And the Smogon player correcting the academics about what actually matters in RBY is gold.

**[PM]:** Here's where I think we land:

**Recommended approach — phased:**

1. **Now (free):** Add `VecNormalize` with obs + reward normalization. See if it helps Phase A convergence.
2. **Next (low effort):** Build a tier-score heuristic. Sum tier ratings for both teams at episode end using revealed Pokemon. Subtract from terminal reward as an additive baseline.
3. **Later (if needed):** Asymmetric actor-critic with privileged team features fed to the value network. Only if (1) and (2) don't sufficiently reduce variance.

**Key design decisions:**
- **Additive baseline, not multiplicative** — preserves policy gradient unbiasedness
- **Tier-score heuristic first** — [Smogon-RBY] says raw Pokemon quality is the biggest matchup factor in randoms
- **Retroactive computation** — use revealed Pokemon at episode end, no Showdown hooks needed

**Status:** Discussion complete. Awaiting user decision on which step to pursue.
