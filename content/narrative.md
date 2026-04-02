# Video Narrative Arc

## Working title
"I trained an AI to play Gen 1 Pokemon — here's what happened"

## Story structure

### Act 1 — Setup & First Heartbeat
The problem: Gen 1 Pokemon is deceptively hard for an AI (partial info, RNG, permanent freeze).
The stack: local Showdown server + Python RL agent.
Payoff: first ever battle completes successfully.

### Act 2 — Does Random Beat Random?
Baseline benchmarking. MaxDamage vs Random.
Tension: if MaxDamage doesn't win 80%+, something is broken.
Payoff: environment confirmed working.

### Act 3 — The Agent Learns Nothing (at first)
Early PPO training. Flat win rate. Why?
Explain sparse rewards, exploration, credit assignment.
Payoff: first sign of learning above random.

### Act 4 — Self-Play Changes Everything
Agent stops improving against fixed opponents.
Self-play: fighting past versions of itself.
Payoff: win rate curve starts climbing again.

### Act 5 — What Did It Actually Learn?
Watch replays. Does it use status moves? Does it switch?
Unexpected behaviors (good and bad).
Payoff: agent does something surprising.

---

## Tone
Technical but accessible. Show the failures, not just the wins.
