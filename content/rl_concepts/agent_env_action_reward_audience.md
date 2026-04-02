# The Agent, the World, and How It Learns — Audience Version

For YouTube viewers. No RL background assumed. Uses analogies.

---

## The Agent — "The Brain"

Think of the agent as a **player who has never seen Pokemon before**.

It has no idea what Thunderbolt does. It doesn't know that Fire beats Grass.
It doesn't know what HP means. It starts completely blank.

All it can do is:
1. Look at a snapshot of the battle (64 numbers)
2. Pick one of up to 10 buttons to press
3. See what happened
4. Slowly figure out which buttons led to winning

That's it. No rules given. No strategy book. Just trial and error, millions of times.

```
What the agent sees:           What the agent does:
64 numbers describing          Picks 1 of 10 actions
the current battle             (move 1/2/3/4 or switch to one of 6 Pokemon)
```

---

## The Environment — "The Game"

The environment is everything the agent plays **against and within**.

In our case: a real Pokemon battle running on Pokemon Showdown,
the same simulator used by competitive players worldwide.

The environment is responsible for:
- Running the actual battle (damage, accuracy, speed, crits)
- Playing as the opponent
- Telling the agent what happened after each move
- Deciding when the battle is over

The agent never sees the game's source code. It only sees numbers in, numbers out.

```
Agent says: "I choose move 3"
                │
                ▼
        Pokemon Showdown
        runs the turn:
        "Starmie used Surf!
         It's super effective!
         Rhydon fainted!"
                │
                ▼
Agent receives: new snapshot of the battle
```

---

## An Action — "Pressing a Button"

Each turn, the agent picks **one number from 0 to 9**.

```
0–5  →  Switch to a different Pokemon
6–9  →  Use one of your 4 moves
```

That's the entire decision space. 10 buttons.

**But not all buttons are available every turn.**
You can't switch to a fainted Pokemon. You can't use a move with no PP left.
The game automatically greys out the illegal buttons so the agent only
ever chooses from what's actually possible.

This is called **action masking** — it prevents the agent from wasting
time learning "don't press buttons that don't work."

---

## The Reward — "The Score"

The reward is the only feedback the agent gets about whether it's doing well.

```
Win the battle  →  +1
Lose the battle →  −1
Every other turn →   0
```

That's it. Dead simple. One point for winning, minus one for losing,
nothing in between.

**Why so simple?**

Because we don't want to accidentally teach the agent the wrong thing.
If we gave it +0.1 every time it dealt damage, it might learn to spam
the highest-damage move every turn — regardless of whether it's smart
strategy. Sparse reward forces it to figure out what actually leads to winning.

**The catch:** With zero feedback for 40 turns and then a single ±1 at the end,
the agent has to work backwards and figure out which of those 40 moves
actually mattered. This is called the **credit assignment problem** — and it's
one of the core reasons RL is hard.

```
Turn 1  2  3  4 ... 38  39  40
Score:  0  0  0  0 ...  0   0  +1

"Something I did in those 40 turns caused me to win.
 But which moves actually mattered?"
```

The neural network gradually gets better at answering that question —
by playing thousands of battles and noticing patterns.

---

## Why Does This Work At All?

The agent starts by pressing random buttons.
Sometimes it randomly wins. Sometimes it randomly loses.

Over thousands of battles, it starts noticing:
*"When I pressed button 8 (Thunderbolt) against that Pokemon, I usually won.
 When I pressed button 6 (Scratch) against that Pokemon, I usually lost."*

It nudges its probabilities: press Thunderbolt more, press Scratch less.
Repeat. A million times.

After enough battles, it's not pressing random buttons anymore.
It's pressing the right ones — not because someone told it to,
but because it figured it out from experience.
