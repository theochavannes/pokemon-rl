# Sprint 5: Observation Completeness + Feature Engineering

Priority: HIGH — Ranked items #4-6, #9-10, #16-17, #20 from panel #5.
The agent is missing critical game state information that prevents strategic play.

---

## Step 1: Secondary effect TYPE encoding (+1 per move)

Currently we encode secondary_chance (Body Slam = 0.3) but NOT what the secondary does.

Add `secondary_effect_type` per move:
- Body Slam: secondary_chance=0.3, secondary_effect_type=0.8 (PAR)
- Blizzard: secondary_chance=0.1, secondary_effect_type=1.0 (FRZ)
- Psychic: secondary_chance=0.33, secondary_effect_type=-0.17 (SpA drop = -1/6)

Source: `move.entry["secondary"]["status"]` or `move.entry["secondary"]["boosts"]`

---

## Step 2: Target already has status flag (+1 per own move)

Prevents wasting Thunder Wave on already-paralyzed opponents.

```python
target_statused = 1.0 if (move.status and opponent.status) else 0.0
```

Only needed for own active moves (4 dims), not bench or opponent moves.

---

## Step 3: Volatile status features (+8 global dims)

| Feature | Game impact | poke-env source |
|---------|------------|-----------------|
| own_substitute | Blocks status moves | Check Pokemon effects |
| opp_substitute | Same | |
| own_reflect | Halves physical damage | `battle.side_conditions` |
| own_light_screen | Halves special damage | `battle.side_conditions` |
| own_confused | 50% self-hit chance | Check Pokemon effects |
| opp_confused | Same | |
| own_leech_seeded | HP drain each turn | Check Pokemon effects |
| opp_leech_seeded | Same | |

---

## Step 4: Opponent status move threat (+1 global dim)

Flag indicating opponent has revealed status moves (Thunder Wave, Sleep Powder). Increases switching urgency.

---

## Step 5: Toxic escalation counter (+1 global dim)

Normalized count of turns Toxic has been active (0-15 -> 0-1). Determines healing urgency.
May not be exposed by poke-env — skip if unavailable.

---

## Step 6: One-hot move category (+2 per move, replacing float)

Replace `category` float (0.0/0.5/1.0) with `[is_physical, is_special]` (Status = both 0).
Removes false ordering. Net +1 dim per move slot.

---

## Step 7: Separate heal vs drain (+1 per move)

Split `heal_val` into `heal` (Recover = 0.5) and `drain` (Mega Drain = 0.5).
They work completely differently: heal is %maxHP, drain is %damage.

---

## Step 8: Trapping move flag (+1 per move)

Wrap/Bind/Clamp/Fire Spin trap opponent 2-5 turns in Gen 1, preventing ALL actions.
One of the most broken Gen 1 mechanics, currently invisible.

Source: Check move flags or maintain a list of known trapping move IDs.

---

## Step 9: Status immunity flag (+1 per own move)

The agent doesn't know that Poison types are immune to Toxic, Fire types immune to burn, etc. Showdown blocks the move silently — wasted turn.

Gen 1 status immunities:
- **Poison/Toxic**: Poison types immune
- **Burn**: Fire types immune
- **Freeze**: Ice types immune
- **Thunder Wave**: Ground types immune (Electric-type move, type immunity)
- **Paralysis (via Body Slam etc.)**: No immunity in Gen 1 (Electric immunity added Gen 6+)
- **Sleep**: No type immunity (Sleep Clause limits to 1 sleeping mon, tracked by Showdown)

Encode as `status_immune = 1.0` if the move's status effect would fail against the target's type.

```python
def _status_immune(move, target) -> float:
    """Check if target's type makes it immune to this move's status effect."""
    if not move.status:
        return 0.0
    status = str(move.status)
    t1 = target.type_1
    t2 = target.type_2
    types = [t for t in [t1, t2] if t is not None]
    type_names = [str(t).split()[0].upper() for t in types]
    if status.startswith("PSN") or status.startswith("TOX"):
        if "POISON" in type_names:
            return 1.0
    elif status.startswith("BRN"):
        if "FIRE" in type_names:
            return 1.0
    elif status.startswith("FRZ"):
        if "ICE" in type_names:
            return 1.0
    # Thunder Wave: handled by effectiveness=0 for Ground types
    return 0.0
```

Also update SmartHeuristicPlayer to check immunity before using status moves.

---

## Dimension estimate

~928 + ~75 new dims = ~1003 dims. Exact count depends on which steps are feasible.

**Breaking change:** Must retrain BC from scratch (mid-vector feature insertion).

---

## Verification

1. Tests pass with updated shapes
2. Spot-check: Thunder Wave vs statused opponent shows target_statused=1
3. Spot-check: Toxic vs Poison-type opponent shows status_immune=1
3. Benchmark SmartHeuristic: >10% status move usage in BC data
