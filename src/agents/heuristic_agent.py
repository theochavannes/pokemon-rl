"""
Heuristic baseline agents for Gen 1 (RBY) random battles.

Each agent tests a different skill the RL agent needs to learn:
  - MaxDamagePlayer:     raw damage output → agent must learn to take hits efficiently
  - TypeMatchupPlayer:   smart switching → agent must handle type-aware opponents
  - StallPlayer:         status + bulk → agent must learn to break through walls
  - AggressiveSwitcher:  type counters → agent must predict switches and punish them
  - EpsilonMaxDamagePlayer: smooth curriculum bridge (Random ↔ MaxDamage)
"""

import random as _random

from poke_env.battle.move_category import MoveCategory
from poke_env.data import GenData
from poke_env.player.baselines import RandomPlayer  # re-exported for convenience
from poke_env.player.player import Player

_GEN1_TYPE_CHART = GenData.from_gen(1).type_chart


def _expected_damage(move, opponent) -> float:
    """Proxy for expected damage: base_power × type_effectiveness."""
    if move.base_power == 0:
        return 0.0
    try:
        effectiveness = move.type.damage_multiplier(
            opponent.type_1, opponent.type_2, type_chart=_GEN1_TYPE_CHART
        )
    except Exception:
        effectiveness = 1.0
    return move.base_power * effectiveness


def _type_advantage(pokemon, opponent) -> float:
    """How well does pokemon's typing resist opponent's types? Higher = better matchup."""
    score = 0.0
    for def_type in [pokemon.type_1, pokemon.type_2]:
        if def_type is None:
            continue
        for atk_type in [opponent.type_1, opponent.type_2]:
            if atk_type is None:
                continue
            try:
                mult = atk_type.damage_multiplier(def_type, None, type_chart=_GEN1_TYPE_CHART)
                score -= mult  # lower mult = better resistance
            except Exception:
                pass
    return score


# ---------------------------------------------------------------------------
# MaxDamagePlayer — always picks highest-damage move, never switches
# Skill tested: agent must survive raw damage output
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# RandomAttackerPlayer — random move 85% of the time, random switch 15%
# More realistic than pure Random (which switches 60% due to action space)
# ---------------------------------------------------------------------------

class RandomAttackerPlayer(Player):
    def choose_move(self, battle):
        if battle.available_moves and _random.random() < 0.85:
            return self.create_order(_random.choice(battle.available_moves))
        return self.choose_random_move(battle)


# ---------------------------------------------------------------------------
# SoftmaxDamagePlayer — samples moves proportional to damage^(1/temperature)
# High temp = near-uniform, low temp = near-argmax (MaxDamage)
# Switches randomly ~10% of turns. Temperature is mutable for annealing.
# ---------------------------------------------------------------------------

class SoftmaxDamagePlayer(Player):
    def __init__(self, temperature: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature

    def choose_move(self, battle):
        # Small chance to switch randomly (realistic but exploitable)
        if battle.available_switches and _random.random() < 0.10:
            return self.create_order(_random.choice(battle.available_switches))

        if battle.available_moves:
            opp = battle.opponent_active_pokemon
            damages = []
            for m in battle.available_moves:
                d = _expected_damage(m, opp)
                damages.append(max(d, 10.0))  # floor for status moves

            # Softmax with temperature
            powered = [d ** (1.0 / max(self.temperature, 0.01)) for d in damages]
            total = sum(powered)
            probs = [p / total for p in powered]

            chosen = _random.choices(battle.available_moves, weights=probs, k=1)[0]
            return self.create_order(chosen)

        if battle.available_switches:
            return self.create_order(
                max(battle.available_switches, key=lambda p: p.current_hp_fraction)
            )
        return self.choose_random_move(battle)


class MaxDamagePlayer(Player):
    def choose_move(self, battle):
        if battle.available_moves:
            return self.create_order(max(
                battle.available_moves,
                key=lambda m: _expected_damage(m, battle.opponent_active_pokemon),
            ))
        if battle.available_switches:
            return self.create_order(
                max(battle.available_switches, key=lambda p: p.current_hp_fraction)
            )
        return self.choose_random_move(battle)


# ---------------------------------------------------------------------------
# TypeMatchupPlayer — picks best type-effective move, switches if walled
# Skill tested: agent faces a type-aware opponent that won't stay in bad matchups
# ---------------------------------------------------------------------------

class TypeMatchupPlayer(Player):
    def choose_move(self, battle):
        opp = battle.opponent_active_pokemon

        # If we have a super-effective move, use the strongest one
        if battle.available_moves:
            best = max(battle.available_moves, key=lambda m: _expected_damage(m, opp))
            if _expected_damage(best, opp) > 0:
                return self.create_order(best)

        # No good damage move — switch to a better type matchup if possible
        if battle.available_switches:
            best_switch = min(
                battle.available_switches,
                key=lambda p: _type_advantage(p, opp),
            )
            if _type_advantage(best_switch, opp) < _type_advantage(battle.active_pokemon, opp):
                return self.create_order(best_switch)

        # Fall through: use any available move
        if battle.available_moves:
            return self.create_order(max(
                battle.available_moves, key=lambda m: _expected_damage(m, opp),
            ))

        return self.choose_random_move(battle)


# ---------------------------------------------------------------------------
# StallPlayer — prioritizes status moves, switches to bulkiest Pokemon
# Skill tested: agent must learn to break through status + bulk
# ---------------------------------------------------------------------------

class StallPlayer(Player):
    _STATUS_CATEGORIES = {"status"}

    def choose_move(self, battle):
        opp = battle.opponent_active_pokemon

        # Prioritize status moves if opponent isn't already statused
        if battle.available_moves and opp.status is None:
            status_moves = [m for m in battle.available_moves if m.category == MoveCategory.STATUS]
            if status_moves:
                return self.create_order(_random.choice(status_moves))

        # Otherwise pick highest damage move
        if battle.available_moves:
            return self.create_order(max(
                battle.available_moves, key=lambda m: _expected_damage(m, opp),
            ))

        # Switch to bulkiest (highest HP) Pokemon
        if battle.available_switches:
            return self.create_order(
                max(battle.available_switches, key=lambda p: p.current_hp_fraction)
            )

        return self.choose_random_move(battle)


# ---------------------------------------------------------------------------
# AggressiveSwitcher — switches to type advantage aggressively, then attacks
# Skill tested: agent must handle frequent switching and punish it
# ---------------------------------------------------------------------------

class AggressiveSwitcher(Player):
    def choose_move(self, battle):
        opp = battle.opponent_active_pokemon

        # Switch if a benched Pokemon has a much better type matchup
        if battle.available_switches:
            current_adv = _type_advantage(battle.active_pokemon, opp)
            best_switch = min(battle.available_switches, key=lambda p: _type_advantage(p, opp))
            # Switch if the bench mon resists significantly better
            if _type_advantage(best_switch, opp) < current_adv - 1.0:
                return self.create_order(best_switch)

        # Attack with best move
        if battle.available_moves:
            return self.create_order(max(
                battle.available_moves,
                key=lambda m: _expected_damage(m, opp),
            ))

        if battle.available_switches:
            return self.create_order(
                max(battle.available_switches, key=lambda p: p.current_hp_fraction)
            )

        return self.choose_random_move(battle)


# ---------------------------------------------------------------------------
# Epsilon wrappers — per-turn epsilon-greedy for any opponent
# ---------------------------------------------------------------------------

class _EpsilonMixin:
    """Mixin: with probability epsilon, play as frozen self; otherwise play strategy.

    When a frozen model is loaded (via load_selfplay_model), the epsilon turns
    use the agent's own policy instead of random. This means the opponent is
    always a blend of 'current agent skill' and 'pure heuristic strategy' —
    no random once training starts.

    Falls back to random if no frozen model is loaded yet.
    """

    def __init__(self, epsilon: float = 0.8, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self._frozen_model = None

    def load_selfplay_model(self, path: str) -> None:
        from sb3_contrib import MaskablePPO
        self._frozen_model = MaskablePPO.load(path)

    def swap_model(self, path: str) -> None:
        """Hot-swap frozen model (called by callback)."""
        self.load_selfplay_model(path)

    def _choose_selfplay_move(self, battle):
        if self._frozen_model is None:
            return self.choose_random_move(battle)
        import numpy as np
        from poke_env.environment.singles_env import SinglesEnv
        from src.env.gen1_env import embed_battle
        obs = embed_battle(battle)
        mask = np.array(SinglesEnv.get_action_mask(battle), dtype=bool)
        action, _ = self._frozen_model.predict(obs, action_masks=mask, deterministic=False)
        return SinglesEnv.action_to_order(action, battle, strict=False)

    def choose_move(self, battle):
        if _random.random() < self.epsilon:
            return self._choose_selfplay_move(battle)
        return super().choose_move(battle)


class EpsilonMaxDamagePlayer(_EpsilonMixin, MaxDamagePlayer):
    pass

class EpsilonTypeMatchupPlayer(_EpsilonMixin, TypeMatchupPlayer):
    pass

class EpsilonStallPlayer(_EpsilonMixin, StallPlayer):
    pass

class EpsilonAggressiveSwitcher(_EpsilonMixin, AggressiveSwitcher):
    pass


__all__ = [
    "MaxDamagePlayer",
    "TypeMatchupPlayer",
    "StallPlayer",
    "AggressiveSwitcher",
    "EpsilonMaxDamagePlayer",
    "EpsilonTypeMatchupPlayer",
    "EpsilonStallPlayer",
    "EpsilonAggressiveSwitcher",
    "RandomPlayer",
]
