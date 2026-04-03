"""
Heuristic baseline agents for Gen 1 (RBY) random battles.

Each agent tests a different skill the RL agent needs to learn:
  - MaxDamagePlayer:     raw damage output → agent must learn to take hits efficiently
  - TypeMatchupPlayer:   smart switching → agent must handle type-aware opponents
  - StallPlayer:         status + bulk → agent must learn to break through walls
  - AggressiveSwitcher:  type counters → agent must predict switches and punish them
  - EpsilonMaxDamagePlayer: smooth curriculum bridge (Random ↔ MaxDamage)
"""

import contextlib
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
        effectiveness = move.type.damage_multiplier(opponent.type_1, opponent.type_2, type_chart=_GEN1_TYPE_CHART)
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
            except Exception:  # noqa: S110
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
        if battle.available_moves and _random.random() < 0.95:
            return self.create_order(_random.choice(battle.available_moves))
        return self.choose_random_move(battle)


# ---------------------------------------------------------------------------
# RandomDamagePlayer — picks a random MOVE (never switches voluntarily)
# Sits between RandomPlayer (60% switch) and SoftmaxDamagePlayer (smart damage)
# ---------------------------------------------------------------------------


class RandomDamagePlayer(Player):
    """Picks a random move every turn, never switches voluntarily.

    Tests if the agent can learn move selection when the opponent
    isn't also picking smart moves.
    """

    def choose_move(self, battle):
        if battle.available_moves:
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
            return self.create_order(max(battle.available_switches, key=lambda p: p.current_hp_fraction))
        return self.choose_random_move(battle)


class MaxDamagePlayer(Player):
    def choose_move(self, battle):
        if battle.available_moves:
            return self.create_order(
                max(
                    battle.available_moves,
                    key=lambda m: _expected_damage(m, battle.opponent_active_pokemon),
                )
            )
        if battle.available_switches:
            return self.create_order(max(battle.available_switches, key=lambda p: p.current_hp_fraction))
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
            return self.create_order(
                max(
                    battle.available_moves,
                    key=lambda m: _expected_damage(m, opp),
                )
            )

        return self.choose_random_move(battle)


# ---------------------------------------------------------------------------
# StallPlayer — prioritizes status moves, switches to bulkiest Pokemon
# Skill tested: agent must learn to break through status + bulk
# ---------------------------------------------------------------------------


class StallPlayer(Player):
    def choose_move(self, battle):
        opp = battle.opponent_active_pokemon

        # Prioritize status moves if opponent isn't already statused
        if battle.available_moves and opp.status is None:
            status_moves = [m for m in battle.available_moves if m.category == MoveCategory.STATUS]
            if status_moves:
                return self.create_order(_random.choice(status_moves))

        # Otherwise pick highest damage move
        if battle.available_moves:
            return self.create_order(
                max(
                    battle.available_moves,
                    key=lambda m: _expected_damage(m, opp),
                )
            )

        # Switch to bulkiest (highest HP) Pokemon
        if battle.available_switches:
            return self.create_order(max(battle.available_switches, key=lambda p: p.current_hp_fraction))

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
            return self.create_order(
                max(
                    battle.available_moves,
                    key=lambda m: _expected_damage(m, opp),
                )
            )

        if battle.available_switches:
            return self.create_order(max(battle.available_switches, key=lambda p: p.current_hp_fraction))

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


# ---------------------------------------------------------------------------
# SmartHeuristicPlayer — competitive-informed BC teacher
# Combines MaxDamage move selection with strategic switching AND status moves.
# Rules from Gen 1 competitive analysis (Smogon RBY experts):
#   1. Use status moves strategically (Thunder Wave, Swords Dance, Recover, Toxic)
#   2. Switch when walled or at severe type disadvantage
#   3. Switch into resistances, skip low-HP targets
#   4. Cap switching at ~20% of turns
#   5. Pick highest damage move otherwise (MaxDamage-style)
# ---------------------------------------------------------------------------


def _is_status_move(move) -> bool:
    """Check if a move is a status-category move."""
    return move.category == MoveCategory.STATUS


def _should_use_status(move, battle) -> bool:
    """Decide if a status move is worth using right now."""
    opp = battle.opponent_active_pokemon
    own = battle.active_pokemon

    # Thunder Wave / Stun Spore: paralyze if opponent isn't already statused
    # and opponent is faster (paralyze to gain speed advantage)
    if move.status and str(move.status).startswith("PAR") and opp.status is None:
        opp_spe = opp.base_stats.get("spe", 0)
        own_spe = own.base_stats.get("spe", 0)
        if opp_spe >= own_spe or opp.base_stats.get("atk", 0) > 120 or opp.base_stats.get("spa", 0) > 120:
            return True

    # Toxic / Poison: use against bulky opponents (high HP) that aren't already statused
    if move.status and str(move.status).startswith("TOX") and opp.status is None and opp.current_hp_fraction > 0.6:
        return True

    # Sleep moves (Hypnosis, Sleep Powder, Sing): always valuable if opponent not statused
    if move.status and str(move.status).startswith("SLP") and opp.status is None:
        return True

    # Swords Dance / stat boosts: use if we have HP to spare and good matchup
    with contextlib.suppress(Exception):
        boosts = move.self_boost or (move.boosts if _is_status_move(move) else None)
        if (
            boosts
            and any(v > 0 for v in boosts.values())
            and own.current_hp_fraction > 0.7
            and _type_advantage(own, opp) <= 0
        ):
            max_current_boost = max(own.boosts.get(k, 0) for k in boosts if boosts[k] > 0)
            if max_current_boost < 4:
                return True

    # Recovery (Recover, Softboiled, Rest): heal if below 50% HP
    with contextlib.suppress(Exception):
        if (
            (move.heal and move.heal > 0) or move.id in ("recover", "softboiled", "rest")
        ) and own.current_hp_fraction < 0.50:
            return True

    return False


class SmartHeuristicPlayer(Player):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._turn_count: int = 0
        self._switch_count: int = 0

    def choose_move(self, battle):
        self._turn_count += 1
        opp = battle.opponent_active_pokemon
        own = battle.active_pokemon

        # --- Status move check (before damage calc) ---
        if battle.available_moves:
            for move in battle.available_moves:
                if _is_status_move(move) and _should_use_status(move, battle):
                    return self.create_order(move)

        # MaxDamage-style move selection — find best damage move
        best_move = None
        best_dmg = 0.0
        if battle.available_moves:
            best_move = max(battle.available_moves, key=lambda m: _expected_damage(m, opp))
            best_dmg = _expected_damage(best_move, opp)

        # --- Switching decision (competitive rules) ---
        should_switch = False

        if battle.available_switches:
            switch_rate = self._switch_count / max(self._turn_count, 1)
            if switch_rate < 0.20:
                # Switch when walled — best move does negligible damage
                if best_dmg < 22.0:
                    should_switch = True
                # Switch at severe type disadvantage
                if not should_switch:
                    own_vs_opp = _type_advantage(own, opp)
                    if own_vs_opp > 2.0:
                        should_switch = True

        if should_switch and battle.available_switches:
            viable = [p for p in battle.available_switches if p.current_hp_fraction > 0.3]
            if not viable:
                viable = battle.available_switches
            best_switch = min(viable, key=lambda p: _type_advantage(p, opp))
            if _type_advantage(best_switch, opp) < _type_advantage(own, opp) - 0.5:
                self._switch_count += 1
                return self.create_order(best_switch)

        # Default: highest damage move
        if best_move and best_dmg > 0:
            return self.create_order(best_move)

        # No good attack — try any move
        if battle.available_moves:
            return self.create_order(best_move)

        # No moves at all — switch to healthiest
        if battle.available_switches:
            return self.create_order(max(battle.available_switches, key=lambda p: p.current_hp_fraction))

        return self.choose_random_move(battle)


__all__ = [
    "MaxDamagePlayer",
    "TypeMatchupPlayer",
    "StallPlayer",
    "AggressiveSwitcher",
    "SmartHeuristicPlayer",
    "EpsilonMaxDamagePlayer",
    "EpsilonTypeMatchupPlayer",
    "EpsilonStallPlayer",
    "EpsilonAggressiveSwitcher",
    "RandomPlayer",
]
