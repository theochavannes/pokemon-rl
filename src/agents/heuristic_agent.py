"""
Heuristic baseline agents.

MaxDamagePlayer: picks the move with highest expected damage (base_power × type_effectiveness).
Outperforms poke-env's built-in MaxBasePowerPlayer by accounting for type matchups.

Used as a benchmark — the RL agent must beat it >60% before moving to self-play.
"""

from poke_env.battle.status import Status
from poke_env.data import GenData
from poke_env.player.baselines import RandomPlayer  # re-exported for convenience
from poke_env.player.player import Player

_GEN1_TYPE_CHART = GenData.from_gen(1).type_chart


def _expected_damage(move, opponent) -> float:
    """
    Proxy for expected damage: base_power × type_effectiveness.
    Status moves (base_power=0) score 0 — the heuristic never uses them.
    """
    if move.base_power == 0:
        return 0.0
    try:
        effectiveness = move.type.damage_multiplier(
            opponent.type_1, opponent.type_2, type_chart=_GEN1_TYPE_CHART
        )
    except Exception:
        effectiveness = 1.0
    return move.base_power * effectiveness


class MaxDamagePlayer(Player):
    """
    Always uses the move with the highest expected damage against the active opponent.
    On forced switches, sends in the benched Pokemon with the highest remaining HP.
    """

    def choose_move(self, battle):
        if battle.available_moves:
            best_move = max(
                battle.available_moves,
                key=lambda m: _expected_damage(m, battle.opponent_active_pokemon),
            )
            return self.create_order(best_move)

        # No moves available — forced switch
        if battle.available_switches:
            return self.create_order(
                max(battle.available_switches, key=lambda p: p.current_hp_fraction)
            )

        return self.choose_random_move(battle)


__all__ = ["MaxDamagePlayer", "RandomPlayer"]
