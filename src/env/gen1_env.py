"""
Gen 1 (RBY) Gymnasium environment.

Wraps poke-env's SinglesEnv for use with MaskablePPO.

Observation space : Box(-1, 1, shape=(64,), float32)
Action space      : Discrete — provided by SinglesEnv, masked via action_masks()
Reward            : +1.0 win | -1.0 loss | 0.0 ongoing
"""

import numpy as np
from gymnasium.spaces import Box
from poke_env.battle.status import Status
from poke_env.data import GenData
from poke_env.environment.singles_env import SinglesEnv

_GEN1_TYPE_CHART = GenData.from_gen(1).type_chart

_STATUS_TO_FLOAT = {
    None: 0.0,
    Status.PSN: 0.2,
    Status.TOX: 0.2,  # Toxic → treated as Poison in Gen 1
    Status.BRN: 0.4,
    Status.SLP: 0.6,
    Status.PAR: 0.8,
    Status.FRZ: 1.0,
}


def _boost(boosts: dict, key: str) -> float:
    """Normalize a stat boost from [-6, 6] → [-1, 1]."""
    return boosts.get(key, 0) / 6.0


def _status(pokemon) -> float:
    return _STATUS_TO_FLOAT.get(pokemon.status, 0.0)


def _hp(pokemon) -> float:
    return 0.0 if pokemon.fainted else float(pokemon.current_hp_fraction)


def _move_features(move, opponent) -> list:
    """
    4 features per move:
      base_power     — normalized by 150 (Selfdestruct is the max at 130 in Gen 1 adjusted)
      type_index     — PokemonType enum value / 18
      pp_fraction    — current_pp / max_pp
      effectiveness  — type multiplier vs opponent active / 4 (max 4x in Gen 1)
    """
    base_power = move.base_power / 150.0

    type_index = move.type.value / 18.0

    if move.max_pp and move.max_pp > 0:
        pp_fraction = move.current_pp / move.max_pp
    else:
        pp_fraction = 0.0

    try:
        effectiveness = move.type.damage_multiplier(
            opponent.type_1, opponent.type_2, type_chart=_GEN1_TYPE_CHART
        ) / 4.0
    except Exception:
        effectiveness = 0.25  # neutral fallback (1x / 4)

    return [base_power, type_index, pp_fraction, effectiveness]


class Gen1Env(SinglesEnv):
    """
    Gen 1 random battle environment for MaskablePPO.

    Observation breakdown (64 dims total):
      Own active Pokemon    : 10 dims  (HP, 6 boosts, status, is_active placeholder, fainted)
      Own moves             : 16 dims  (4 moves × 4 features)
      Opponent active       :  7 dims  (HP, 4 boosts, status, fainted)
      Own bench             : 12 dims  (6 slots × HP + fainted)
      Opponent bench        : 18 dims  (6 slots × HP + fainted + revealed)
      Speed context         :  1 dim   (own_speed > opp_speed)

    Phase 6 expansions: add bench Pokemon types, explicit type-chart encoding.
    """

    def calc_reward(self, last_battle, current_battle) -> float:
        return self.reward_computing_helper(
            current_battle,
            fainted_value=0.0,
            hp_value=0.0,
            victory_value=1.0,
        )

    def embed_battle(self, battle) -> np.ndarray:
        obs = []
        own = battle.active_pokemon
        opp = battle.opponent_active_pokemon

        # --- Own active Pokemon (10 dims) ---
        obs.append(_hp(own))
        obs.append(_boost(own.boosts, "atk"))
        obs.append(_boost(own.boosts, "def"))
        obs.append(_boost(own.boosts, "spe"))
        obs.append(_boost(own.boosts, "spa"))   # Gen 1 Special stat
        obs.append(_boost(own.boosts, "accuracy"))
        obs.append(_boost(own.boosts, "evasion"))
        obs.append(_status(own))
        obs.append(1.0)                          # is_active placeholder (always 1)
        obs.append(1.0 if own.fainted else 0.0)

        # --- Own moves (4 × 4 = 16 dims) ---
        # Use active_pokemon.moves (all known moves) not available_moves
        # (available_moves changes per step and excludes PP=0 moves)
        moves = list(own.moves.values())[:4]
        for move in moves:
            obs.extend(_move_features(move, opp))
        for _ in range(4 - len(moves)):          # pad if fewer than 4 moves known
            obs.extend([0.0, 0.0, 0.0, 0.0])

        # --- Opponent active Pokemon (7 dims) ---
        obs.append(_hp(opp))
        obs.append(_boost(opp.boosts, "atk"))
        obs.append(_boost(opp.boosts, "def"))
        obs.append(_boost(opp.boosts, "spe"))
        obs.append(_boost(opp.boosts, "spa"))
        obs.append(_status(opp))
        obs.append(1.0 if opp.fainted else 0.0)

        # --- Own bench (6 × 2 = 12 dims) ---
        own_team = list(battle.team.values())
        for i in range(6):
            if i < len(own_team):
                mon = own_team[i]
                obs.append(_hp(mon))
                obs.append(1.0 if mon.fainted else 0.0)
            else:
                obs.extend([0.0, 0.0])

        # --- Opponent bench (6 × 3 = 18 dims) ---
        # Slots beyond what's been revealed stay at [-1, 0, 0]
        # This makes "not revealed" a learnable signal distinct from "fainted"
        opp_team = list(battle.opponent_team.values())
        for i in range(6):
            if i < len(opp_team):
                mon = opp_team[i]
                obs.append(_hp(mon))
                obs.append(1.0 if mon.fainted else 0.0)
                obs.append(1.0)                  # revealed = True
            else:
                obs.extend([-1.0, 0.0, 0.0])    # unrevealed slot

        # --- Speed context (1 dim) ---
        own_speed = own.base_stats.get("spe", 0)
        opp_speed = opp.base_stats.get("spe", 0) if opp else 0
        obs.append(1.0 if own_speed > opp_speed else 0.0)

        result = np.array(obs, dtype=np.float32)
        assert result.shape == (64,), f"Obs shape mismatch: expected (64,), got {result.shape}"
        return result

    def describe_embedding(self):
        return Box(low=-1.0, high=1.0, shape=(64,), dtype=np.float32)
