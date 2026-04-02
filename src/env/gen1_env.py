"""
Gen 1 (RBY) Gymnasium environment.

Wraps poke-env's SinglesEnv for use with MaskablePPO.

Observation space : Box(-1, 1, shape=(64,), float32)
Action space      : Discrete — provided by SinglesEnv, masked via action_masks()
Reward            : +1.0 win | -1.0 loss | 0.0 ongoing

Env stack for training:
    Gen1Env (SinglesEnv subclass)
        ↓
    SingleAgentWrapper   obs = {"observation": np.array(64,), "action_mask": np.array(N,)}
        ↓
    SB3Wrapper           obs = np.array(64,)  +  action_masks() -> bool array
        ↓
    DummyVecEnv / SubprocVecEnv
"""

import numpy as np
import gymnasium
from gymnasium.spaces import Box
from poke_env.battle.status import Status
from poke_env.data import GenData
from poke_env.environment.singles_env import SinglesEnv
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from poke_env.ps_client.account_configuration import AccountConfiguration

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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # PokeEnv.__init__ does not set observation_spaces — must be set by subclass.
        # __setattr__ intercepts this and wraps each value in Dict(observation, action_mask).
        self.observation_spaces = {
            agent: self.describe_embedding() for agent in self.possible_agents
        }

    def calc_reward(self, battle) -> float:
        return self.reward_computing_helper(
            battle,
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


class SB3Wrapper(gymnasium.Wrapper):
    """
    Adapts SingleAgentWrapper's dict observations to flat numpy arrays for SB3.

    SingleAgentWrapper returns obs = {"observation": array(64,), "action_mask": array(N,)}
    This wrapper unpacks it so SB3 sees a plain Box observation and action_masks() method.
    """

    def __init__(self, env: SingleAgentWrapper):
        super().__init__(env)
        self.observation_space = env.observation_space["observation"]
        self._last_action_mask = np.ones(env.action_space.n, dtype=bool)

    def step(self, action):
        obs_dict, reward, terminated, truncated, info = self.env.step(action)
        self._last_action_mask = obs_dict["action_mask"].astype(bool)
        return obs_dict["observation"], reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs_dict, info = self.env.reset(**kwargs)
        self._last_action_mask = obs_dict["action_mask"].astype(bool)
        return obs_dict["observation"], info

    def action_masks(self) -> np.ndarray:
        return self._last_action_mask


def make_env(env_index: int = 0, battle_format: str = "gen1randombattle") -> SB3Wrapper:
    """
    Factory function for creating a single SB3-compatible Gen 1 environment.

    Each env gets unique usernames to allow parallel instances without conflicts.
    A random 6-char suffix is appended so reconnecting after a crash never
    collides with a leftover battle on the Showdown server.

    The opponent player is a puppet (start_listening=False) — it provides move
    choices but has no WebSocket connection; the battle is injected directly.

    Use with functools.partial for SubprocVecEnv (lambdas are not picklable):
        envs = SubprocVecEnv([partial(make_env, i) for i in range(N_ENVS)])
    """
    import random
    import string
    from poke_env.player.baselines import RandomPlayer

    suffix = "".join(random.choices(string.ascii_lowercase, k=6))
    inner = Gen1Env(
        account_configuration1=AccountConfiguration(f"PPOAgent{env_index}{suffix}", None),
        account_configuration2=AccountConfiguration(f"RandOpp{env_index}{suffix}", None),
        battle_format=battle_format,
        log_level=25,
    )
    opponent = RandomPlayer(
        battle_format=battle_format,
        account_configuration=AccountConfiguration(f"Puppet{env_index}{suffix}", None),
        start_listening=False,
        log_level=25,
    )
    return SB3Wrapper(SingleAgentWrapper(inner, opponent))
