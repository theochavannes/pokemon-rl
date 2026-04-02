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


_STAT_SCALE = 255.0  # Gen 1 max base stat (Mewtwo Special 154, HP 255 for Chansey — safe ceiling)


def _types(pokemon) -> list:
    """Two type floats: PokemonType.value / 18.0, or 0.0 if monotype / unknown."""
    t1 = pokemon.type_1.value / 18.0 if pokemon.type_1 else 0.0
    t2 = pokemon.type_2.value / 18.0 if pokemon.type_2 else 0.0
    return [t1, t2]


def _base_stats(pokemon) -> list:
    """Normalized base Attack, Defense, Special, Speed — deterministic from species at L100."""
    bs = pokemon.base_stats
    return [
        bs.get("atk", 0) / _STAT_SCALE,
        bs.get("def", 0) / _STAT_SCALE,
        bs.get("spa", 0) / _STAT_SCALE,   # Gen 1 Special (same for SpAtk and SpDef)
        bs.get("spe", 0) / _STAT_SCALE,
    ]


def embed_battle(battle) -> np.ndarray:
    """
    Produces a 127-dim float32 observation from a poke-env Battle object.

    In gen1randombattle all Pokemon are level 100 with maxed DVs and stat experience,
    so base stats are fully deterministic from species — including them lets the agent
    reason about damage ranges, speed tiers, and switch-in matchups.

    Observation breakdown (127 dims total):
      Own active        16  HP, 6 boosts (incl acc/eva), status, active, fainted,
                            type_1, type_2, base_atk, base_def, base_spa, base_spe
      Own moves         16  4 moves × (base_power, type, PP, effectiveness vs opp)
      Own bench         36  6 slots × (HP, fainted, type_1, type_2, status, base_spe)
      Opp active        15  HP, 6 boosts (incl acc/eva), status, fainted,
                            type_1, type_2, base_atk, base_def, base_spa, base_spe
      Opp bench         30  6 slots × (HP, fainted, revealed, type_1, type_2)
                            unrevealed = [-1, 0, 0, 0, 0]
      Opp revealed mvs  14  up to 2 seen opp moves × 4 features + 6 reserved zeros

    Stat boosts (own + opp) both include accuracy and evasion — Double Team /
    Sand-Attack are real strategies in Gen 1 and the agent needs to track them
    for both sides.

    Status on own bench is encoded so the agent knows if its switch-in is asleep
    or paralyzed before committing to the switch.
    """
    obs = []
    own = battle.active_pokemon
    opp = battle.opponent_active_pokemon

    # --- Own active Pokemon (16 dims) ---
    obs.append(_hp(own))
    obs.append(_boost(own.boosts, "atk"))
    obs.append(_boost(own.boosts, "def"))
    obs.append(_boost(own.boosts, "spe"))
    obs.append(_boost(own.boosts, "spa"))
    obs.append(_boost(own.boosts, "accuracy"))
    obs.append(_boost(own.boosts, "evasion"))
    obs.append(_status(own))
    obs.append(1.0)                            # is_active placeholder
    obs.append(1.0 if own.fainted else 0.0)
    obs.extend(_types(own))
    obs.extend(_base_stats(own))

    # --- Own moves (4 × 4 = 16 dims) ---
    moves = list(own.moves.values())[:4]
    for move in moves:
        obs.extend(_move_features(move, opp))
    for _ in range(4 - len(moves)):
        obs.extend([0.0, 0.0, 0.0, 0.0])

    # --- Own bench (6 × 6 = 36 dims) ---
    # Types + status so agent can pick a switch-in that resists the current threat
    # and isn't already crippled. base_spe for speed tier awareness.
    own_team = list(battle.team.values())
    for i in range(6):
        if i < len(own_team):
            mon = own_team[i]
            obs.append(_hp(mon))
            obs.append(1.0 if mon.fainted else 0.0)
            obs.extend(_types(mon))
            obs.append(_status(mon))
            obs.append(mon.base_stats.get("spe", 0) / _STAT_SCALE)
        else:
            obs.extend([0.0] * 6)

    # --- Opponent active Pokemon (15 dims) ---
    # Full 6 boosts (acc+eva included) — Double Team abuse is a real Gen 1 threat.
    # Base stats always available once the Pokemon is on the field.
    obs.append(_hp(opp))
    obs.append(_boost(opp.boosts, "atk"))
    obs.append(_boost(opp.boosts, "def"))
    obs.append(_boost(opp.boosts, "spe"))
    obs.append(_boost(opp.boosts, "spa"))
    obs.append(_boost(opp.boosts, "accuracy"))
    obs.append(_boost(opp.boosts, "evasion"))
    obs.append(_status(opp))
    obs.append(1.0 if opp.fainted else 0.0)
    obs.extend(_types(opp))
    obs.extend(_base_stats(opp))

    # --- Opponent bench (6 × 5 = 30 dims) ---
    # unrevealed = [-1, 0, 0, 0, 0] — distinct from fainted (0, 1, 0, 0, 0)
    # types are 0 until the Pokemon is revealed (switched in)
    opp_team = list(battle.opponent_team.values())
    for i in range(6):
        if i < len(opp_team):
            mon = opp_team[i]
            obs.append(_hp(mon))
            obs.append(1.0 if mon.fainted else 0.0)
            obs.append(1.0)                    # revealed = True
            obs.extend(_types(mon))
        else:
            obs.extend([-1.0, 0.0, 0.0, 0.0, 0.0])

    # --- Opponent revealed moves (up to 2 × 4 = 8 dims + 6 reserved = 14 dims) ---
    # effectiveness calculated vs own active so agent knows the threat level
    opp_revealed = list(opp.moves.values())[:2] if opp else []
    for move in opp_revealed:
        obs.extend(_move_features(move, own))
    for _ in range(2 - len(opp_revealed)):
        obs.extend([0.0, 0.0, 0.0, 0.0])
    obs.extend([0.0] * 6)                      # reserved for future expansion

    result = np.array(obs, dtype=np.float32)
    assert result.shape == (127,), f"Obs shape mismatch: expected (127,), got {result.shape}"
    return result


class Gen1Env(SinglesEnv):
    """
    Gen 1 random battle environment for MaskablePPO.

    Observation breakdown (127 dims total):
      Own active        : 16 dims  HP, 6 boosts (incl acc/eva), status, active, fainted,
                                   type_1, type_2, base_atk, base_def, base_spa, base_spe
      Own moves         : 16 dims  4 moves × (base_power, type, PP, effectiveness vs opp)
      Own bench         : 36 dims  6 slots × (HP, fainted, type_1, type_2, status, base_spe)
      Opp active        : 15 dims  HP, 6 boosts (incl acc/eva), status, fainted,
                                   type_1, type_2, base_atk, base_def, base_spa, base_spe
      Opp bench         : 30 dims  6 slots × (HP, fainted, revealed, type_1, type_2)
      Opp revealed mvs  : 14 dims  up to 2 seen opp moves × 4 features + 6 reserved
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
        return embed_battle(battle)

    def describe_embedding(self):
        return Box(low=-1.0, high=1.0, shape=(127,), dtype=np.float32)


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


def make_env(
    env_index: int = 0,
    battle_format: str = "gen1randombattle",
    save_replays: bool | str = False,
    opponent_type: str = "random",
    opponent_model_path: str | None = None,
) -> "Monitor":
    """
    Factory for a single SB3-compatible Gen 1 environment.

    opponent_type:
        "random"    — RandomPlayer (default, Phase A)
        "maxdamage" — MaxDamagePlayer (Phase B)
        "policy"    — FrozenPolicyPlayer loaded from opponent_model_path (Phase C / self-play)

    Use with functools.partial for SubprocVecEnv (lambdas are not picklable):
        envs = SubprocVecEnv([partial(make_env, i, opponent_type="maxdamage") for i in range(N)])
    """
    import random
    import string
    from poke_env.player.baselines import RandomPlayer
    from stable_baselines3.common.monitor import Monitor

    suffix = "".join(random.choices(string.ascii_lowercase, k=6))
    inner = Gen1Env(
        account_configuration1=AccountConfiguration(f"PPOAgent{env_index}{suffix}", None),
        account_configuration2=AccountConfiguration(f"RandOpp{env_index}{suffix}", None),
        battle_format=battle_format,
        log_level=25,
        save_replays=save_replays,
    )

    if opponent_type == "random":
        from poke_env.player.baselines import RandomPlayer as _RP
        opponent = _RP(
            battle_format=battle_format,
            account_configuration=AccountConfiguration(f"Puppet{env_index}{suffix}", None),
            start_listening=False,
            log_level=25,
        )
    elif opponent_type == "maxdamage":
        from src.agents.heuristic_agent import MaxDamagePlayer
        opponent = MaxDamagePlayer(
            battle_format=battle_format,
            account_configuration=AccountConfiguration(f"Puppet{env_index}{suffix}", None),
            start_listening=False,
            log_level=25,
        )
    elif opponent_type == "policy":
        if opponent_model_path is None:
            raise ValueError("opponent_model_path required for opponent_type='policy'")
        from src.agents.policy_player import FrozenPolicyPlayer
        opponent = FrozenPolicyPlayer(
            model_path=opponent_model_path,
            battle_format=battle_format,
            account_configuration=AccountConfiguration(f"Puppet{env_index}{suffix}", None),
            start_listening=False,
            log_level=25,
        )
    else:
        raise ValueError(f"Unknown opponent_type: {opponent_type!r}")

    return Monitor(SB3Wrapper(SingleAgentWrapper(inner, opponent)))
