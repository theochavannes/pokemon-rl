"""
Gen 1 (RBY) Gymnasium environment.

Wraps poke-env's SinglesEnv for use with MaskablePPO.

Observation space : Box(-1, 1, shape=(421,), float32)
Action space      : Discrete — provided by SinglesEnv, masked via action_masks()
Reward            : Shaped (fainted/HP/status deltas) + ±3.0 terminal victory bonus

Env stack for training:
    Gen1Env (SinglesEnv subclass)
        ↓
    SingleAgentWrapper   obs = {"observation": np.array(421,), "action_mask": np.array(N,)}
        ↓
    SB3Wrapper           obs = np.array(421,)  +  action_masks() -> bool array
        ↓
    DummyVecEnv / SubprocVecEnv
"""

import gymnasium
import numpy as np
from gymnasium.spaces import Box
from poke_env.battle.status import Status
from poke_env.data import GenData
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from poke_env.environment.singles_env import SinglesEnv
from poke_env.ps_client.account_configuration import AccountConfiguration

from src.tier_baseline import matchup_baseline

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
    5 features per move:
      base_power     — normalized by 150
      type_index     — PokemonType enum value / 18
      pp_fraction    — current_pp / max_pp
      effectiveness  — type multiplier vs opponent / 4 (max 4x in Gen 1)
      accuracy       — move accuracy 0.0–1.0 (1.0 = always hits)
                       Distinguishes Thunder (70%) from Thunderbolt (100%)
    """
    base_power = move.base_power / 150.0
    type_index = move.type.value / 18.0

    pp_fraction = move.current_pp / move.max_pp if move.max_pp and move.max_pp > 0 else 0.0

    try:
        effectiveness = move.type.damage_multiplier(opponent.type_1, opponent.type_2, type_chart=_GEN1_TYPE_CHART) / 4.0
    except Exception:
        effectiveness = 0.25

    try:
        accuracy = float(move.accuracy) if move.accuracy is not True else 1.0
    except Exception:
        accuracy = 1.0

    return [base_power, type_index, pp_fraction, effectiveness, accuracy]


_STAT_SCALE = 255.0  # Gen 1 max base stat ceiling (Chansey HP=250, safe upper bound)
_PAR_SPEED_MULT = 0.25  # Paralysis quarters speed in Gen 1 (separate from stage boosts)


def _types(pokemon) -> list:
    """Two type floats: PokemonType.value / 18.0, or 0.0 if monotype / unknown."""
    t1 = pokemon.type_1.value / 18.0 if pokemon.type_1 else 0.0
    t2 = pokemon.type_2.value / 18.0 if pokemon.type_2 else 0.0
    return [t1, t2]


def _base_stats(pokemon) -> list:
    """
    Normalized base Atk, Def, Special, Speed — species constants (not level-adjusted).
    Speed is paralysis-adjusted: PAR quarters speed in Gen 1 independently of stage boosts.
    This is NOT captured in pokemon.boosts["spe"], so must be applied here explicitly.
    """
    bs = pokemon.base_stats
    spe = bs.get("spe", 0) / _STAT_SCALE
    if pokemon.status == Status.PAR:
        spe *= _PAR_SPEED_MULT
    return [
        bs.get("atk", 0) / _STAT_SCALE,
        bs.get("def", 0) / _STAT_SCALE,
        bs.get("spa", 0) / _STAT_SCALE,
        spe,
    ]


def embed_battle(battle) -> np.ndarray:
    """
    Produces a 421-dim float32 observation from a poke-env Battle object.

    All Pokemon are forced to level 100 in our Showdown config, so level is
    omitted (always 1.0). Speed in base_stats is paralysis-adjusted (PAR
    quarters speed independently of stage boosts, a Gen 1 mechanic NOT
    captured in pokemon.boosts["spe"]).

    Observation breakdown (421 dims):
      Own active        16  HP, 6 boosts (incl acc+eva), status, active, fainted,
                            type_1, type_2, base_atk, base_def, base_spa, base_spe*
      Own moves         20  4 moves × 5 features (base_power, type, PP, effectiveness,
                            accuracy) — accuracy distinguishes Thunder(70%) vs Tbolt(100%)
      Own bench        174  6 slots × 29 features:
                            HP, fainted, type_1, type_2, status,
                            base_atk, base_def, base_spa, base_spe*,
                            4 moves × 5 features (power, type, PP, effectiveness vs opp, accuracy)
      Opp active        15  HP, 6 boosts (incl acc+eva), status, fainted,
                            type_1, type_2, base_atk, base_def, base_spa, base_spe*
      Opp bench        174  6 slots × 29 features:
                            HP, fainted, revealed, type_1, type_2,
                            base_atk, base_def, base_spa, base_spe*,
                            up to 4 REVEALED moves × 5 features (effectiveness vs own active)
                            unrevealed = [-1, 0, 0, ..., 0]
      Opp revealed mvs  20  up to 4 seen opp ACTIVE moves × 5 features (padded)
      Trapping           2  own_trapped, opp_maybe_trapped (0/1 flags)

    * base_spe is paralysis-adjusted where applicable.
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
    obs.append(1.0)  # is_active placeholder
    obs.append(1.0 if own.fainted else 0.0)
    obs.extend(_types(own))
    obs.extend(_base_stats(own))

    # --- Own moves (4 × 5 = 20 dims) ---
    moves = list(own.moves.values())[:4]
    for move in moves:
        obs.extend(_move_features(move, opp))
    for _ in range(4 - len(moves)):
        obs.extend([0.0, 0.0, 0.0, 0.0, 0.0])

    # --- Own bench (6 × 29 = 174 dims) ---
    # Full info per bench Pokemon: stats, types, status, and all 4 moves with
    # effectiveness vs current opponent. Agent can make informed switch decisions.
    own_team = list(battle.team.values())
    for i in range(6):
        if i < len(own_team):
            mon = own_team[i]
            obs.append(_hp(mon))
            obs.append(1.0 if mon.fainted else 0.0)
            obs.extend(_types(mon))
            obs.append(_status(mon))
            obs.extend(_base_stats(mon))  # atk, def, spa, spe (4 dims)
            # Bench Pokemon moves (4 × 5 = 20 dims)
            bench_moves = list(mon.moves.values())[:4]
            for move in bench_moves:
                obs.extend(_move_features(move, opp))
            for _ in range(4 - len(bench_moves)):
                obs.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            obs.extend([0.0] * 29)

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

    # --- Opponent bench (6 × 29 = 174 dims) ---
    # Full info for revealed Pokemon: stats, types, and observed moves with
    # effectiveness vs our active. Unrevealed slots are all -1/0 padding.
    opp_team = list(battle.opponent_team.values())
    for i in range(6):
        if i < len(opp_team):
            mon = opp_team[i]
            obs.append(_hp(mon))
            obs.append(1.0 if mon.fainted else 0.0)
            obs.append(1.0)  # revealed = True
            obs.extend(_types(mon))
            obs.extend(_base_stats(mon))  # atk, def, spa, spe (4 dims)
            # Revealed moves (only moves we've seen this opponent use)
            opp_bench_moves = list(mon.moves.values())[:4]
            for move in opp_bench_moves:
                obs.extend(_move_features(move, own))  # effectiveness vs OUR active
            for _ in range(4 - len(opp_bench_moves)):
                obs.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            obs.extend([-1.0] + [0.0] * 28)

    # --- Opponent revealed moves (up to 4 × 5 = 20 dims) ---
    # Up to 4 moves the opponent has used, each with 5 features.
    # Effectiveness is vs own active so agent knows incoming threat level.
    # Padding = zeros for unseen moves.
    opp_revealed = list(opp.moves.values())[:4] if opp else []
    for move in opp_revealed:
        obs.extend(_move_features(move, own))
    for _ in range(4 - len(opp_revealed)):
        obs.extend([0.0, 0.0, 0.0, 0.0, 0.0])

    # --- Trapping (2 dims) ---
    # battle.trapped: OUR active Pokemon cannot switch (e.g. being Wrapped)
    # battle.maybe_trapped: we might be trapped (opponent has trapping move)
    obs.append(1.0 if battle.trapped else 0.0)
    obs.append(1.0 if battle.maybe_trapped else 0.0)

    result = np.array(obs, dtype=np.float32)
    assert result.shape == (421,), f"Obs shape mismatch: expected (421,), got {result.shape}"
    assert np.all(np.isfinite(result)), "Obs contains NaN/Inf"
    return result


class Gen1Env(SinglesEnv):
    """
    Gen 1 random battle environment for MaskablePPO.

    Observation breakdown (421 dims total):
      Own active        :  16 dims  HP, 6 boosts, status, active, fainted, types, 4 stats
      Own moves         :  20 dims  4 moves × 5 features
      Own bench         : 174 dims  6 × 29 (stats, types, status, 4 moves w/ effectiveness)
      Opp active        :  15 dims  HP, 6 boosts, status, fainted, types, 4 stats
      Opp bench         : 174 dims  6 × 29 (stats, types, revealed moves w/ effectiveness vs own)
      Opp revealed mvs  :  20 dims  up to 4 seen opp active moves × 5 features
      Trapping          :   2 dims  own_trapped, own_maybe_trapped
      * base_spe is paralysis-adjusted (PAR quarters speed in Gen 1)
    """

    def __init__(self, shaping_factor: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.shaping_factor = shaping_factor
        # PokeEnv.__init__ does not set observation_spaces — must be set by subclass.
        # __setattr__ intercepts this and wraps each value in Dict(observation, action_mask).
        self.observation_spaces = {agent: self.describe_embedding() for agent in self.possible_agents}

    def calc_reward(self, battle) -> float:
        reward = self.reward_computing_helper(
            battle,
            fainted_value=0.5,
            hp_value=0.0,
            status_value=0.0,
            victory_value=1.0,
        )
        # Matchup baseline: subtract team-quality advantage from terminal reward
        # so the agent gets more credit for winning bad matchups and less for easy ones.
        if battle.won or battle.lost:
            reward -= matchup_baseline(battle)
        return reward

    def embed_battle(self, battle) -> np.ndarray:
        return embed_battle(battle)

    def describe_embedding(self):
        return Box(low=-1.0, high=1.0, shape=(421,), dtype=np.float32)


class SB3Wrapper(gymnasium.Wrapper):
    """
    Adapts SingleAgentWrapper's dict observations to flat numpy arrays for SB3.

    SingleAgentWrapper returns obs = {"observation": array(421,), "action_mask": array(N,)}
    This wrapper unpacks it so SB3 sees a plain Box observation and action_masks() method.
    """

    def __init__(self, env: SingleAgentWrapper):
        super().__init__(env)
        self.observation_space = env.observation_space["observation"]
        self._last_action_mask = np.ones(env.action_space.n, dtype=bool)

    def step(self, action):
        try:
            obs_dict, reward, terminated, truncated, info = self.env.step(action)
        except ValueError:
            # poke-env occasionally desyncs on opponent action. Use neutral reward
            # (0.0) so this doesn't poison training with fake losses.
            obs, info = self.reset()
            info["desync"] = True
            return obs, 0.0, True, False, info
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
    shaping_factor: float = 1.0,
    opponent_difficulty: float = 0.8,
    selfplay_model_path: str | None = None,
):
    """
    Factory for a single SB3-compatible Gen 1 environment.

    Args:
        opponent_difficulty: Controls opponent strength. Meaning depends on opponent_type:
            - softmax_damage: temperature (2.0=soft → 0.1=near-argmax)
            - epsilon_* / mixed: epsilon for policy blend (0.0=pure heuristic → 1.0=pure selfplay)

    opponent_type:
        "random"           — RandomPlayer (Phase A)
        "random_attacker"  — RandomAttackerPlayer (Phase B)
        "softmax_damage"   — SoftmaxDamagePlayer (Phase C, temp anneals)
        "epsilon_maxdamage" — EpsilonMaxDamagePlayer
        "maxdamage"        — MaxDamagePlayer (pure)
        "mixed"            — Mixed heuristic pool + optional self-play (Phase D)
        "policy"           — FrozenPolicyPlayer from opponent_model_path

    Use with functools.partial for SubprocVecEnv (lambdas are not picklable):
        envs = SubprocVecEnv([partial(make_env, i, opponent_type="maxdamage") for i in range(N)])
    """
    import random
    import string

    from stable_baselines3.common.monitor import Monitor

    suffix = "".join(random.choices(string.ascii_lowercase, k=6))
    inner = Gen1Env(
        shaping_factor=shaping_factor,
        account_configuration1=AccountConfiguration(f"PPOAgent{env_index}{suffix}", None),
        account_configuration2=AccountConfiguration(f"RandOpp{env_index}{suffix}", None),
        battle_format=battle_format,
        log_level=40,
        save_replays=save_replays,
    )

    if opponent_type == "random_attacker":
        from src.agents.heuristic_agent import RandomAttackerPlayer

        opponent = RandomAttackerPlayer(
            battle_format=battle_format,
            account_configuration=AccountConfiguration(f"Puppet{env_index}{suffix}", None),
            start_listening=False,
            log_level=40,
        )
    elif opponent_type == "random":
        from poke_env.player.baselines import RandomPlayer as _RP

        opponent = _RP(
            battle_format=battle_format,
            account_configuration=AccountConfiguration(f"Puppet{env_index}{suffix}", None),
            start_listening=False,
            log_level=40,
        )
    elif opponent_type == "softmax_damage":
        from src.agents.heuristic_agent import SoftmaxDamagePlayer

        opponent = SoftmaxDamagePlayer(
            temperature=opponent_difficulty,
            battle_format=battle_format,
            account_configuration=AccountConfiguration(f"Puppet{env_index}{suffix}", None),
            start_listening=False,
            log_level=40,
        )
    elif opponent_type == "epsilon_maxdamage":
        from src.agents.heuristic_agent import EpsilonMaxDamagePlayer

        opponent = EpsilonMaxDamagePlayer(
            epsilon=opponent_difficulty,
            battle_format=battle_format,
            account_configuration=AccountConfiguration(f"Puppet{env_index}{suffix}", None),
            start_listening=False,
            log_level=40,
        )
    elif opponent_type == "maxdamage":
        from src.agents.heuristic_agent import MaxDamagePlayer

        opponent = MaxDamagePlayer(
            battle_format=battle_format,
            account_configuration=AccountConfiguration(f"Puppet{env_index}{suffix}", None),
            start_listening=False,
            log_level=40,
        )
    elif opponent_type == "typematchup":
        from src.agents.heuristic_agent import TypeMatchupPlayer

        opponent = TypeMatchupPlayer(
            battle_format=battle_format,
            account_configuration=AccountConfiguration(f"Puppet{env_index}{suffix}", None),
            start_listening=False,
            log_level=40,
        )
    elif opponent_type == "stall":
        from src.agents.heuristic_agent import StallPlayer

        opponent = StallPlayer(
            battle_format=battle_format,
            account_configuration=AccountConfiguration(f"Puppet{env_index}{suffix}", None),
            start_listening=False,
            log_level=40,
        )
    elif opponent_type == "aggressive_switcher":
        from src.agents.heuristic_agent import AggressiveSwitcher

        opponent = AggressiveSwitcher(
            battle_format=battle_format,
            account_configuration=AccountConfiguration(f"Puppet{env_index}{suffix}", None),
            start_listening=False,
            log_level=40,
        )
    elif opponent_type == "mixed":
        from src.agents.heuristic_agent import (
            EpsilonAggressiveSwitcher,
            EpsilonMaxDamagePlayer,
            EpsilonStallPlayer,
            EpsilonTypeMatchupPlayer,
        )

        _HEURISTIC_POOL = [
            EpsilonMaxDamagePlayer,
            EpsilonTypeMatchupPlayer,
            EpsilonStallPlayer,
            EpsilonAggressiveSwitcher,
        ]
        # If selfplay model provided, last env slot uses frozen self-play
        if selfplay_model_path and env_index == 3:
            from src.agents.policy_player import FrozenPolicyPlayer

            opponent = FrozenPolicyPlayer(
                model_path=selfplay_model_path,
                battle_format=battle_format,
                account_configuration=AccountConfiguration(f"Puppet{env_index}{suffix}", None),
                start_listening=False,
                log_level=40,
            )
        else:
            cls = _HEURISTIC_POOL[env_index % len(_HEURISTIC_POOL)]
            opponent = cls(
                epsilon=opponent_difficulty,
                battle_format=battle_format,
                account_configuration=AccountConfiguration(f"Puppet{env_index}{suffix}", None),
                start_listening=False,
                log_level=40,
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
            log_level=40,
        )
    else:
        raise ValueError(f"Unknown opponent_type: {opponent_type!r}")

    wrapped = Monitor(SB3Wrapper(SingleAgentWrapper(inner, opponent)))
    wrapped._opponent = opponent  # expose for epsilon annealing
    return wrapped
