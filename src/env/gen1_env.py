"""
Gen 1 (RBY) Gymnasium environment.

Wraps poke-env's SinglesEnv for use with MaskablePPO.

Observation space : Box(-1, 1, shape=(1727,), float32)
Action space      : Discrete — provided by SinglesEnv, masked via action_masks()
Reward            : Shaped (fainted/HP/status deltas) + terminal victory bonus

Env stack for training:
    Gen1Env (SinglesEnv subclass)
        ↓
    SingleAgentWrapper   obs = {"observation": np.array(1727,), "action_mask": np.array(N,)}
        ↓
    SB3Wrapper           obs = np.array(1727,)  +  action_masks() -> bool array
        ↓
    DummyVecEnv / SubprocVecEnv
"""

import contextlib

import gymnasium
import numpy as np
from gymnasium.spaces import Box
from poke_env.battle.effect import Effect
from poke_env.battle.move_category import MoveCategory
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.status import Status
from poke_env.data import GenData
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from poke_env.environment.singles_env import SinglesEnv
from poke_env.ps_client.account_configuration import AccountConfiguration

from src.tier_baseline import NUM_ROLES, matchup_baseline, roles_for

_GEN1_TYPE_CHART = GenData.from_gen(1).type_chart

_STATUS_TO_FLOAT = {
    None: 0.0,
    Status.PSN: 0.15,
    Status.TOX: 0.3,  # Toxic is much stronger (escalating damage each turn)
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


_MOVE_STATUS_MAP = {
    Status.PSN: 0.15,
    Status.TOX: 0.3,  # Toxic is much stronger than regular poison (escalating damage)
    Status.BRN: 0.4,
    Status.SLP: 0.6,
    Status.PAR: 0.8,
    Status.FRZ: 1.0,
}

_MOVE_FEATURES_PER_MOVE = 25
_MOVE_PADDING = [0.0] * _MOVE_FEATURES_PER_MOVE

# Gen 1 trapping moves — prevent opponent from acting for 2-5 turns
_TRAPPING_MOVES = {"wrap", "bind", "clamp", "firespin"}

# Moves that require sleeping target (Dream Eater)
_REQUIRES_SLEEP = {"dreameater"}

# Secondary effect type encoding: what the secondary effect DOES
# Maps secondary status/volatile to a float. Positive = status, negative = stat drops.
_SECONDARY_STATUS_TYPE = {
    "par": 0.8,
    "brn": 0.6,
    "frz": 1.0,
    "psn": 0.3,
    "tox": 0.3,
    "slp": 0.9,
}


def _secondary_effect_type(move) -> float:
    """Encode what a move's secondary effect does (not just the chance)."""
    try:
        entry_sec = move.entry.get("secondary")
        if not entry_sec:
            return 0.0
        # Status effect (Body Slam → par, Blizzard → frz)
        if "status" in entry_sec:
            return _SECONDARY_STATUS_TYPE.get(entry_sec["status"], 0.0)
        # Volatile status (flinch, confusion)
        if "volatileStatus" in entry_sec:
            vs = entry_sec["volatileStatus"]
            if vs == "flinch":
                return 0.4
            if vs == "confusion":
                return 0.5
            return 0.2
        # Stat drops on opponent (Psychic → -spa, Acid → -def)
        if "boosts" in entry_sec:
            return sum(entry_sec["boosts"].values()) / 6.0  # negative for drops
        return 0.0
    except Exception:
        return 0.0


def _status_immune(move, target) -> float:
    """Check if target's type makes it immune to this move's status effect.

    Gen 1 status immunities:
      - Poison/Toxic: Poison types immune
      - Burn: Fire types immune
      - Freeze: Ice types immune
      - Thunder Wave: Ground types immune (Electric-type move, handled by effectiveness=0)
      - Paralysis via Body Slam etc.: No immunity in Gen 1
      - Sleep: No type immunity
    """
    if not move.status:
        return 0.0
    status = str(move.status).upper()
    type_names = set()
    for t in [target.type_1, target.type_2]:
        if t is not None:
            type_names.add(t.name.upper() if hasattr(t, "name") else str(t).upper())
    if ("PSN" in status or "TOX" in status) and "POISON" in type_names:
        return 1.0
    if "BRN" in status and "FIRE" in type_names:
        return 1.0
    if "FRZ" in status and "ICE" in type_names:
        return 1.0
    return 0.0


def _move_features(move, opponent) -> list:
    """
    25 features per move (19 from Sprint 5 + 6 from Sprint 7):
      [Sprint 5] base_power, type_index, pp_fraction, effectiveness, accuracy,
      is_physical, is_special, status_effect, priority, self_boost,
      heal, drain, secondary_chance, secondary_effect_type,
      recoil, self_destruct, fixed_damage, trapping, status_immune
      [Sprint 7] must_recharge, requires_sleep, pp_max_norm,
      is_contact, is_sound, ignore_accuracy
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

    # Step 6: One-hot category (replaces single float)
    is_physical = 1.0 if move.category == MoveCategory.PHYSICAL else 0.0
    is_special = 1.0 if move.category == MoveCategory.SPECIAL else 0.0

    # Status effect the move inflicts
    try:
        status_effect = _MOVE_STATUS_MAP.get(move.status, 0.0) if move.status else 0.0
    except Exception:
        status_effect = 0.0

    # Priority (Gen 1: typically -1 to +1, normalize by 5)
    try:
        priority = max(min(move.priority / 5.0, 1.0), -1.0)
    except Exception:
        priority = 0.0

    # Self boost (sum of boost stage changes / 6)
    self_boost_val = 0.0
    with contextlib.suppress(Exception):
        boosts = move.self_boost
        if not boosts and move.category == MoveCategory.STATUS:
            boosts = move.boosts
        if boosts:
            self_boost_val = sum(boosts.values()) / 6.0

    # Step 7: Separate heal and drain (they work differently)
    heal = 0.0
    with contextlib.suppress(Exception):
        if move.heal and move.heal > 0:
            heal = move.heal
    drain = 0.0
    with contextlib.suppress(Exception):
        if move.drain and move.drain > 0:
            drain = move.drain

    # Secondary status chance (Body Slam 30% para, Blizzard 10% freeze, etc.)
    secondary_chance = 0.0
    with contextlib.suppress(Exception):
        entry_sec = move.entry.get("secondary")
        if entry_sec and "chance" in entry_sec:
            secondary_chance = entry_sec["chance"] / 100.0

    # Step 1: Secondary effect TYPE (what the secondary does, not just the chance)
    sec_effect_type = _secondary_effect_type(move)

    # Recoil fraction (Submission 0.25, Double-Edge 0.25)
    recoil_val = 0.0
    with contextlib.suppress(Exception):
        recoil_val = move.recoil if move.recoil else 0.0

    # Self-destruct flag (Explosion, Self-Destruct)
    self_destruct = 1.0 if move.self_destruct else 0.0

    # Fixed damage (Seismic Toss = "level" = 100 in our config, Dragon Rage = 40)
    fixed_damage = 0.0
    with contextlib.suppress(Exception):
        if move.damage == "level":
            fixed_damage = 1.0  # always 100 damage at level 100
        elif isinstance(move.damage, int | float) and move.damage > 0:
            fixed_damage = move.damage / 100.0

    # Step 8: Trapping move flag (Wrap/Bind/Clamp/Fire Spin — broken in Gen 1)
    trapping = 1.0 if getattr(move, "id", "") in _TRAPPING_MOVES else 0.0

    # Step 9: Status immunity flag
    immune = _status_immune(move, opponent)

    # Sprint 7 Step 3: Hyper Beam recharge flag
    must_recharge = 0.0
    with contextlib.suppress(Exception):
        self_entry = move.entry.get("self") or {}
        if self_entry.get("volatileStatus") == "mustrecharge" or "recharge" in (getattr(move, "flags", None) or {}):
            must_recharge = 1.0

    # Sprint 7 Step 4: Dream Eater conditional flag
    requires_sleep = 1.0 if getattr(move, "id", "") in _REQUIRES_SLEEP else 0.0

    # Sprint 7 Step 5: PP max normalized (move scarcity indicator)
    pp_max_norm = (move.max_pp / 35.0) if move.max_pp and move.max_pp > 0 else 0.0

    # Sprint 7 Step 6: Move-specific flags
    flags = getattr(move, "flags", None) or {}
    is_contact = 1.0 if "contact" in flags else 0.0
    is_sound = 1.0 if "sound" in flags else 0.0
    ignore_accuracy = 1.0 if accuracy >= 1.0 and getattr(move, "accuracy", True) is True else 0.0

    return [
        base_power,
        type_index,
        pp_fraction,
        effectiveness,
        accuracy,
        is_physical,
        is_special,
        status_effect,
        priority,
        self_boost_val,
        heal,
        drain,
        secondary_chance,
        sec_effect_type,
        recoil_val,
        self_destruct,
        fixed_damage,
        trapping,
        immune,
        must_recharge,
        requires_sleep,
        pp_max_norm,
        is_contact,
        is_sound,
        ignore_accuracy,
    ]


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
    Gen 1 status adjustments applied explicitly (not captured in pokemon.boosts):
      - PAR quarters Speed (independently of stage boosts)
      - BRN halves Attack (independently of stage boosts)
    """
    bs = pokemon.base_stats
    atk = bs.get("atk", 0) / _STAT_SCALE
    spe = bs.get("spe", 0) / _STAT_SCALE
    if pokemon.status == Status.PAR:
        spe *= _PAR_SPEED_MULT
    if pokemon.status == Status.BRN:
        atk *= 0.5  # Burn halves physical Attack in Gen 1
    return [
        atk,
        bs.get("def", 0) / _STAT_SCALE,
        bs.get("spa", 0) / _STAT_SCALE,
        spe,
    ]


_ROLE_DIMS = NUM_ROLES * 14  # 14 slots × 12 roles = 168
_OBS_DIM = 1559 + _ROLE_DIMS  # 1727


def _role_features(pokemon) -> list:
    """Return the 12-dim competitive role vector for a Pokemon (zeros if unknown)."""
    if pokemon is None:
        return [0.0] * NUM_ROLES
    try:
        species = pokemon.species.lower().replace("-", "").replace(" ", "")
    except Exception:
        return [0.0] * NUM_ROLES
    return [float(x) for x in roles_for(species)]


def embed_battle(battle) -> np.ndarray:
    """
    Produces a 1727-dim float32 observation from a poke-env Battle object.

    Observation layout:
      Own active        16    HP, 6 boosts, status, active, fainted, types, base stats
      Own moves        104    4 × 25 move features + 4 target_statused
      Own bench        654    6 × 109 (9 mon features + 4 × 25 move features)
      Opp active        15    HP, 6 boosts, status, fainted, types, base stats
      Opp bench        654    6 × 109 (9 mon features + 4 × 25 move features)
      Opp revealed     100    4 × 25 move features
      Trapping           2    own_trapped, opp_maybe_trapped
      Speed advantage    1
      Alive counts       2
      Volatile status    8    sub, reflect, light_screen, confused, leech_seed
      Opp status threat  1
      Toxic counter      1
      Turn phase         1    normalized turn number (S7.2)
      Role features    168    14 slots × 12 roles (own active+6 bench, opp active+6 bench)
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

    # --- Own moves (4 × 25 + 4 = 104 dims) ---
    moves = list(own.moves.values())[:4]
    for move in moves:
        obs.extend(_move_features(move, opp))
        # Step 2: Target already has status flag
        obs.append(1.0 if (move.status and opp.status) else 0.0)
    for _ in range(4 - len(moves)):
        obs.extend(_MOVE_PADDING)
        obs.append(0.0)  # target_statused padding

    _OWN_BENCH_SLOT = 9 + 4 * _MOVE_FEATURES_PER_MOVE  # 109 dims per slot

    # --- Own bench (6 × 85 = 510 dims) ---
    own_team = list(battle.team.values())
    for i in range(6):
        if i < len(own_team):
            mon = own_team[i]
            obs.append(_hp(mon))
            obs.append(1.0 if mon.fainted else 0.0)
            obs.extend(_types(mon))
            obs.append(_status(mon))
            obs.extend(_base_stats(mon))
            bench_moves = list(mon.moves.values())[:4]
            for move in bench_moves:
                obs.extend(_move_features(move, opp))
            for _ in range(4 - len(bench_moves)):
                obs.extend(_MOVE_PADDING)
        else:
            obs.extend([0.0] * _OWN_BENCH_SLOT)

    # --- Opponent active Pokemon (15 dims) ---
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

    _OPP_BENCH_SLOT = 9 + 4 * _MOVE_FEATURES_PER_MOVE  # 109 dims per slot

    # --- Opponent bench (6 × 85 = 510 dims) ---
    opp_team = list(battle.opponent_team.values())
    for i in range(6):
        if i < len(opp_team):
            mon = opp_team[i]
            obs.append(_hp(mon))
            obs.append(1.0 if mon.fainted else 0.0)
            obs.append(1.0)  # revealed = True
            obs.extend(_types(mon))
            obs.extend(_base_stats(mon))
            opp_bench_moves = list(mon.moves.values())[:4]
            for move in opp_bench_moves:
                obs.extend(_move_features(move, own))
            for _ in range(4 - len(opp_bench_moves)):
                obs.extend(_MOVE_PADDING)
        else:
            obs.extend([-1.0] + [0.0] * (_OPP_BENCH_SLOT - 1))

    # --- Opponent revealed moves (4 × 19 = 76 dims) ---
    opp_revealed = list(opp.moves.values())[:4] if opp else []
    for move in opp_revealed:
        obs.extend(_move_features(move, own))
    for _ in range(4 - len(opp_revealed)):
        obs.extend(_MOVE_PADDING)

    # --- Trapping (2 dims) ---
    obs.append(1.0 if battle.trapped else 0.0)
    obs.append(1.0 if battle.maybe_trapped else 0.0)

    # --- Speed advantage (1 dim) ---
    own_spe = own.base_stats.get("spe", 0) / _STAT_SCALE
    opp_spe = opp.base_stats.get("spe", 0) / _STAT_SCALE
    if own.status == Status.PAR:
        own_spe *= _PAR_SPEED_MULT
    if opp.status == Status.PAR:
        opp_spe *= _PAR_SPEED_MULT
    max_spe = max(own_spe, opp_spe, 0.01)
    speed_adv = max(min((own_spe - opp_spe) / max_spe, 1.0), -1.0)
    obs.append(speed_adv)

    # --- Alive counts (2 dims) ---
    own_alive = sum(1 for p in battle.team.values() if not p.fainted) / 6.0
    opp_alive = (6 - sum(1 for p in battle.opponent_team.values() if p.fainted)) / 6.0
    obs.append(own_alive)
    obs.append(opp_alive)

    # --- Step 3: Volatile status features (8 dims) ---
    own_effects = own.effects if hasattr(own, "effects") and own.effects else {}
    opp_effects = opp.effects if hasattr(opp, "effects") and opp.effects else {}
    own_side = battle.side_conditions if hasattr(battle, "side_conditions") else {}

    obs.append(1.0 if Effect.SUBSTITUTE in own_effects else 0.0)
    obs.append(1.0 if Effect.SUBSTITUTE in opp_effects else 0.0)
    obs.append(1.0 if SideCondition.REFLECT in own_side else 0.0)
    obs.append(1.0 if SideCondition.LIGHT_SCREEN in own_side else 0.0)
    obs.append(1.0 if Effect.CONFUSION in own_effects else 0.0)
    obs.append(1.0 if Effect.CONFUSION in opp_effects else 0.0)
    obs.append(1.0 if Effect.LEECH_SEED in own_effects else 0.0)
    obs.append(1.0 if Effect.LEECH_SEED in opp_effects else 0.0)

    # --- Step 4: Opponent status move threat (1 dim) ---
    opp_has_status = 0.0
    with contextlib.suppress(Exception):
        for mon in battle.opponent_team.values():
            for m in mon.moves.values():
                if m.category == MoveCategory.STATUS and m.status:
                    opp_has_status = 1.0
                    break
            if opp_has_status:
                break
    obs.append(opp_has_status)

    # --- Step 5: Toxic escalation counter (1 dim) ---
    toxic_counter = 0.0
    with contextlib.suppress(Exception):
        if own.status == Status.TOX:
            toxic_counter = min(own.status_counter / 15.0, 1.0)
    obs.append(toxic_counter)

    # --- Sprint 7 Step 2: Turn phase (1 dim) ---
    turn_phase = min(battle.turn / 50.0, 1.0) if hasattr(battle, "turn") else 0.0
    obs.append(turn_phase)

    # --- Role features (168 dims = 14 slots × 12 roles) ---
    # Own active
    obs.extend(_role_features(own))
    # Own bench (up to 6 slots, zero-padded)
    own_team = list(battle.team.values())
    for i in range(6):
        if i < len(own_team):
            obs.extend(_role_features(own_team[i]))
        else:
            obs.extend([0.0] * NUM_ROLES)
    # Opponent active
    obs.extend(_role_features(opp))
    # Opponent bench (up to 6 slots, zero-padded — unrevealed => zeros)
    opp_team = list(battle.opponent_team.values())
    for i in range(6):
        if i < len(opp_team):
            obs.extend(_role_features(opp_team[i]))
        else:
            obs.extend([0.0] * NUM_ROLES)

    result = np.array(obs, dtype=np.float32)
    assert result.shape == (_OBS_DIM,), f"Obs shape mismatch: expected ({_OBS_DIM},), got {result.shape}"
    assert np.all(np.isfinite(result)), "Obs contains NaN/Inf"
    return result


class Gen1Env(SinglesEnv):
    """
    Gen 1 random battle environment for MaskablePPO.

    Observation: 1727-dim float32 vector (see embed_battle docstring for layout).
    """

    def __init__(self, shaping_factor: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.shaping_factor = shaping_factor
        self.observation_spaces = {agent: self.describe_embedding() for agent in self.possible_agents}

    def calc_reward(self, battle) -> float:
        reward = self.reward_computing_helper(
            battle,
            fainted_value=0.5,
            hp_value=0.0,
            status_value=0.0,
            victory_value=1.0,
        )
        if battle.won or battle.lost:
            reward -= matchup_baseline(battle)
        return reward

    def embed_battle(self, battle) -> np.ndarray:
        return embed_battle(battle)

    def describe_embedding(self):
        return Box(low=-1.0, high=1.0, shape=(_OBS_DIM,), dtype=np.float32)


class SB3Wrapper(gymnasium.Wrapper):
    """
    Adapts SingleAgentWrapper's dict observations to flat numpy arrays for SB3.

    SingleAgentWrapper returns obs = {"observation": array(1222,), "action_mask": array(N,)}
    This wrapper unpacks it so SB3 sees a plain Box observation and action_masks() method.
    """

    def __init__(self, env: SingleAgentWrapper):
        super().__init__(env)
        self.observation_space = env.observation_space["observation"]
        self._last_action_mask = np.ones(env.action_space.n, dtype=bool)
        self.desync_count: int = 0
        self.step_count: int = 0

    def step(self, action):
        self.step_count += 1
        try:
            obs_dict, reward, terminated, truncated, info = self.env.step(action)
        except ValueError as e:
            # poke-env occasionally desyncs on opponent action. Use neutral reward
            # (0.0) so this doesn't poison training with fake losses.
            self.desync_count += 1
            import logging

            logging.getLogger("pokemon_rl.env").warning(
                "DESYNC #%d (step %d): %s — forfeiting battle", self.desync_count, self.step_count, e
            )
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
        strict=False,  # Gracefully handle invalid opponent orders (fallback to random)
    )

    if opponent_type == "random_damage":
        from src.agents.heuristic_agent import RandomDamagePlayer

        opponent = RandomDamagePlayer(
            battle_format=battle_format,
            account_configuration=AccountConfiguration(f"Puppet{env_index}{suffix}", None),
            start_listening=False,
            log_level=40,
        )
    elif opponent_type == "random_attacker":
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
    elif opponent_type == "mixed_league":
        # Fixed opponent-per-env mapping: diverse opponents every rollout
        _puppet_cfg = AccountConfiguration(f"Puppet{env_index}{suffix}", None)
        _puppet_kw = dict(
            battle_format=battle_format, account_configuration=_puppet_cfg, start_listening=False, log_level=40
        )
        if env_index == 0:
            from src.agents.policy_player import FrozenPolicyPlayer

            if selfplay_model_path is None:
                raise ValueError("mixed_league requires selfplay_model_path for env 0 (self-play slot)")
            opponent = FrozenPolicyPlayer(model_path=selfplay_model_path, **_puppet_kw)
        elif env_index == 1:
            from src.agents.heuristic_agent import MaxDamagePlayer

            opponent = MaxDamagePlayer(**_puppet_kw)
        elif env_index == 2:
            from src.agents.heuristic_agent import TypeMatchupPlayer

            opponent = TypeMatchupPlayer(**_puppet_kw)
        elif env_index == 3:
            from src.agents.heuristic_agent import SoftmaxDamagePlayer

            opponent = SoftmaxDamagePlayer(temperature=opponent_difficulty, **_puppet_kw)
        else:
            raise ValueError(f"mixed_league only supports env_index 0-3, got {env_index}")
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
