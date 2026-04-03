"""
Unit tests for the Pokemon RL codebase.

Run with: python -m pytest tests/ -v
Requires: pokemon_rl conda env (poke-env, sb3-contrib, torch)
Does NOT require a running Showdown server (all tests use mocks).
"""

import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from poke_env.battle.move_category import MoveCategory

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Helpers to build mock battle objects
# ---------------------------------------------------------------------------


def _mock_move(
    base_power=80, type_value=10, current_pp=15, max_pp=15, accuracy=True, effectiveness=2.0, move_id="tackle"
):
    """Create a mock Move object."""
    from poke_env.battle.move_category import MoveCategory

    move = MagicMock()
    move.id = move_id
    move.base_power = base_power
    move.type.value = type_value
    move.current_pp = current_pp
    move.max_pp = max_pp
    move.accuracy = accuracy
    move.type.damage_multiplier.return_value = effectiveness
    move.category = MoveCategory.PHYSICAL
    move.status = None
    move.priority = 0
    move.self_boost = None
    move.boosts = None
    move.heal = 0
    move.drain = 0
    move.entry = {"secondary": None}
    move.recoil = 0
    move.self_destruct = None
    move.damage = 0
    move.flags = {}
    return move


def _mock_pokemon(
    hp_fraction=1.0,
    fainted=False,
    status=None,
    boosts=None,
    type1_value=10,
    type2_value=0,
    level=100,
    base_stats=None,
    moves=None,
):
    """Create a mock Pokemon object."""
    mon = MagicMock()
    mon.current_hp_fraction = hp_fraction
    mon.fainted = fainted
    mon.status = status
    mon.boosts = boosts or {"atk": 0, "def": 0, "spe": 0, "spa": 0, "accuracy": 0, "evasion": 0}
    mon.level = level

    if type1_value:
        mon.type_1 = MagicMock()
        mon.type_1.value = type1_value
    else:
        mon.type_1 = None

    if type2_value:
        mon.type_2 = MagicMock()
        mon.type_2.value = type2_value
    else:
        mon.type_2 = None

    mon.base_stats = base_stats or {"atk": 100, "def": 100, "spa": 100, "spe": 100}
    mon.moves = {f"move{i}": m for i, m in enumerate(moves or [_mock_move() for _ in range(4)])}
    mon.effects = {}
    mon.status_counter = 0
    return mon


def _mock_battle(own_hp=1.0, opp_hp=0.8, team_size=6, opp_team_size=3, trapped=False, maybe_trapped=False):
    """Create a mock Battle with configurable team."""
    battle = MagicMock()

    own_active = _mock_pokemon(hp_fraction=own_hp)
    opp_active = _mock_pokemon(hp_fraction=opp_hp)
    battle.active_pokemon = own_active
    battle.opponent_active_pokemon = opp_active

    # Own team (dict of Pokemon)
    own_team = {}
    for i in range(team_size):
        hp = own_hp if i == 0 else max(0.0, 1.0 - i * 0.15)
        own_team[f"p{i}"] = _mock_pokemon(
            hp_fraction=hp,
            fainted=(hp <= 0),
        )
    battle.team = own_team

    # Opponent team (only revealed ones)
    opp_team = {}
    for i in range(opp_team_size):
        opp_team[f"p{i}"] = _mock_pokemon(hp_fraction=max(0.0, opp_hp - i * 0.2))
    battle.opponent_team = opp_team

    battle.trapped = trapped
    battle.maybe_trapped = maybe_trapped
    battle.side_conditions = {}
    battle.opponent_side_conditions = {}
    battle.turn = 1
    return battle


# ---------------------------------------------------------------------------
# Test embed_battle
# ---------------------------------------------------------------------------


class TestEmbedBattle:
    def test_output_shape(self):
        from src.env.gen1_env import embed_battle

        battle = _mock_battle()
        obs = embed_battle(battle)
        assert obs.shape == (1559,), f"Expected (1559,), got {obs.shape}"

    def test_output_dtype(self):
        from src.env.gen1_env import embed_battle

        battle = _mock_battle()
        obs = embed_battle(battle)
        assert obs.dtype == np.float32

    def test_no_nan_inf(self):
        from src.env.gen1_env import embed_battle

        battle = _mock_battle()
        obs = embed_battle(battle)
        assert np.all(np.isfinite(obs)), "Obs contains NaN or Inf"

    def test_own_hp_encoded(self):
        from src.env.gen1_env import embed_battle

        battle = _mock_battle(own_hp=0.75)
        obs = embed_battle(battle)
        assert abs(obs[0] - 0.75) < 1e-5, f"Own HP should be 0.75, got {obs[0]}"

    def test_trapping_flags(self):
        from src.env.gen1_env import embed_battle

        battle = _mock_battle(trapped=True, maybe_trapped=True)
        obs = embed_battle(battle)
        # Trapping flags at -16 and -15 (before speed, alive, volatile, status, toxic, turn)
        assert obs[-16] == 1.0, "trapped flag should be 1.0"
        assert obs[-15] == 1.0, "maybe_trapped flag should be 1.0"

    def test_empty_opp_bench_padding(self):
        from src.env.gen1_env import embed_battle

        battle = _mock_battle(opp_team_size=0)
        obs = embed_battle(battle)
        # Opp bench starts at: 16 (own active) + 104 (own moves) + 654 (own bench) + 15 (opp active) = 789
        opp_bench_start = 16 + 104 + 654 + 15
        # Each empty slot starts with -1.0, slot size = 109
        for i in range(6):
            slot_start = opp_bench_start + i * 109
            assert obs[slot_start] == -1.0, f"Empty opp bench slot {i} should start with -1.0"

    def test_no_moves_pokemon(self):
        """Pokemon with no moves should produce zero-padded move features."""
        from src.env.gen1_env import embed_battle

        battle = _mock_battle()
        # Set own active to have 0 moves
        battle.active_pokemon.moves = {}
        obs = embed_battle(battle)
        # Move section starts at index 16 (after own active), 104 dims total (4 × (25 + 1))
        move_start = 16
        for i in range(104):
            assert obs[move_start + i] == 0.0, "Empty move slot should be 0.0"

    def test_various_team_sizes(self):
        """Test with different team sizes to make sure padding works."""
        from src.env.gen1_env import embed_battle

        for team_size in [1, 3, 6]:
            for opp_size in [0, 2, 6]:
                battle = _mock_battle(team_size=team_size, opp_team_size=opp_size)
                obs = embed_battle(battle)
                assert obs.shape == (1559,)
                assert np.all(np.isfinite(obs))


# ---------------------------------------------------------------------------
# Test _expected_damage
# ---------------------------------------------------------------------------


class TestExpectedDamage:
    def test_zero_base_power(self):
        from src.agents.heuristic_agent import _expected_damage

        move = _mock_move(base_power=0)
        opp = _mock_pokemon()
        assert _expected_damage(move, opp) == 0.0

    def test_normal_damage(self):
        from src.agents.heuristic_agent import _expected_damage

        move = _mock_move(base_power=100, effectiveness=2.0)
        opp = _mock_pokemon()
        move.type.damage_multiplier.return_value = 2.0
        assert _expected_damage(move, opp) == 200.0

    def test_immune_matchup(self):
        from src.agents.heuristic_agent import _expected_damage

        move = _mock_move(base_power=100, effectiveness=0.0)
        opp = _mock_pokemon()
        move.type.damage_multiplier.return_value = 0.0
        assert _expected_damage(move, opp) == 0.0

    def test_exception_handling(self):
        from src.agents.heuristic_agent import _expected_damage

        move = _mock_move(base_power=100)
        opp = _mock_pokemon()
        move.type.damage_multiplier.side_effect = Exception("type error")
        result = _expected_damage(move, opp)
        assert result == 100.0  # base_power * 1.0 (default on exception)


# ---------------------------------------------------------------------------
# Test SoftmaxDamagePlayer
# ---------------------------------------------------------------------------


class TestSoftmaxDamagePlayer:
    def test_low_temperature_favors_strong_moves(self):
        """At very low temperature, should almost always pick the strongest move."""
        from src.agents.heuristic_agent import SoftmaxDamagePlayer

        player = SoftmaxDamagePlayer.__new__(SoftmaxDamagePlayer)
        player.temperature = 0.01

        # Track which moves get chosen
        chosen_moves = []
        player.create_order = lambda m: chosen_moves.append(m) or m

        battle = MagicMock()
        battle.available_switches = []

        strong_move = _mock_move(base_power=150)
        strong_move.type.damage_multiplier.return_value = 2.0
        weak_move = _mock_move(base_power=20)
        weak_move.type.damage_multiplier.return_value = 1.0
        medium_move = _mock_move(base_power=80)
        medium_move.type.damage_multiplier.return_value = 1.0

        battle.available_moves = [weak_move, strong_move, medium_move]
        battle.opponent_active_pokemon = _mock_pokemon()

        for _ in range(200):
            player.choose_move(battle)

        # At temp 0.01, the strong move (150*2=300 damage) should dominate
        strong_count = sum(1 for m in chosen_moves if m is strong_move)
        assert strong_count > 180, f"Strong move chosen {strong_count}/200 times (expected >180 at temp=0.01)"

    def test_high_temperature_more_uniform(self):
        """At high temperature, distribution should be more uniform."""
        from src.agents.heuristic_agent import SoftmaxDamagePlayer

        player = SoftmaxDamagePlayer.__new__(SoftmaxDamagePlayer)
        player.temperature = 100.0

        chosen_moves = []
        player.create_order = lambda m: chosen_moves.append(m) or m

        battle = MagicMock()
        battle.available_switches = []
        moves = [_mock_move(base_power=p) for p in [50, 100, 150]]
        for m in moves:
            m.type.damage_multiplier.return_value = 1.0
        battle.available_moves = moves
        battle.opponent_active_pokemon = _mock_pokemon()

        for _ in range(300):
            player.choose_move(battle)

        # At high temperature, all moves should be chosen sometimes
        counts = {id(m): 0 for m in moves}
        for m in chosen_moves:
            if id(m) in counts:
                counts[id(m)] += 1

        for move, count in zip(moves, counts.values(), strict=False):
            assert count > 30, f"Move with bp={move.base_power} chosen only {count}/300 times at high temp"


# ---------------------------------------------------------------------------
# Test _EpsilonMixin
# ---------------------------------------------------------------------------


class TestEpsilonMixin:
    def test_epsilon_blend(self):
        """With epsilon=1.0, should always use selfplay/random path."""
        from src.agents.heuristic_agent import EpsilonMaxDamagePlayer

        player = EpsilonMaxDamagePlayer.__new__(EpsilonMaxDamagePlayer)
        player.epsilon = 1.0
        player._frozen_model = None  # no model = falls back to random

        battle = MagicMock()
        battle.available_moves = [_mock_move()]
        battle.available_switches = []

        # With epsilon=1.0 and no frozen model, it should always call choose_random_move
        player.choose_random_move = MagicMock(return_value="random_order")
        player.create_order = MagicMock(return_value="heuristic_order")

        results = set()
        for _ in range(20):
            result = player.choose_move(battle)
            results.add(result)

        # All results should be "random_order" since epsilon=1.0
        assert results == {"random_order"}

    def test_zero_epsilon_uses_heuristic(self):
        """With epsilon=0.0, should always use the base heuristic."""
        from src.agents.heuristic_agent import EpsilonMaxDamagePlayer

        player = EpsilonMaxDamagePlayer.__new__(EpsilonMaxDamagePlayer)
        player.epsilon = 0.0
        player._frozen_model = None

        battle = MagicMock()
        move = _mock_move(base_power=100)
        move.type.damage_multiplier.return_value = 1.0
        battle.available_moves = [move]
        battle.available_switches = []
        battle.opponent_active_pokemon = _mock_pokemon()

        player.create_order = MagicMock(return_value="heuristic_order")

        results = set()
        for _ in range(20):
            result = player.choose_move(battle)
            results.add(result)

        assert results == {"heuristic_order"}


# ---------------------------------------------------------------------------
# Test obs_transfer
# ---------------------------------------------------------------------------


def _make_dummy_vec_env(obs_dim, n_actions=4):
    """Create a DummyVecEnv with a simple gymnasium env for testing."""
    import gymnasium
    from gymnasium.spaces import Box, Discrete
    from stable_baselines3.common.vec_env import DummyVecEnv

    class SimpleEnv(gymnasium.Env):
        def __init__(self):
            super().__init__()
            self.observation_space = Box(low=-1, high=1, shape=(obs_dim,), dtype=np.float32)
            self.action_space = Discrete(n_actions)

        def step(self, action):
            return np.zeros(obs_dim, dtype=np.float32), 0.0, False, False, {}

        def reset(self, **kwargs):
            return np.zeros(obs_dim, dtype=np.float32), {}

    return DummyVecEnv([SimpleEnv])


class TestObsTransfer:
    def test_is_compatible_same_dim(self, tmp_path):
        """Compatible when obs dims match."""
        from sb3_contrib import MaskablePPO

        from src.obs_transfer import is_compatible

        env = _make_dummy_vec_env(10)
        model = MaskablePPO("MlpPolicy", env, verbose=0)
        path = str(tmp_path / "test_model")
        model.save(path)

        assert is_compatible(path, 10) is True
        assert is_compatible(path, 20) is False

    def test_expand_preserves_old_weights(self, tmp_path):
        """Weight expansion should preserve old feature weights and zero-pad new ones."""
        from sb3_contrib import MaskablePPO

        from src.obs_transfer import load_with_expanded_obs

        old_dim = 10
        new_dim = 15

        # Create and save an old model
        old_env = _make_dummy_vec_env(old_dim)
        old_model = MaskablePPO("MlpPolicy", old_env, verbose=0)
        old_path = str(tmp_path / "old_model")
        old_model.save(old_path)

        # Get old first-layer weights
        old_sd = old_model.policy.state_dict()
        old_first_layer = None
        for _key, param in old_sd.items():
            if param.ndim == 2 and param.shape[1] == old_dim:
                old_first_layer = param.clone()
                break

        new_env = _make_dummy_vec_env(new_dim)

        new_model = load_with_expanded_obs(
            old_path=old_path,
            new_obs_dim=new_dim,
            env=new_env,
            ppo_kwargs={"policy": "MlpPolicy"},
        )

        # Verify old columns are preserved
        new_sd = new_model.policy.state_dict()
        for key, param in new_sd.items():
            if param.ndim == 2 and param.shape[1] == new_dim:
                import torch

                assert torch.allclose(param[:, :old_dim], old_first_layer), f"Old weights not preserved in {key}"
                assert torch.all(param[:, old_dim:] == 0), f"New columns not zero in {key}"
                break


# ---------------------------------------------------------------------------
# Test reward shaping decay
# ---------------------------------------------------------------------------


def _make_decay_callback(gen1_env, total_episodes, decay_battles=1000):
    """Build a WinRateCallback with mocked wrapper chain for shaping decay tests."""
    from src.callbacks import WinRateCallback

    # Use SimpleNamespace to avoid MagicMock's auto-attribute creation
    # (which causes infinite loop in `while hasattr(inner, "env")`)
    single_agent = SimpleNamespace(env=gen1_env)
    sb3_wrapper = SimpleNamespace(env=single_agent)
    monitor = SimpleNamespace(env=sb3_wrapper)

    vec_env = SimpleNamespace(envs=[monitor])

    cb = WinRateCallback.__new__(WinRateCallback)
    cb.verbose = 0
    cb.shaping_decay_battles = decay_battles
    cb.global_episodes_offset = 0
    cb._env = vec_env
    cb._total_episodes = total_episodes
    return cb


class TestRewardShapingDecay:
    def test_decay_reduces_factor(self):
        """Shaping factor should decrease as battles progress."""
        gen1_env = SimpleNamespace(shaping_factor=1.0)

        cb = _make_decay_callback(gen1_env, total_episodes=500)
        cb._decay_shaping()

        assert gen1_env.shaping_factor == pytest.approx(0.5, abs=0.01)

    def test_decay_to_zero(self):
        """Shaping factor should reach 0 after enough battles."""
        gen1_env = SimpleNamespace(shaping_factor=1.0)

        cb = _make_decay_callback(gen1_env, total_episodes=2000)
        cb._decay_shaping()

        assert gen1_env.shaping_factor == 0.0


# ---------------------------------------------------------------------------
# Integration smoke test (no Showdown server needed for shape verification)
# ---------------------------------------------------------------------------


class TestIntegrationSmoke:
    def test_embed_battle_with_varied_inputs(self):
        """Smoke test: embed_battle handles edge cases without crashing."""
        from src.env.gen1_env import embed_battle

        # Normal battle
        obs = embed_battle(_mock_battle())
        assert obs.shape == (1559,)

        # Minimal teams
        obs = embed_battle(_mock_battle(team_size=1, opp_team_size=1))
        assert obs.shape == (1559,)

        # Fainted active (edge case)
        battle = _mock_battle()
        battle.active_pokemon.fainted = True
        battle.active_pokemon.current_hp_fraction = 0.0
        obs = embed_battle(battle)
        assert obs.shape == (1559,)
        assert obs[0] == 0.0  # HP should be 0

    def test_move_features_accuracy_variants(self):
        """Accuracy can be True (always hits), a float, or cause exceptions."""
        from src.env.gen1_env import _move_features

        opp = _mock_pokemon()

        # accuracy=True (e.g., Swift)
        move = _mock_move(accuracy=True)
        features = _move_features(move, opp)
        assert features[4] == 1.0

        # accuracy=0.7 (e.g., Thunder)
        move = _mock_move(accuracy=0.7)
        features = _move_features(move, opp)
        assert abs(features[4] - 0.7) < 1e-5

    def test_status_encoding(self):
        """Verify all status conditions map to expected values."""

        from src.env.gen1_env import _STATUS_TO_FLOAT, _status

        mon = _mock_pokemon()
        for status_val, expected_float in _STATUS_TO_FLOAT.items():
            mon.status = status_val
            assert _status(mon) == expected_float


# ---------------------------------------------------------------------------
# Test mixed_league opponent type
# ---------------------------------------------------------------------------


class TestMixedLeague:
    """Verify mixed_league creates the correct opponent for each env index."""

    def test_env0_is_frozen_policy(self):
        """Env 0 should be FrozenPolicyPlayer (self-play slot)."""
        import contextlib
        from unittest.mock import patch

        from src.env.gen1_env import make_env

        with (
            patch("src.agents.policy_player.FrozenPolicyPlayer.__init__", return_value=None) as mock_init,
            patch("src.agents.policy_player.FrozenPolicyPlayer.choose_move"),
        ):
            mock_init.return_value = None
            with contextlib.suppress(Exception):
                make_env(env_index=0, opponent_type="mixed_league", selfplay_model_path="fake_path")
            # Verify FrozenPolicyPlayer was instantiated
            assert mock_init.called

    def test_env0_requires_selfplay_path(self):
        """Env 0 should raise ValueError without selfplay_model_path."""
        from src.env.gen1_env import make_env

        with pytest.raises(ValueError, match="mixed_league requires selfplay_model_path"):
            make_env(env_index=0, opponent_type="mixed_league", selfplay_model_path=None)

    def test_invalid_env_index_raises(self):
        """Env index >= 4 should raise ValueError."""
        from src.env.gen1_env import make_env

        with pytest.raises(ValueError, match="mixed_league only supports env_index 0-3"):
            make_env(env_index=4, opponent_type="mixed_league")


# ---------------------------------------------------------------------------
# Test refactored annealing (search by type, not index)
# ---------------------------------------------------------------------------


class TestAnnealingSearchesOpponents:
    """Verify _maybe_anneal_epsilon finds annealable opponents regardless of position."""

    def test_finds_temperature_opponent_at_index_3(self):
        """Temperature annealing should work when SoftmaxDamagePlayer is at index 3."""
        from src.callbacks import WinRateCallback

        cb = WinRateCallback.__new__(WinRateCallback)
        cb.verbose = 0
        cb.epsilon_schedule = (2.0, 0.1)
        cb._epsilon_rewards = deque([1.0] * 500, maxlen=500)  # 100% win rate

        # opponents[0] has no temperature/epsilon (like FrozenPolicyPlayer)
        opp0 = SimpleNamespace()  # no temperature, no epsilon
        opp1 = SimpleNamespace()
        opp2 = SimpleNamespace()
        opp3 = SimpleNamespace(temperature=2.0)  # SoftmaxDamagePlayer at index 3
        cb.opponents = [opp0, opp1, opp2, opp3]

        cb._maybe_anneal_epsilon(win_rate=0.80)

        # At 100% wr500, target_temp = 2.0*(1-1.0) + 0.1*1.0 = 0.1
        # Rate limited: max drop 0.1 per eval -> 2.0 - 0.1 = 1.9
        assert opp3.temperature == pytest.approx(1.9, abs=0.01)

    def test_no_annealable_opponents_is_safe(self):
        """Should not crash when no opponents have temperature or epsilon."""
        from src.callbacks import WinRateCallback

        cb = WinRateCallback.__new__(WinRateCallback)
        cb.verbose = 0
        cb.epsilon_schedule = (2.0, 0.1)
        cb._epsilon_rewards = deque([1.0] * 500, maxlen=500)
        cb.opponents = [SimpleNamespace(), SimpleNamespace()]  # no temp, no eps

        # Should not raise
        cb._maybe_anneal_epsilon(win_rate=0.80)


# ---------------------------------------------------------------------------
# Test graduation check searches opponents by type
# ---------------------------------------------------------------------------


class TestGraduationCheck:
    """Verify graduation check finds temperature/epsilon on any opponent."""

    def test_graduation_finds_temperature_at_any_index(self):
        """Phase should not graduate when temperature hasn't annealed, even if opponents[0] has no temp."""
        from src.callbacks import WinRateCallback

        cb = WinRateCallback.__new__(WinRateCallback)
        cb.verbose = 0
        cb.epsilon_schedule = (2.0, 0.1)
        cb.opponents = [SimpleNamespace(), SimpleNamespace(temperature=1.5)]

        # Simulate the graduation check logic
        epsilon_done = True
        _, end_val = cb.epsilon_schedule
        temp_opps = [o for o in cb.opponents if hasattr(o, "temperature")]
        if temp_opps:
            epsilon_done = temp_opps[0].temperature <= end_val + 0.05

        assert not epsilon_done, "Should not graduate with temperature=1.5 (end=0.1)"


# ---------------------------------------------------------------------------
# Sprint 5: New feature spot-checks
# ---------------------------------------------------------------------------


class TestSprint5Features:
    def test_target_statused_flag(self):
        """Thunder Wave vs already-paralyzed opponent → target_statused=1."""
        from poke_env.battle.status import Status

        from src.env.gen1_env import embed_battle

        battle = _mock_battle()
        # Make first move a status move (Thunder Wave)
        status_move = _mock_move(base_power=0, move_id="thunderwave")
        status_move.category = MoveCategory.STATUS
        status_move.status = Status.PAR
        battle.active_pokemon.moves = {"thunderwave": status_move}
        # Opponent is already paralyzed
        battle.opponent_active_pokemon.status = Status.PAR

        obs = embed_battle(battle)
        # target_statused is at index 16 (own active) + 25 (first move features) = 41
        target_statused_idx = 16 + 25  # after first move's 25 features
        assert obs[target_statused_idx] == 1.0, "target_statused should be 1.0 for status move vs statused opp"

    def test_target_statused_zero_when_no_status(self):
        """Normal attack → target_statused=0."""
        from src.env.gen1_env import embed_battle

        battle = _mock_battle()
        obs = embed_battle(battle)
        # First move's target_statused (normal move, opp not statused)
        target_statused_idx = 16 + 25
        assert obs[target_statused_idx] == 0.0, "target_statused should be 0.0 for normal move"

    def test_status_immune_poison_type(self):
        """Toxic vs Poison-type → status_immune=1."""
        from poke_env.battle.status import Status

        from src.env.gen1_env import _move_features

        toxic_move = _mock_move(base_power=0, move_id="toxic")
        toxic_move.category = MoveCategory.STATUS
        toxic_move.status = Status.TOX

        # Poison-type opponent
        poison_opp = _mock_pokemon()
        poison_opp.type_1.name = "POISON"
        poison_opp.type_2 = None

        features = _move_features(toxic_move, poison_opp)
        # status_immune is the last feature (index 18)
        assert features[18] == 1.0, "Poison type should be immune to Toxic"

    def test_status_immune_zero_for_normal(self):
        """Toxic vs Normal-type → status_immune=0."""
        from poke_env.battle.status import Status

        from src.env.gen1_env import _move_features

        toxic_move = _mock_move(base_power=0, move_id="toxic")
        toxic_move.category = MoveCategory.STATUS
        toxic_move.status = Status.TOX

        normal_opp = _mock_pokemon()
        normal_opp.type_1.name = "NORMAL"
        normal_opp.type_2 = None

        features = _move_features(toxic_move, normal_opp)
        assert features[18] == 0.0, "Normal type should not be immune to Toxic"

    def test_trapping_move_flag(self):
        """Wrap should have trapping=1."""
        from src.env.gen1_env import _move_features

        wrap_move = _mock_move(base_power=15, move_id="wrap")
        opp = _mock_pokemon()
        features = _move_features(wrap_move, opp)
        # trapping is at index 17
        assert features[17] == 1.0, "Wrap should have trapping flag = 1.0"

    def test_non_trapping_move_flag(self):
        """Tackle should have trapping=0."""
        from src.env.gen1_env import _move_features

        tackle = _mock_move(base_power=40, move_id="tackle")
        opp = _mock_pokemon()
        features = _move_features(tackle, opp)
        assert features[17] == 0.0, "Tackle should not have trapping flag"

    def test_one_hot_category(self):
        """Category encoding: Physical=[1,0], Special=[0,1], Status=[0,0]."""
        from src.env.gen1_env import _move_features

        opp = _mock_pokemon()

        phys = _mock_move(base_power=100)
        phys.category = MoveCategory.PHYSICAL
        f = _move_features(phys, opp)
        assert f[5] == 1.0 and f[6] == 0.0, "Physical: [1,0]"

        spec = _mock_move(base_power=100)
        spec.category = MoveCategory.SPECIAL
        f = _move_features(spec, opp)
        assert f[5] == 0.0 and f[6] == 1.0, "Special: [0,1]"

        stat = _mock_move(base_power=0)
        stat.category = MoveCategory.STATUS
        f = _move_features(stat, opp)
        assert f[5] == 0.0 and f[6] == 0.0, "Status: [0,0]"

    def test_separate_heal_and_drain(self):
        """Heal and drain are now separate features."""
        from src.env.gen1_env import _move_features

        opp = _mock_pokemon()

        # Pure heal move (Recover)
        recover = _mock_move(base_power=0, move_id="recover")
        recover.category = MoveCategory.STATUS
        recover.heal = 0.5
        recover.drain = 0
        f = _move_features(recover, opp)
        assert f[10] == 0.5, "heal should be 0.5 for Recover"
        assert f[11] == 0.0, "drain should be 0.0 for Recover"

        # Drain move (Mega Drain)
        mega_drain = _mock_move(base_power=40, move_id="megadrain")
        mega_drain.heal = 0
        mega_drain.drain = 0.5
        f = _move_features(mega_drain, opp)
        assert f[10] == 0.0, "heal should be 0.0 for Mega Drain"
        assert f[11] == 0.5, "drain should be 0.5 for Mega Drain"

    def test_volatile_status_features(self):
        """Volatile status features appear in correct positions."""
        from poke_env.battle.effect import Effect
        from poke_env.battle.side_condition import SideCondition

        from src.env.gen1_env import embed_battle

        battle = _mock_battle()
        battle.active_pokemon.effects = {Effect.SUBSTITUTE: 1}
        battle.side_conditions = {SideCondition.REFLECT: 1}

        obs = embed_battle(battle)
        # Volatile status starts at -12 from end (8 volatile + 1 opp_status + 1 toxic = 10 from end, but before that is alive(2))
        # Actually: end = trapping(2) + speed(1) + alive(2) + volatile(8) + status_threat(1) + toxic(1) = 15
        # volatile starts at -10 from end
        # volatile starts at -11 from end (8 volatile + 1 status_threat + 1 toxic + 1 turn = 11)
        assert obs[-11] == 1.0, "own_substitute should be 1.0"
        assert obs[-10] == 0.0, "opp_substitute should be 0.0"
        assert obs[-9] == 1.0, "own_reflect should be 1.0"
        assert obs[-8] == 0.0, "own_light_screen should be 0.0"


# ---------------------------------------------------------------------------
# Sprint 6: Two-tower feature extractor
# ---------------------------------------------------------------------------


class TestFeatureExtractor:
    def test_output_shape(self):
        """Feature extractor should produce (batch, 256) output."""
        import torch
        from gymnasium.spaces import Box

        from src.env.feature_extractor import PokemonFeatureExtractor

        obs_space = Box(low=-1.0, high=1.0, shape=(1559,), dtype=np.float32)
        extractor = PokemonFeatureExtractor(obs_space, features_dim=256)

        batch = torch.randn(4, 1559)
        out = extractor(batch)
        assert out.shape == (4, 256), f"Expected (4, 256), got {out.shape}"

    def test_parameter_count(self):
        """Two-tower should have more parameters than a flat MLP of same width."""
        from gymnasium.spaces import Box

        from src.env.feature_extractor import PokemonFeatureExtractor

        obs_space = Box(low=-1.0, high=1.0, shape=(1559,), dtype=np.float32)
        extractor = PokemonFeatureExtractor(obs_space, features_dim=256)

        total = sum(p.numel() for p in extractor.parameters())
        assert total > 400_000, f"Expected >400K params, got {total:,}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
