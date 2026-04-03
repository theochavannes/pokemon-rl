"""
Unit tests for the Pokemon RL codebase.

Run with: python -m pytest tests/ -v
Requires: pokemon_rl conda env (poke-env, sb3-contrib, torch)
Does NOT require a running Showdown server (all tests use mocks).
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Helpers to build mock battle objects
# ---------------------------------------------------------------------------


def _mock_move(base_power=80, type_value=10, current_pp=15, max_pp=15, accuracy=True, effectiveness=2.0):
    """Create a mock Move object."""
    move = MagicMock()
    move.base_power = base_power
    move.type.value = type_value
    move.current_pp = current_pp
    move.max_pp = max_pp
    move.accuracy = accuracy
    move.type.damage_multiplier.return_value = effectiveness
    move.category = MagicMock()
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
    return battle


# ---------------------------------------------------------------------------
# Test embed_battle
# ---------------------------------------------------------------------------


class TestEmbedBattle:
    def test_output_shape(self):
        from src.env.gen1_env import embed_battle

        battle = _mock_battle()
        obs = embed_battle(battle)
        assert obs.shape == (478,), f"Expected (478,), got {obs.shape}"

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
        assert obs[-3] == 1.0, "trapped flag should be 1.0"
        assert obs[-2] == 1.0, "maybe_trapped flag should be 1.0"

    def test_empty_opp_bench_padding(self):
        from src.env.gen1_env import embed_battle

        battle = _mock_battle(opp_team_size=0)
        obs = embed_battle(battle)
        # With 0 revealed opponents, all 6 opp bench slots should be padded
        # Opp bench starts at: 16 (own active) + 24 (own moves) + 198 (own bench) + 15 (opp active) = 253
        opp_bench_start = 16 + 24 + 198 + 15
        # Each empty slot starts with -1.0
        for i in range(6):
            slot_start = opp_bench_start + i * 33
            assert obs[slot_start] == -1.0, f"Empty opp bench slot {i} should start with -1.0"

    def test_no_moves_pokemon(self):
        """Pokemon with no moves should produce zero-padded move features."""
        from src.env.gen1_env import embed_battle

        battle = _mock_battle()
        # Set own active to have 0 moves
        battle.active_pokemon.moves = {}
        obs = embed_battle(battle)
        # Move section starts at index 16 (after own active), 20 dims total
        move_start = 16
        for i in range(20):
            assert obs[move_start + i] == 0.0, "Empty move slot should be 0.0"

    def test_various_team_sizes(self):
        """Test with different team sizes to make sure padding works."""
        from src.env.gen1_env import embed_battle

        for team_size in [1, 3, 6]:
            for opp_size in [0, 2, 6]:
                battle = _mock_battle(team_size=team_size, opp_team_size=opp_size)
                obs = embed_battle(battle)
                assert obs.shape == (478,)
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
        assert obs.shape == (478,)

        # Minimal teams
        obs = embed_battle(_mock_battle(team_size=1, opp_team_size=1))
        assert obs.shape == (478,)

        # Fainted active (edge case)
        battle = _mock_battle()
        battle.active_pokemon.fainted = True
        battle.active_pokemon.current_hp_fraction = 0.0
        obs = embed_battle(battle)
        assert obs.shape == (478,)
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
