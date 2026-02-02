"""Tests for Maskmen game environment."""

import gymnasium as gym
import numpy as np
import pytest

from games.maskmen.constants import PlayerConsts
from games.maskmen.maskmen_game_env import MaskmenGameEnv
from games.oink_game import OinkGameEnv


class TestMaskmenGameEnvContract:
    """Test MaskmenGameEnv adheres to OinkGameEnv contract."""

    @pytest.fixture
    def env(self) -> MaskmenGameEnv:
        return MaskmenGameEnv(player_num=4)

    def test_is_oink_game_env(self, env):
        assert isinstance(env, OinkGameEnv)

    def test_is_gym_env(self, env):
        assert isinstance(env, gym.Env)

    def test_has_observation_space(self, env):
        assert hasattr(env, "observation_space")
        assert isinstance(env.observation_space, gym.Space)

    def test_has_action_space(self, env):
        assert hasattr(env, "action_space")
        assert isinstance(env.action_space, gym.Space)

    @pytest.mark.parametrize(
        argnames="player_num", argvalues=PlayerConsts.ALLOWED_PLAYER_NUM
    )
    def test_valid_player_nums(self, player_num):
        env = MaskmenGameEnv(player_num=player_num)
        assert env.num_players == player_num

    @pytest.mark.parametrize(argnames="invalid_num", argvalues=[1, 7, 0, -1])
    def test_invalid_player_nums(self, invalid_num):
        with pytest.raises(ValueError):
            MaskmenGameEnv(player_num=invalid_num)


class TestMaskmenReset:
    """Test reset functionality."""

    @pytest.fixture
    def env(self) -> MaskmenGameEnv:
        return MaskmenGameEnv(player_num=4)

    def test_reset_returns_observation_and_info(self, env):
        obs, info = env.reset(seed=42)
        assert obs is not None
        assert isinstance(info, dict)

    def test_reset_with_seed(self, env):
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        assert np.array_equal(obs1, obs2)

    def test_reset_different_seeds(self, env):
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=123)
        assert not np.array_equal(obs1, obs2)

    def test_reset_info_contains_action_mask(self, env):
        _, info = env.reset(seed=42)
        assert "action_mask" in info

    def test_reset_sets_current_player_to_zero(self, env):
        env.reset(seed=42)
        assert env.current_player_idx == 0


class TestMaskmenStep:
    """Test step functionality."""

    @pytest.fixture
    def env(self) -> MaskmenGameEnv:
        env = MaskmenGameEnv(player_num=4)
        env.reset(seed=42)
        return env

    def test_step_returns_correct_tuple(self, env):
        _, info = env.reset(seed=42)
        action_mask = info["action_mask"]
        valid_action = action_mask.index(1) if 1 in action_mask else 0

        result = env.step(valid_action)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_advances_player(self, env):
        _, info = env.reset(seed=42)
        action_mask = info["action_mask"]
        valid_action = action_mask.index(1)

        initial_player = env.current_player_idx
        env.step(valid_action)
        assert env.current_player_idx != initial_player or env._game_state.is_terminated


class TestMaskmenObservation:
    """Test observation space."""

    @pytest.fixture
    def env(self) -> MaskmenGameEnv:
        env = MaskmenGameEnv(player_num=4)
        env.reset(seed=42)
        return env

    def test_observation_shape(self, env):
        obs, _ = env.reset(seed=42)
        assert env.observation_space.contains(obs)


class TestMaskmenActionSpace:
    """Test action space."""

    @pytest.fixture
    def env(self) -> MaskmenGameEnv:
        env = MaskmenGameEnv(player_num=4)
        env.reset(seed=42)
        return env

    def test_action_space_type(self, env):
        assert isinstance(env.action_space, gym.spaces.Discrete)

    def test_action_mask_size(self, env):
        _, info = env.reset(seed=42)
        action_mask = info["action_mask"]
        assert len(action_mask) == env.action_space.n

    def test_at_least_one_valid_action(self, env):
        _, info = env.reset(seed=42)
        action_mask = info["action_mask"]
        assert sum(action_mask) > 0


class TestMaskmenRender:
    """Test render functionality."""

    def test_render_human(self, capsys):
        env = MaskmenGameEnv(player_num=4, render_mode="human")
        env.reset(seed=42)
        env.render()
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_render_ansi(self):
        env = MaskmenGameEnv(player_num=4, render_mode="ansi")
        env.reset(seed=42)
        result = env.render()
        assert isinstance(result, str)
        assert "Maskmen" in result


class TestMaskmenGameplay:
    """Test gameplay scenarios."""

    def test_play_until_termination(self):
        env = MaskmenGameEnv(player_num=2)
        _, info = env.reset(seed=42)
        terminated = False
        max_steps = 500

        for _ in range(max_steps):
            action_mask = info["action_mask"]
            valid_actions = [i for i, v in enumerate(action_mask) if v == 1]
            if not valid_actions:
                break

            action = np.random.choice(valid_actions)
            _, _, terminated, _, info = env.step(action)

            if terminated:
                break

        assert terminated, "Game should terminate within max_steps"
