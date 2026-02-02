"""Tests for Startups game environment."""

import gymnasium as gym
import numpy as np
import pytest

from games.oink_game import OinkGameEnv
from games.startups.constants import PlayerConsts
from games.startups.startups_game_env import StartupsGameEnv


class TestStartupsContract:
    """Test StartupsGameEnv adheres to OinkGameEnv contract."""

    @pytest.fixture
    def env(self) -> StartupsGameEnv:
        return StartupsGameEnv(player_num=4)

    def test_is_oink_game_env(self, env):
        assert isinstance(env, OinkGameEnv)

    def test_is_gym_env(self, env):
        assert isinstance(env, gym.Env)

    def test_has_spaces(self, env):
        assert hasattr(env, "observation_space")
        assert hasattr(env, "action_space")

    @pytest.mark.parametrize("player_num", PlayerConsts.ALLOWED_PLAYER_NUM)
    def test_valid_player_nums(self, player_num):
        env = StartupsGameEnv(player_num=player_num)
        assert env.num_players == player_num

    @pytest.mark.parametrize("invalid_num", [1, 2, 8, 0])
    def test_invalid_player_nums(self, invalid_num):
        with pytest.raises(ValueError):
            StartupsGameEnv(player_num=invalid_num)


class TestStartupsReset:
    """Test reset functionality."""

    @pytest.fixture
    def env(self) -> StartupsGameEnv:
        return StartupsGameEnv(player_num=4)

    def test_reset_returns_observation_and_info(self, env):
        obs, info = env.reset(seed=42)
        assert obs is not None
        assert isinstance(info, dict)

    def test_reset_with_seed(self, env):
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        assert np.array_equal(obs1, obs2)

    def test_reset_info_contains_action_mask(self, env):
        _, info = env.reset(seed=42)
        assert "action_mask" in info

    def test_reset_without_seed(self, env):
        obs, info = env.reset()
        assert obs is not None


class TestStartupsStep:
    """Test step functionality."""

    @pytest.fixture
    def env(self) -> StartupsGameEnv:
        env = StartupsGameEnv(player_num=4)
        env.reset(seed=42)
        return env

    def test_step_returns_correct_tuple(self, env):
        _, info = env.reset(seed=42)
        action_mask = info["action_mask"]
        valid_action = action_mask.index(1)
        result = env.step(valid_action)
        assert len(result) == 5

    def test_at_least_one_valid_action(self, env):
        _, info = env.reset(seed=42)
        action_mask = info["action_mask"]
        assert sum(action_mask) > 0


class TestStartupsObservation:
    """Test observation space."""

    def test_observation_shape(self):
        env = StartupsGameEnv(player_num=4)
        obs, _ = env.reset(seed=42)
        assert env.observation_space.contains(obs)


class TestStartupsRender:
    """Test render functionality."""

    def test_render_ansi(self):
        env = StartupsGameEnv(player_num=4, render_mode="ansi")
        env.reset(seed=42)
        result = env.render()
        assert isinstance(result, str)
        assert "Startups" in result


class TestStartupsGameplay:
    """Test gameplay scenarios."""

    def test_play_until_termination(self):
        env = StartupsGameEnv(player_num=3)
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

    def test_get_global_state(self):
        env = StartupsGameEnv(player_num=4)
        env.reset(seed=42)
        state = env._get_global_state()
        assert "current_player" in state
        assert "market" in state

    def test_get_global_state_before_reset(self):
        env = StartupsGameEnv(player_num=4)
        state = env._get_global_state()
        assert state == {}
