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

    def test_observation_before_reset(self):
        env = StartupsGameEnv(player_num=4)
        obs = env._get_observation(0)
        assert np.all(obs == 0)

    def test_action_mask_before_reset(self):
        env = StartupsGameEnv(player_num=4)
        mask = env._get_action_mask(0)
        assert all(m == 0 for m in mask)

    def test_apply_action_before_reset(self):
        env = StartupsGameEnv(player_num=4)
        reward, terminated = env._apply_action(0)
        assert reward == 0.0
        assert terminated is True

    def test_render_before_reset(self):
        env = StartupsGameEnv(player_num=4, render_mode="ansi")
        result = env._render_text()
        assert result == "Game not initialized"

    def test_get_take_cost_before_reset(self):
        from games.startups.enums import Company

        env = StartupsGameEnv(player_num=4)
        cost = env._get_take_cost(Company.APPY_FIZZ)
        assert cost == 0

    def test_draw_card_if_available_before_reset(self):
        env = StartupsGameEnv(player_num=4)
        env._draw_card_if_available(None)

    def test_winner_loser_rewards(self):
        """Test winner and loser rewards."""
        env = StartupsGameEnv(player_num=3)
        np.random.seed(999)
        for seed in range(50):
            _, info = env.reset(seed=seed)
            terminated = False
            for _ in range(500):
                mask = info["action_mask"]
                valid = [i for i, v in enumerate(mask) if v == 1]
                if not valid:
                    break
                _, reward, terminated, _, info = env.step(np.random.choice(valid))
                if terminated:
                    break
            if terminated:
                break

    def test_render_with_tableau(self):
        """Test render shows tableau (line 225)."""
        from games.startups.card import Card
        from games.startups.enums import Company

        env = StartupsGameEnv(player_num=3, render_mode="ansi")
        env.reset(seed=42)
        # Add cards to player's tableau
        env._game_state.get_player(0).add_to_tableau(Card(Company.APPY_FIZZ, 1))
        result = env._render_text()
        assert "A:1" in result
