"""Tests for Guandan game environment."""

import gymnasium as gym
import numpy as np
import pytest

from games.board_game import BoardGameEnv
from games.guandan.constants import GameConsts
from games.guandan.guandan_game_env import GuandanGameEnv


class TestGuandanContract:
    """Test GuandanGameEnv adheres to BoardGameEnv contract."""

    @pytest.fixture
    def env(self) -> GuandanGameEnv:
        return GuandanGameEnv()

    def test_is_board_game_env(self, env):
        assert isinstance(env, BoardGameEnv)

    def test_is_gym_env(self, env):
        assert isinstance(env, gym.Env)

    def test_has_spaces(self, env):
        assert hasattr(env, "observation_space")
        assert hasattr(env, "action_space")

    def test_num_players(self, env):
        assert env.num_players == GameConsts.NUM_PLAYERS


class TestGuandanReset:
    """Test reset functionality."""

    @pytest.fixture
    def env(self) -> GuandanGameEnv:
        return GuandanGameEnv()

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

    def test_reset_info_contains_global_state(self, env):
        _, info = env.reset(seed=42)
        assert "global_state" in info


class TestGuandanStep:
    """Test step functionality."""

    @pytest.fixture
    def env(self) -> GuandanGameEnv:
        env = GuandanGameEnv()
        env.reset(seed=42)
        return env

    def test_step_returns_correct_tuple(self, env):
        _, info = env.reset(seed=42)
        action_mask = info["action_mask"]
        valid_action = next((i for i, v in enumerate(action_mask) if v == 1), 0)
        result = env.step(valid_action)
        assert len(result) == 5

    def test_at_least_one_valid_action(self, env):
        _, info = env.reset(seed=42)
        action_mask = info["action_mask"]
        assert sum(action_mask) > 0


class TestGuandanObservation:
    """Test observation space."""

    def test_observation_shape(self):
        env = GuandanGameEnv()
        obs, _ = env.reset(seed=42)
        assert env.observation_space.contains(obs)

    def test_observation_before_reset(self):
        env = GuandanGameEnv()
        obs = env._get_observation(0)
        assert np.all(obs == 0)


class TestGuandanRender:
    """Test render functionality."""

    def test_render_ansi(self):
        env = GuandanGameEnv(render_mode="ansi")
        env.reset(seed=42)
        result = env.render()
        assert isinstance(result, str)
        assert "掼蛋" in result or "Guandan" in result

    def test_render_before_reset(self):
        env = GuandanGameEnv(render_mode="ansi")
        result = env._render_text()
        assert result == "Game not initialized"


class TestGuandanGameplay:
    """Test gameplay scenarios."""

    def test_play_until_termination(self):
        env = GuandanGameEnv()
        _, info = env.reset(seed=42)
        terminated = False
        max_steps = 1000

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
        env = GuandanGameEnv()
        env.reset(seed=42)
        state = env._get_global_state()
        assert "current_player" in state
        assert "players" in state
        assert "phase" in state
        assert "level_rank" in state

    def test_get_global_state_before_reset(self):
        env = GuandanGameEnv()
        state = env._get_global_state()
        assert state == {}

    def test_action_mask_before_reset(self):
        env = GuandanGameEnv()
        mask = env._get_action_mask(0)
        assert all(m == 0 for m in mask)

    def test_apply_action_before_reset(self):
        env = GuandanGameEnv()
        reward, terminated = env._apply_action(0)
        assert reward == 0.0
        assert terminated is True

    def test_multiple_games(self):
        """Test playing multiple games in sequence."""
        env = GuandanGameEnv()
        for seed in range(3):
            _, info = env.reset(seed=seed)
            terminated = False
            for _ in range(500):
                action_mask = info["action_mask"]
                valid_actions = [i for i, v in enumerate(action_mask) if v == 1]
                if not valid_actions:
                    break
                action = np.random.choice(valid_actions)
                _, _, terminated, _, info = env.step(action)
                if terminated:
                    break


class TestGuandanTeams:
    """Test team-based gameplay."""

    def test_players_have_correct_teams(self):
        env = GuandanGameEnv()
        env.reset(seed=42)
        state = env._get_global_state()
        assert state["players"][0]["team"] == "TEAM_A"
        assert state["players"][1]["team"] == "TEAM_B"
        assert state["players"][2]["team"] == "TEAM_A"
        assert state["players"][3]["team"] == "TEAM_B"
