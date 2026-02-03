"""Tests for Mahjong game environment."""

import gymnasium as gym
import numpy as np
import pytest

from games.board_game import BoardGameEnv
from games.mahjong.constants import GameConsts
from games.mahjong.mahjong_game_env import MahjongGameEnv


class TestMahjongContract:
    """Test MahjongGameEnv adheres to BoardGameEnv contract."""

    @pytest.fixture
    def env(self) -> MahjongGameEnv:
        return MahjongGameEnv()

    def test_is_board_game_env(self, env):
        assert isinstance(env, BoardGameEnv)

    def test_is_gym_env(self, env):
        assert isinstance(env, gym.Env)

    def test_has_spaces(self, env):
        assert hasattr(env, "observation_space")
        assert hasattr(env, "action_space")

    def test_num_players(self, env):
        assert env.num_players == GameConsts.NUM_PLAYERS


class TestMahjongReset:
    """Test reset functionality."""

    @pytest.fixture
    def env(self) -> MahjongGameEnv:
        return MahjongGameEnv()

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


class TestMahjongStep:
    """Test step functionality."""

    @pytest.fixture
    def env(self) -> MahjongGameEnv:
        env = MahjongGameEnv()
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


class TestMahjongObservation:
    """Test observation space."""

    def test_observation_shape(self):
        env = MahjongGameEnv()
        obs, _ = env.reset(seed=42)
        assert env.observation_space.contains(obs)

    def test_observation_before_reset(self):
        env = MahjongGameEnv()
        obs = env._get_observation(0)
        assert np.all(obs == 0)


class TestMahjongRender:
    """Test render functionality."""

    def test_render_ansi(self):
        env = MahjongGameEnv(render_mode="ansi")
        env.reset(seed=42)
        result = env.render()
        assert isinstance(result, str)
        assert "麻将" in result or "Mahjong" in result

    def test_render_before_reset(self):
        env = MahjongGameEnv(render_mode="ansi")
        result = env._render_text()
        assert result == "Game not initialized"


class TestMahjongGameplay:
    """Test gameplay scenarios."""

    def test_play_until_termination(self):
        env = MahjongGameEnv()
        _, info = env.reset(seed=42)
        terminated = False
        max_steps = 2000  # Mahjong can take many steps

        for _ in range(max_steps):
            action_mask = info["action_mask"]
            valid_actions = [i for i, v in enumerate(action_mask) if v == 1]
            if not valid_actions:
                # No valid actions means game should be over
                terminated = True
                break
            action = np.random.choice(valid_actions)
            _, _, terminated, _, info = env.step(action)
            if terminated:
                break

        assert terminated, "Game should terminate within max_steps"

    def test_get_global_state(self):
        env = MahjongGameEnv()
        env.reset(seed=42)
        state = env._get_global_state()
        assert "current_player" in state
        assert "players" in state
        assert "phase" in state
        assert "wall_count" in state

    def test_get_global_state_before_reset(self):
        env = MahjongGameEnv()
        state = env._get_global_state()
        assert state == {}

    def test_action_mask_before_reset(self):
        env = MahjongGameEnv()
        mask = env._get_action_mask(0)
        assert all(m == 0 for m in mask)

    def test_apply_action_before_reset(self):
        env = MahjongGameEnv()
        reward, terminated = env._apply_action(0)
        assert reward == 0.0
        assert terminated is True

    def test_multiple_games(self):
        """Test playing multiple games in sequence."""
        env = MahjongGameEnv()
        for seed in range(3):
            _, info = env.reset(seed=seed)
            terminated = False
            for _ in range(200):
                action_mask = info["action_mask"]
                valid_actions = [i for i, v in enumerate(action_mask) if v == 1]
                if not valid_actions:
                    break
                action = np.random.choice(valid_actions)
                _, _, terminated, _, info = env.step(action)
                if terminated:
                    break


class TestMahjongActions:
    """Test specific action types."""

    def test_draw_action(self):
        env = MahjongGameEnv()
        _, info = env.reset(seed=42)
        # First action should be draw
        action_mask = info["action_mask"]
        assert action_mask[MahjongGameEnv.ACTION_DRAW] == 1

    def test_discard_after_draw(self):
        env = MahjongGameEnv()
        _, info = env.reset(seed=42)
        # Draw
        env.step(MahjongGameEnv.ACTION_DRAW)
        # Now should be able to discard
        action_mask = env._get_action_mask(env.current_player_idx)
        discard_actions = sum(
            action_mask[
                MahjongGameEnv.ACTION_DISCARD_START : MahjongGameEnv.ACTION_DISCARD_END
                + 1
            ]
        )
        assert discard_actions > 0
