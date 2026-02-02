"""Tests for Kobayakawa game environment."""

import gymnasium as gym
import numpy as np
import pytest

from games.kobayakawa.constants import PlayerConsts
from games.kobayakawa.kobayakawa_game_env import KobayakawaGameEnv
from games.oink_game import OinkGameEnv


class TestKobayakawaContract:
    """Test KobayakawaGameEnv adheres to OinkGameEnv contract."""

    @pytest.fixture
    def env(self) -> KobayakawaGameEnv:
        return KobayakawaGameEnv(player_num=4)

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
        env = KobayakawaGameEnv(player_num=player_num)
        assert env.num_players == player_num

    @pytest.mark.parametrize(argnames="invalid_num", argvalues=[1, 2, 7, 0])
    def test_invalid_player_nums(self, invalid_num):
        with pytest.raises(ValueError):
            KobayakawaGameEnv(player_num=invalid_num)


class TestKobayakawaReset:
    """Test reset functionality."""

    @pytest.fixture
    def env(self) -> KobayakawaGameEnv:
        return KobayakawaGameEnv(player_num=4)

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


class TestKobayakawaStep:
    """Test step functionality."""

    @pytest.fixture
    def env(self) -> KobayakawaGameEnv:
        env = KobayakawaGameEnv(player_num=4)
        env.reset(seed=42)
        return env

    def test_step_returns_correct_tuple(self, env):
        _, info = env.reset(seed=42)
        action_mask = info["action_mask"]
        valid_action = action_mask.index(1) if 1 in action_mask else 0

        result = env.step(valid_action)
        assert len(result) == 5

    def test_at_least_one_valid_action(self, env):
        _, info = env.reset(seed=42)
        action_mask = info["action_mask"]
        assert sum(action_mask) > 0


class TestKobayakawaObservation:
    """Test observation space."""

    @pytest.fixture
    def env(self) -> KobayakawaGameEnv:
        env = KobayakawaGameEnv(player_num=4)
        env.reset(seed=42)
        return env

    def test_observation_shape(self, env):
        obs, _ = env.reset(seed=42)
        assert env.observation_space.contains(obs)


class TestKobayakawaRender:
    """Test render functionality."""

    def test_render_ansi(self):
        env = KobayakawaGameEnv(player_num=4, render_mode="ansi")
        env.reset(seed=42)
        result = env.render()
        assert isinstance(result, str)
        assert "Kobayakawa" in result


class TestKobayakawaGameplay:
    """Test gameplay scenarios."""

    def test_play_until_termination(self):
        env = KobayakawaGameEnv(player_num=3)
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

    def test_reset_without_seed(self):
        env = KobayakawaGameEnv(player_num=4)
        obs, info = env.reset()
        assert obs is not None
        assert "action_mask" in info

    def test_get_global_state(self):
        env = KobayakawaGameEnv(player_num=4)
        env.reset(seed=42)
        state = env._get_global_state()
        assert "current_player" in state
        assert "players" in state
        assert "kobayakawa" in state

    def test_get_global_state_before_reset(self):
        env = KobayakawaGameEnv(player_num=4)
        state = env._get_global_state()
        assert state == {}

    def test_action_mask_for_eliminated_player(self):
        env = KobayakawaGameEnv(player_num=3)
        env.reset(seed=42)
        # Eliminate player by using all coins
        player = env._game_state.get_player(0)
        for _ in range(4):
            player.place_bet()
        mask = env._get_action_mask(0)
        assert all(m == 0 for m in mask)

    def test_render_human(self, capsys):
        env = KobayakawaGameEnv(player_num=4, render_mode="human")
        env.reset(seed=42)
        env.render()
        captured = capsys.readouterr()
        assert "Kobayakawa" in captured.out

    def test_winner_reward(self):
        env = KobayakawaGameEnv(player_num=3)
        env.reset(seed=42)
        # Play game until terminated
        terminated = False
        for _ in range(500):
            _, info = env.reset(seed=42)
            for _ in range(200):
                mask = info["action_mask"]
                valid = [i for i, v in enumerate(mask) if v == 1]
                if not valid:
                    break
                _, reward, terminated, _, info = env.step(valid[0])
                if terminated:
                    break
            if terminated:
                break
