from typing import SupportsFloat

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from games.board_game import BoardGameEnv


class DummyBoardGameEnv(BoardGameEnv):
    OBSERVATION = 5
    GLOBAL_STATE = {"step": 0}
    ACTION_MASK = [1, 1]
    REWARD = 1.0
    TERMINATED = False
    RENDER_TEXT = "DummyBoardGameEnv"

    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)
        self.observation_space = gym.spaces.Discrete(10)
        self.action_space = gym.spaces.Discrete(2)

    def _apply_action(self, action):
        return self.REWARD, self.TERMINATED

    def _get_action_mask(self, player_idx):
        return self.ACTION_MASK

    def _get_global_state(self):
        return self.GLOBAL_STATE

    def _get_observation(self, player_idx):
        return self.OBSERVATION

    def _reset_logic(self, seed, options):
        return None

    def _render_text(self):
        return self.RENDER_TEXT


class TestGymContract:
    @pytest.fixture
    def env(self):
        return DummyBoardGameEnv()

    @staticmethod
    def assert_info(info: dict):
        assert isinstance(info, dict)
        assert "global_state" in info
        assert "action_mask" in info
        assert isinstance(info.get("action_mask"), list)

    def test_gym_compliance(self, env):
        check_env(env.unwrapped, skip_render_check=True)

    def test_is_gym_environment(self, env):
        assert isinstance(env, gym.Env)

    def test_spaces_exists(self, env):
        assert isinstance(env.observation_space, gym.Space)
        assert isinstance(env.action_space, gym.Space)

    def test_current_player_idx_property(self, env):
        assert isinstance(env.current_player_idx, int)
        with pytest.raises(
            AttributeError,
            match="property 'current_player_idx' of '.*' object has no setter",
        ):
            env.current_player_idx = 1

    def test_num_players_property(self, env):
        assert isinstance(env.num_players, int)
        with pytest.raises(
            AttributeError, match="property 'num_players' of '.*' object has no setter"
        ):
            env.num_players = 1

    def test_reset_signature(self, env):
        reset_result = env.reset(seed=213)
        assert isinstance(reset_result, tuple)
        assert len(reset_result) == 2
        obs, info = reset_result
        assert env.observation_space.contains(obs), f"Obdervation {obs} not in space"

        self.assert_info(info=info)
        assert env.np_random is not None

    def test_step_signature(self, env):
        env.reset()
        step_result = env.step(action=0)
        assert isinstance(step_result, tuple)
        assert len(step_result) == 5
        _, reward, terminated, truncated, info = step_result
        assert isinstance(reward, (float | int | np.number | SupportsFloat))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

        self.assert_info(info=info)


class TestEnvInAgentUsage:
    @pytest.mark.parametrize(
        argnames="render_mode, expected_result",
        argvalues=[
            (None, None),
            ("human", DummyBoardGameEnv.RENDER_TEXT),
            ("ansi", DummyBoardGameEnv.RENDER_TEXT),
            ("json", DummyBoardGameEnv.GLOBAL_STATE),
        ],
        ids=["None", "human", "ansi", "json"],
    )
    def test_render(self, render_mode, expected_result):
        env = DummyBoardGameEnv(render_mode=render_mode)
        env.reset()
        result = env.render()
        assert result == expected_result

    def test_invalid_render_on_initialization(self):
        with pytest.raises(
            NotImplementedError, match="render mode .* is not supported"
        ):
            DummyBoardGameEnv(render_mode="dummy_invalid_mode")

    def test_invalid_render_on_render(self):
        env = DummyBoardGameEnv()
        env.reset()
        env.render_mode = "dummy_invalid_mode"
        with pytest.raises(
            NotImplementedError, match="render mode .* is not supported"
        ):
            env.render()

    def test_step(self):
        env = DummyBoardGameEnv()

        env.reset()
        observation, reward, terminated, truncated, info = env.step(action=1)
        assert observation == DummyBoardGameEnv.OBSERVATION
        assert reward == DummyBoardGameEnv.REWARD
        assert terminated == DummyBoardGameEnv.TERMINATED
        assert truncated is False
        assert info == {
            "global_state": DummyBoardGameEnv.GLOBAL_STATE,
            "action_mask": DummyBoardGameEnv.ACTION_MASK,
        }

    def test_reset(self):
        env = DummyBoardGameEnv()
        observation, info = env.reset()
        assert observation == DummyBoardGameEnv.OBSERVATION
        assert info == {
            "global_state": DummyBoardGameEnv.GLOBAL_STATE,
            "action_mask": DummyBoardGameEnv.ACTION_MASK,
        }


class TestBoardGameEnvInteraction:
    @pytest.fixture
    def env(self):
        return DummyBoardGameEnv()

    def test_step_calls_internal_methods(self, env, mocker):
        env.reset()  # as reset will also call internal functions, we did it before spys

        spy_apply_action = mocker.spy(env, "_apply_action")
        spy_get_observation = mocker.spy(env, "_get_observation")
        spy_get_action_mask = mocker.spy(env, "_get_action_mask")
        spy_get_global_state = mocker.spy(env, "_get_global_state")

        action = 1
        current_player_idx = env.current_player_idx
        env.step(action)

        spy_apply_action.assert_called_once_with(action=action)
        spy_get_action_mask.assert_called_once_with(player_idx=current_player_idx)
        spy_get_observation.assert_called_once_with(player_idx=current_player_idx)
        assert spy_get_global_state.call_count == 1

    def test_reset_calls_interal_methods(self, env, mocker):
        spy_reset_logic = mocker.spy(env, "_reset_logic")
        spy_get_observation = mocker.spy(env, "_get_observation")
        spy_get_action_mask = mocker.spy(env, "_get_action_mask")

        seed = 213
        current_player_idx = env.current_player_idx
        env.reset(seed=seed)

        spy_reset_logic.assert_called_once_with(seed=seed, options=None)
        spy_get_action_mask.assert_called_once_with(player_idx=current_player_idx)
        spy_get_observation.assert_called_once_with(player_idx=current_player_idx)

    def test_render_calls_render_text(self, mocker):
        env = DummyBoardGameEnv(render_mode="ansi")
        env.reset()

        spy_render_text = mocker.spy(env, "_render_text")

        env.render()

        spy_render_text.assert_called_once()

    def test_reward_with_mock(self, env, mocker):
        mock_reward = 10.0
        mock_terminated = True
        mocker.patch.object(
            env, "_apply_action", return_value=(mock_reward, mock_terminated)
        )

        env.reset()

        _, reward, terminated, _, _ = env.step(1)
        assert np.isclose(reward, mock_reward, rtol=1e-09, atol=1e-09)
        assert terminated == mock_terminated
