import pytest
from typing import SupportsFloat
from games.OinkGame import OinkGameEnv
import gymnasium as gym


class DummyOinkGameEnv(OinkGameEnv):
    OBSERVATION = 101
    GLOBAL_STATE = {"step": 0}
    ACTION_MASK = [1, 1]
    REWARD = 1.0
    TERMINATED = False
    RENDER_TEXT = "DummyOinkGameEnv"

    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)
        self.observation_space = gym.spaces.Discrete(10)
        self.action_space = gym.spaces.Discrete(2)
        self.reset_called_count = 0
        self.step_called_count = 0
        self.render_called_count = 0

    def _apply_action(self, action):
        self.step_called_count += 1
        return self.REWARD, self.TERMINATED

    def _get_action_mask(self, player_idx):
        return self.ACTION_MASK

    def _get_global_state(self):
        return self.GLOBAL_STATE

    def _get_observation(self, player_idx):
        return self.OBSERVATION

    def _reset_logic(self, seed, options):
        self.reset_called_count += 1
        return None

    def _render_text(self):
        self.render_called_count += 1
        return self.RENDER_TEXT


class TestGymContract:
    @pytest.fixture
    def env(self):
        return DummyOinkGameEnv()

    @staticmethod
    def assert_info(info: dict):
        assert isinstance(info, dict)
        assert "global_state" in info.keys()
        assert "action_mask" in info.keys()
        assert isinstance(info.get("action_mask"), list)

    def test_is_gym_environment(self, env):
        assert isinstance(env, gym.Env)

    def test_spaces_exists(self, env):
        assert isinstance(env.observation_space, gym.Space)
        assert isinstance(env.action_space, gym.Space)

    def test_reset_signature(self, env):
        reset_result = env.reset(seed=213)
        assert isinstance(reset_result, tuple)
        assert len(reset_result) == 2
        _, info = reset_result

        self.assert_info(info=info)
        assert env.np_random is not None

    def test_step_signature(self, env):
        env.reset()
        step_result = env.step(action=0)
        assert isinstance(step_result, tuple)
        assert len(step_result) == 5
        _, reward, terminated, truncated, info = step_result
        assert isinstance(reward, SupportsFloat)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

        self.assert_info(info=info)


class TestMultiAgentUsage:
    @pytest.mark.parametrize(
        argnames="render_mode, expected_result",
        argvalues=[
            (None, None),
            ("human", DummyOinkGameEnv.RENDER_TEXT),
            ("ansi", DummyOinkGameEnv.RENDER_TEXT),
            ("json", DummyOinkGameEnv.GLOBAL_STATE),
        ],
        ids=["None", "human", "ansi", "json"],
    )
    def test_render(self, render_mode, expected_result):
        env = DummyOinkGameEnv(render_mode=render_mode)
        result = env.render()
        assert result == expected_result

    def test_invalid_render(self):
        with pytest.raises(
            NotImplementedError, match="render mode .* is not supported"
        ):
            env = DummyOinkGameEnv(render_mode="dummy")

    def test_step(self):
        env = DummyOinkGameEnv()

        env.reset()
        observation, reward, terminated, truncated, info = env.step(action=1)
        assert observation == DummyOinkGameEnv.OBSERVATION
        assert reward == DummyOinkGameEnv.REWARD
        assert terminated == DummyOinkGameEnv.TERMINATED
        assert truncated == False
        assert info == {
            "global_state": DummyOinkGameEnv.GLOBAL_STATE,
            "action_mask": DummyOinkGameEnv.ACTION_MASK,
        }
        assert env.step_called_count == 1

    def test_reset(self):
        env = DummyOinkGameEnv()
        observation, info = env.reset()
        assert observation == DummyOinkGameEnv.OBSERVATION
        assert info == {
            "global_state": DummyOinkGameEnv.GLOBAL_STATE,
            "action_mask": DummyOinkGameEnv.ACTION_MASK,
        }
        assert env.reset_called_count == 1

    # TODO
    # 1. test if the _get_observation,_get_global_state, _get_action_mask works, as we might need them in wrapper
    # 2. test if reset, step, render works
    # 2.1 for step, should check if all returns are back to according player idx, and format
    ...
