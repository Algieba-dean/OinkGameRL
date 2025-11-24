import pytest
from typing import SupportsFloat
from games.OinkGame import OinkGameEnv
import gymnasium as gym


class DummyOinkGameEnv(OinkGameEnv):
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Discrete(10)
        self.action_space = gym.spaces.Discrete(2)

    def _apply_action(self, action):
        return 1.0, False

    def _get_action_mask(self, player_idx):
        return [1, 1]

    def _get_global_state(self):
        return {"step": 0}

    def _get_observation(self, player_idx):
        return 0

    def _reset_logic(self, seed, options):
        return None

    def _render_text(self):
        return f"{__class__}"


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
    # TODO
    # 1. test if the _get_observation,_get_global_state, _get_action_mask works, as we might need them in wrapper
    # 2. test if reset, step, render works
    # 2.1 for step, should check if all returns are back to according player idx, and format
    ...
