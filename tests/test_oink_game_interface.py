import pytest
from typing import SupportsFloat
from games.OinkGame import OinkGame
import gymnasium as gym


class DummyOinkGame(OinkGame):
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

    def _render_text(self):
        return f"{__class__}"


class TestGymContract:
    @pytest.fixture
    def env(self):
        return DummyOinkGame()

    def test_is_gym_environment(self, env):
        assert isinstance(env, gym.Env)

    def test_spaces_exists(self, env):
        assert isinstance(env.observation_space, gym.Space)
        assert isinstance(env.action_space, gym.Space)

    def test_reset_signature(self, env):
        reset_result = env.reset(seed=213)
        assert isinstance(reset_result, tuple)
        assert len(reset_result) == 2
        obs, info = reset_result

        assert isinstance(info, dict)

        assert env.np_random is not None

    def test_step_signature(self, env):
        env.reset()
        step_result = env.step(action=0)
        assert isinstance(step_result, tuple)
        assert len(step_result) == 5
        observation, reward, terminated, truncated, info = step_result
        assert isinstance(reward, SupportsFloat)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
