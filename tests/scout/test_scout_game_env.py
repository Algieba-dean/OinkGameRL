import gymnasium as gym
import numpy as np
import pytest

from games.board_game import BoardGameEnv
from games.scout.constants import PlayerConsts
from games.scout.scout_game_env import ScoutGameEnv


class TestScoutGameEnvContract:
    """Test ScoutGameEnv adheres to BoardGameEnv contract."""

    @pytest.fixture
    def env(self) -> ScoutGameEnv:
        return ScoutGameEnv(player_num=4)

    def test_is_board_game_env(self, env):
        assert isinstance(env, BoardGameEnv)

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
        env = ScoutGameEnv(player_num=player_num)
        assert env.num_players == player_num

    @pytest.mark.parametrize(argnames="invalid_num", argvalues=[1, 6, 0, -1])
    def test_invalid_player_nums(self, invalid_num):
        with pytest.raises(ValueError):
            ScoutGameEnv(player_num=invalid_num)

    def test_render_modes(self, env):
        assert "human" in env.metadata["render_modes"]
        assert "json" in env.metadata["render_modes"]
        assert "ansi" in env.metadata["render_modes"]


class TestScoutGameEnvReset:
    """Test reset functionality."""

    @pytest.fixture
    def env(self) -> ScoutGameEnv:
        return ScoutGameEnv(player_num=4)

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
        assert isinstance(info["action_mask"], list)

    def test_reset_info_contains_global_state(self, env):
        _, info = env.reset(seed=42)
        assert "global_state" in info
        assert isinstance(info["global_state"], dict)

    def test_reset_sets_current_player_to_zero(self, env):
        env.reset(seed=42)
        assert env.current_player_idx == 0


class TestScoutGameEnvStep:
    """Test step functionality."""

    @pytest.fixture
    def env(self) -> ScoutGameEnv:
        env = ScoutGameEnv(player_num=4)
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
        valid_action = action_mask.index(1) if 1 in action_mask else 0

        initial_player = env.current_player_idx
        env.step(valid_action)
        assert env.current_player_idx != initial_player or env._game_state.is_terminated

    def test_step_info_contains_action_mask(self, env):
        _, info = env.reset(seed=42)
        action_mask = info["action_mask"]
        valid_action = action_mask.index(1) if 1 in action_mask else 0

        _, _, _, _, step_info = env.step(valid_action)
        assert "action_mask" in step_info

    def test_truncated_always_false(self, env):
        _, info = env.reset(seed=42)
        action_mask = info["action_mask"]
        valid_action = action_mask.index(1) if 1 in action_mask else 0

        _, _, _, truncated, _ = env.step(valid_action)
        assert truncated is False


class TestScoutGameEnvObservation:
    """Test observation space and generation."""

    @pytest.fixture
    def env(self) -> ScoutGameEnv:
        env = ScoutGameEnv(player_num=4)
        env.reset(seed=42)
        return env

    def test_observation_shape(self, env):
        obs, _ = env.reset(seed=42)
        assert env.observation_space.contains(obs)

    def test_observation_per_player(self, env):
        env.reset(seed=42)
        obs0 = env._get_observation(player_idx=0)
        obs1 = env._get_observation(player_idx=1)
        assert not np.array_equal(obs0, obs1)


class TestScoutGameEnvActionSpace:
    """Test action space and action mask."""

    @pytest.fixture
    def env(self) -> ScoutGameEnv:
        env = ScoutGameEnv(player_num=4)
        env.reset(seed=42)
        return env

    def test_action_space_type(self, env):
        assert isinstance(env.action_space, gym.spaces.Discrete)

    def test_action_mask_size(self, env):
        _, info = env.reset(seed=42)
        action_mask = info["action_mask"]
        assert len(action_mask) == env.action_space.n

    def test_action_mask_values_binary(self, env):
        _, info = env.reset(seed=42)
        action_mask = info["action_mask"]
        assert all(v in [0, 1] for v in action_mask)

    def test_at_least_one_valid_action(self, env):
        _, info = env.reset(seed=42)
        action_mask = info["action_mask"]
        assert sum(action_mask) > 0


class TestScoutGameEnvRender:
    """Test render functionality."""

    def test_render_human(self, capsys):
        env = ScoutGameEnv(player_num=4, render_mode="human")
        env.reset(seed=42)
        env.render()
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_render_ansi(self):
        env = ScoutGameEnv(player_num=4, render_mode="ansi")
        env.reset(seed=42)
        result = env.render()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_render_json(self):
        env = ScoutGameEnv(player_num=4, render_mode="json")
        env.reset(seed=42)
        result = env.render()
        assert isinstance(result, dict)


class TestScoutGameEnvGameplay:
    """Test actual gameplay scenarios."""

    @pytest.fixture
    def env(self) -> ScoutGameEnv:
        env = ScoutGameEnv(player_num=2)
        env.reset(seed=42)
        return env

    def test_play_until_termination(self, env):
        """Play random valid actions until game ends."""
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

    def test_final_scores_calculated(self, env):
        """Verify scores are tracked during gameplay."""
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

        global_state = info["global_state"]
        assert "scores" in global_state
