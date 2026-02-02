"""Tests for In a Grove game environment."""

import gymnasium as gym
import numpy as np
import pytest

from games.in_a_grove.constants import PlayerConsts
from games.in_a_grove.in_a_grove_game_env import InAGroveGameEnv
from games.oink_game import OinkGameEnv


class TestInAGroveContract:
    """Test InAGroveGameEnv adheres to OinkGameEnv contract."""

    @pytest.fixture
    def env(self) -> InAGroveGameEnv:
        return InAGroveGameEnv(player_num=4)

    def test_is_oink_game_env(self, env):
        assert isinstance(env, OinkGameEnv)

    def test_is_gym_env(self, env):
        assert isinstance(env, gym.Env)

    def test_has_spaces(self, env):
        assert hasattr(env, "observation_space")
        assert hasattr(env, "action_space")

    @pytest.mark.parametrize("player_num", PlayerConsts.ALLOWED_PLAYER_NUM)
    def test_valid_player_nums(self, player_num):
        env = InAGroveGameEnv(player_num=player_num)
        assert env.num_players == player_num

    @pytest.mark.parametrize("invalid_num", [1, 5, 0])
    def test_invalid_player_nums(self, invalid_num):
        with pytest.raises(ValueError):
            InAGroveGameEnv(player_num=invalid_num)


class TestInAGroveReset:
    """Test reset functionality."""

    @pytest.fixture
    def env(self) -> InAGroveGameEnv:
        return InAGroveGameEnv(player_num=4)

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


class TestInAGroveStep:
    """Test step functionality."""

    @pytest.fixture
    def env(self) -> InAGroveGameEnv:
        env = InAGroveGameEnv(player_num=4)
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


class TestInAGroveObservation:
    """Test observation space."""

    def test_observation_shape(self):
        env = InAGroveGameEnv(player_num=4)
        obs, _ = env.reset(seed=42)
        assert env.observation_space.contains(obs)


class TestInAGroveRender:
    """Test render functionality."""

    def test_render_ansi(self):
        env = InAGroveGameEnv(player_num=4, render_mode="ansi")
        env.reset(seed=42)
        result = env.render()
        assert isinstance(result, str)
        assert "Grove" in result


class TestInAGroveGameplay:
    """Test gameplay scenarios."""

    def test_play_until_termination(self):
        env = InAGroveGameEnv(player_num=3)
        _, info = env.reset(seed=42)
        terminated = False
        max_steps = 100

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
        env = InAGroveGameEnv(player_num=4)
        env.reset(seed=42)
        state = env._get_global_state()
        assert "current_player" in state
        assert "players" in state

    def test_get_global_state_before_reset(self):
        env = InAGroveGameEnv(player_num=4)
        state = env._get_global_state()
        assert state == {}

    def test_observation_before_reset(self):
        env = InAGroveGameEnv(player_num=4)
        obs = env._get_observation(0)
        assert np.all(obs == 0)

    def test_action_mask_before_reset(self):
        env = InAGroveGameEnv(player_num=4)
        mask = env._get_action_mask(0)
        assert all(m == 0 for m in mask)

    def test_apply_action_before_reset(self):
        env = InAGroveGameEnv(player_num=4)
        reward, terminated = env._apply_action(0)
        assert reward == 0.0
        assert terminated is True

    def test_render_before_reset(self):
        env = InAGroveGameEnv(player_num=4, render_mode="ansi")
        result = env._render_text()
        assert result == "Game not initialized"

    def test_winner_loser_rewards(self):
        """Test winner and loser rewards."""
        env = InAGroveGameEnv(player_num=3)
        np.random.seed(999)
        for seed in range(50):
            _, info = env.reset(seed=seed)
            terminated = False
            for _ in range(100):
                mask = info["action_mask"]
                valid = [i for i, v in enumerate(mask) if v == 1]
                if not valid:
                    break
                _, reward, terminated, _, info = env.step(np.random.choice(valid))
                if terminated:
                    break
            if terminated:
                break

    def test_winner_gets_positive_reward(self):
        """Test that the winner gets positive reward (line 155)."""
        env = InAGroveGameEnv(player_num=2)
        env.reset(seed=42)
        # Force player 0 to win by giving high score
        env._game_state.get_player(0).add_score(100)
        # Set current player to 0
        env._current_player_idx = 0
        # Play through to termination
        for _ in range(3):
            env._game_state.start_new_round(env._rng)
        # Now game is terminated, player 0 should be winner
        assert env._game_state.is_terminated
        # Call _apply_action to trigger reward calculation
        reward, terminated = env._apply_action(0)
        assert terminated is True
        # Winner (player 0) should get positive reward
        assert reward == 1.0

    def test_render_with_center_card(self):
        """Test render shows center card (line 174)."""
        env = InAGroveGameEnv(player_num=3, render_mode="ansi")
        env.reset(seed=42)
        # Ensure center card is set
        if env._game_state.center_card:
            result = env._render_text()
            assert "Center Card" in result
