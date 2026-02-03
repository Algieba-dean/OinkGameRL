"""Tests for Doudizhu game environment."""

import gymnasium as gym
import numpy as np
import pytest

from games.board_game import BoardGameEnv
from games.doudizhu.constants import GameConsts
from games.doudizhu.doudizhu_game_env import DoudizhuGameEnv


class TestDoudizhuContract:
    """Test DoudizhuGameEnv adheres to BoardGameEnv contract."""

    @pytest.fixture
    def env(self) -> DoudizhuGameEnv:
        return DoudizhuGameEnv()

    def test_is_board_game_env(self, env):
        assert isinstance(env, BoardGameEnv)

    def test_is_gym_env(self, env):
        assert isinstance(env, gym.Env)

    def test_has_spaces(self, env):
        assert hasattr(env, "observation_space")
        assert hasattr(env, "action_space")

    def test_num_players(self, env):
        assert env.num_players == GameConsts.NUM_PLAYERS


class TestDoudizhuReset:
    """Test reset functionality."""

    @pytest.fixture
    def env(self) -> DoudizhuGameEnv:
        return DoudizhuGameEnv()

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

    def test_reset_without_seed(self, env):
        obs, info = env.reset()
        assert obs is not None


class TestDoudizhuStep:
    """Test step functionality."""

    @pytest.fixture
    def env(self) -> DoudizhuGameEnv:
        env = DoudizhuGameEnv()
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


class TestDoudizhuObservation:
    """Test observation space."""

    def test_observation_shape(self):
        env = DoudizhuGameEnv()
        obs, _ = env.reset(seed=42)
        assert env.observation_space.contains(obs)

    def test_observation_before_reset(self):
        env = DoudizhuGameEnv()
        obs = env._get_observation(0)
        assert np.all(obs == 0)


class TestDoudizhuRender:
    """Test render functionality."""

    def test_render_ansi(self):
        env = DoudizhuGameEnv(render_mode="ansi")
        env.reset(seed=42)
        result = env.render()
        assert isinstance(result, str)
        assert "斗地主" in result or "Doudizhu" in result

    def test_render_before_reset(self):
        env = DoudizhuGameEnv(render_mode="ansi")
        result = env._render_text()
        assert result == "Game not initialized"


class TestDoudizhuBidding:
    """Test bidding phase."""

    @pytest.fixture
    def env(self) -> DoudizhuGameEnv:
        env = DoudizhuGameEnv()
        env.reset(seed=42)
        return env

    def test_bidding_actions_available(self, env):
        _, info = env.reset(seed=42)
        action_mask = info["action_mask"]
        # Actions 0 (pass) and 1 (bid) should be available
        assert action_mask[0] == 1
        assert action_mask[1] == 1

    def test_bid_for_landlord(self, env):
        env.reset(seed=42)
        _, _, terminated, _, info = env.step(1)  # Bid for landlord
        assert not terminated
        # Should now be in playing phase
        global_state = info["global_state"]
        assert global_state["phase"] == "PLAYING"

    def test_pass_bidding(self, env):
        env.reset(seed=42)
        env.step(0)  # Pass
        # Next player should be current
        assert env.current_player_idx == 1


class TestDoudizhuPlaying:
    """Test playing phase."""

    @pytest.fixture
    def env(self) -> DoudizhuGameEnv:
        env = DoudizhuGameEnv()
        env.reset(seed=42)
        env.step(1)  # Bid for landlord to enter playing phase
        return env

    def test_playing_has_valid_actions(self, env):
        _, info = env.reset(seed=42)
        env.step(1)  # Bid
        action_mask = env._get_action_mask(env.current_player_idx)
        assert sum(action_mask) > 0

    def test_pass_action_in_playing(self, env):
        # After bidding, landlord plays first
        # Play a card, then next player can pass
        _, info = env.reset(seed=42)
        env.step(1)  # Bid

        # Find a valid play action
        action_mask = env._get_action_mask(env.current_player_idx)
        valid_actions = [i for i, v in enumerate(action_mask) if v == 1 and i > 0]
        if valid_actions:
            env.step(valid_actions[0])  # Play something
            # Now next player should be able to pass
            action_mask = env._get_action_mask(env.current_player_idx)
            assert action_mask[0] == 1  # Pass should be available


class TestDoudizhuGameplay:
    """Test gameplay scenarios."""

    def test_play_until_termination(self):
        env = DoudizhuGameEnv()
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

        # Game should eventually terminate
        assert terminated, "Game should terminate within max_steps"

    def test_get_global_state(self):
        env = DoudizhuGameEnv()
        env.reset(seed=42)
        state = env._get_global_state()
        assert "current_player" in state
        assert "players" in state
        assert "phase" in state

    def test_get_global_state_before_reset(self):
        env = DoudizhuGameEnv()
        state = env._get_global_state()
        assert state == {}

    def test_action_mask_before_reset(self):
        env = DoudizhuGameEnv()
        mask = env._get_action_mask(0)
        assert all(m == 0 for m in mask)

    def test_apply_action_before_reset(self):
        env = DoudizhuGameEnv()
        reward, terminated = env._apply_action(0)
        assert reward == 0.0
        assert terminated is True

    def test_multiple_games(self):
        """Test playing multiple games in sequence."""
        env = DoudizhuGameEnv()
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
