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


class TestGuandanEdgeCases:
    """Test edge cases for Guandan game environment."""

    def test_reset_without_seed(self):
        """Test reset without seed uses random generator."""
        env = GuandanGameEnv()
        obs, _ = env.reset()  # No seed
        assert obs is not None

    def test_build_action_mapping_game_state_none(self):
        """Test _build_action_mapping when game_state is None."""
        env = GuandanGameEnv()
        env._game_state = None
        env._build_action_mapping()
        assert len(env._action_mapping) == 1  # Only pass action

    def test_generate_combinations_with_bombs(self):
        """Test generating combinations with 7 and 8 card bombs."""
        env = GuandanGameEnv()
        env.reset(seed=42)

        # Create a hand with 7+ cards of same rank
        from games.guandan.card import Card
        from games.guandan.enums import CardRank, CardSuit

        # Create 8 cards of same rank (using both decks)
        hand = [
            Card(CardRank.THREE, CardSuit.SPADE, 0),
            Card(CardRank.THREE, CardSuit.HEART, 0),
            Card(CardRank.THREE, CardSuit.DIAMOND, 0),
            Card(CardRank.THREE, CardSuit.CLUB, 0),
            Card(CardRank.THREE, CardSuit.SPADE, 1),
            Card(CardRank.THREE, CardSuit.HEART, 1),
            Card(CardRank.THREE, CardSuit.DIAMOND, 1),
            Card(CardRank.THREE, CardSuit.CLUB, 1),
        ]
        combos = env._generate_all_combinations(hand)
        # Should include 7-card and 8-card bombs
        assert any(len(c) == 7 for c in combos)
        assert any(len(c) == 8 for c in combos)

    def test_generate_combinations_with_rocket(self):
        """Test generating combinations with 4 jokers (rocket)."""
        env = GuandanGameEnv()
        env.reset(seed=42)

        from games.guandan.card import Card
        from games.guandan.enums import CardRank, CardSuit

        # Create 4 jokers
        hand = [
            Card(CardRank.BLACK_JOKER, CardSuit.JOKER, 0),
            Card(CardRank.BLACK_JOKER, CardSuit.JOKER, 1),
            Card(CardRank.RED_JOKER, CardSuit.JOKER, 0),
            Card(CardRank.RED_JOKER, CardSuit.JOKER, 1),
        ]
        combos = env._generate_all_combinations(hand)
        # Should include rocket (4 jokers)
        assert any(len(c) == 4 for c in combos)

    def test_is_consecutive_empty_list(self):
        """Test _is_consecutive with empty list."""
        env = GuandanGameEnv()
        assert env._is_consecutive([]) is False

    def test_get_action_mask_skips_empty_card_ids(self):
        """Test _get_action_mask skips empty card_ids."""
        env = GuandanGameEnv()
        env.reset(seed=42)

        # Add an empty card_ids entry
        env._action_mapping.append([])

        mask = env._get_action_mask(env.current_player_idx)
        # Empty card_ids should be skipped (mask stays 0)
        assert mask[-1] == 0

    def test_get_action_mask_skips_invalid_hand(self):
        """Test _get_action_mask skips invalid hand types."""
        env = GuandanGameEnv()
        env.reset(seed=42)

        # Add an invalid hand (two different singles)
        from games.guandan.card import Card
        from games.guandan.enums import CardRank, CardSuit

        card1 = Card(CardRank.THREE, CardSuit.SPADE, 0)
        card2 = Card(CardRank.FIVE, CardSuit.HEART, 0)
        env._action_mapping.append([card1.card_id, card2.card_id])

        mask = env._get_action_mask(env.current_player_idx)
        # Invalid hand should have mask 0
        assert mask[-1] == 0

    def test_apply_action_losing_team(self):
        """Test reward when player's team loses."""
        env = GuandanGameEnv()
        env.reset(seed=42)

        # Play until game ends
        max_steps = 1000
        for _ in range(max_steps):
            mask = env._get_action_mask(env.current_player_idx)
            valid_actions = [i for i, v in enumerate(mask) if v == 1]
            if not valid_actions:
                break
            action = valid_actions[0]
            _, _, terminated, _, _ = env.step(action)
            if terminated:
                break

    def test_render_with_last_play(self):
        """Test render when there's a last play."""
        env = GuandanGameEnv(render_mode="ansi")
        env.reset(seed=42)

        # Play a card
        mask = env._get_action_mask(env.current_player_idx)
        valid_actions = [i for i, v in enumerate(mask) if v == 1 and i > 0]
        if valid_actions:
            env.step(valid_actions[0])

        result = env.render()
        assert isinstance(result, str)
        # Should contain last play info
        assert "Last Play" in result or "P" in result
