"""Tests for Startups GameState module."""

import numpy as np
import pytest

from games.startups.enums import Company
from games.startups.game_state import GameState


class TestGameStateInit:
    """Test GameState initialization."""

    def test_create_game_state(self):
        gs = GameState(player_num=4)
        assert gs.player_num == 4
        assert len(gs.players) == 4

    @pytest.mark.parametrize("invalid_num", [1, 2, 8])
    def test_invalid_player_num(self, invalid_num):
        with pytest.raises(ValueError):
            GameState(player_num=invalid_num)


class TestGameStateProperties:
    """Test GameState properties."""

    def test_current_player_idx(self):
        gs = GameState(player_num=4)
        assert gs.current_player_idx == 0

    def test_market(self):
        gs = GameState(player_num=4)
        rng = np.random.default_rng(42)
        gs.reset(rng)
        assert len(gs.market) == 4

    def test_dealer(self):
        gs = GameState(player_num=4)
        assert gs.dealer is not None


class TestGameStateMarket:
    """Test market operations."""

    def test_take_from_market(self):
        gs = GameState(player_num=4)
        rng = np.random.default_rng(42)
        gs.reset(rng)
        initial_count = len(gs.market)
        card = gs.take_from_market(0)
        assert card is not None
        assert len(gs.market) == initial_count - 1

    def test_take_from_market_invalid(self):
        gs = GameState(player_num=4)
        rng = np.random.default_rng(42)
        gs.reset(rng)
        with pytest.raises(ValueError):
            gs.take_from_market(100)


class TestGameStateScoring:
    """Test scoring functionality."""

    def test_calculate_scores(self):
        gs = GameState(player_num=4)
        rng = np.random.default_rng(42)
        gs.reset(rng)
        scores = gs.calculate_scores()
        assert len(scores) == 4

    def test_has_majority(self):
        gs = GameState(player_num=3)
        rng = np.random.default_rng(42)
        gs.reset(rng)
        # Player 0 gets majority in a company
        from games.startups.card import Card

        gs.get_player(0).add_to_tableau(Card(Company.APPY_FIZZ, 1))
        gs.get_player(0).add_to_tableau(Card(Company.APPY_FIZZ, 2))
        assert gs._has_majority(0, Company.APPY_FIZZ)


class TestGameStateTermination:
    """Test game termination."""

    def test_not_terminated_initially(self):
        gs = GameState(player_num=4)
        rng = np.random.default_rng(42)
        gs.reset(rng)
        assert gs.is_terminated is False

    def test_get_winner_not_terminated(self):
        gs = GameState(player_num=4)
        rng = np.random.default_rng(42)
        gs.reset(rng)
        assert gs.get_winner() is None


class TestGameStateNextPlayer:
    """Test next player functionality."""

    def test_next_player(self):
        gs = GameState(player_num=4)
        gs.next_player()
        assert gs.current_player_idx == 1
