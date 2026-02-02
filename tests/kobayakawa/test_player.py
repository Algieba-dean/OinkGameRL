"""Tests for Kobayakawa Player module."""

import pytest

from games.kobayakawa.card import Card
from games.kobayakawa.constants import PlayerConsts
from games.kobayakawa.player import Player


class TestPlayerInit:
    """Test Player initialization."""

    def test_create_player(self):
        player = Player(player_idx=0)
        assert player.player_idx == 0
        assert player.card is None
        assert player.coins == PlayerConsts.INITIAL_COINS
        assert player.has_bet is False
        assert player.is_eliminated is False


class TestPlayerCard:
    """Test card management."""

    def test_set_card(self):
        player = Player(player_idx=0)
        card = Card(value=5)
        player.set_card(card)
        assert player.card == card

    def test_swap_card(self):
        player = Player(player_idx=0)
        card1 = Card(value=5)
        card2 = Card(value=10)
        player.set_card(card1)
        old_card = player.swap_card(card2)
        assert player.card == card2
        assert old_card == card1

    def test_swap_card_when_empty(self):
        player = Player(player_idx=0)
        card = Card(value=5)
        old_card = player.swap_card(card)
        assert player.card == card
        assert old_card is None


class TestPlayerBetting:
    """Test betting functionality."""

    def test_place_bet(self):
        player = Player(player_idx=0)
        initial_coins = player.coins
        player.place_bet()
        assert player.coins == initial_coins - 1
        assert player.has_bet is True

    def test_place_bet_no_coins_raises(self):
        player = Player(player_idx=0)
        # Use all coins (INITIAL_COINS = 4)
        for _ in range(PlayerConsts.INITIAL_COINS):
            player.place_bet()
        # Now player has 0 coins, next bet should raise
        with pytest.raises(ValueError, match="Cannot bet"):
            player.place_bet()

    def test_win_pot(self):
        player = Player(player_idx=0)
        initial_coins = player.coins
        player.win_pot(5)
        assert player.coins == initial_coins + 5

    def test_reset_bet(self):
        player = Player(player_idx=0)
        player.place_bet()
        player.reset_bet()
        assert player.has_bet is False


class TestPlayerElimination:
    """Test elimination status."""

    def test_not_eliminated_with_coins(self):
        player = Player(player_idx=0)
        assert player.is_eliminated is False

    def test_eliminated_when_no_coins(self):
        player = Player(player_idx=0)
        # Use all coins (INITIAL_COINS = 4)
        for _ in range(PlayerConsts.INITIAL_COINS):
            player.place_bet()
        # Now has 0 coins
        assert player.is_eliminated is True


class TestPlayerReset:
    """Test reset method."""

    def test_reset_restores_state(self):
        player = Player(player_idx=0)
        player.set_card(Card(value=5))
        player.place_bet()
        player.reset()
        assert player.card is None
        assert player.coins == PlayerConsts.INITIAL_COINS
        assert player.has_bet is False
