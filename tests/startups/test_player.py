"""Tests for Startups Player module."""

import pytest

from games.startups.card import Card
from games.startups.constants import PlayerConsts
from games.startups.enums import Company
from games.startups.player import Player


class TestPlayerInit:
    """Test Player initialization."""

    def test_create_player(self):
        player = Player(0)
        assert player.player_idx == 0
        assert player.hand_count == 0
        assert player.coins == PlayerConsts.INITIAL_COINS


class TestPlayerHand:
    """Test Player hand management."""

    def test_set_hand(self):
        player = Player(0)
        cards = [Card(Company.APPY_FIZZ, 1), Card(Company.BEESWAX, 2)]
        player.set_hand(cards)
        assert player.hand_count == 2

    def test_add_card_to_hand(self):
        player = Player(0)
        card = Card(Company.APPY_FIZZ, 1)
        player.add_card_to_hand(card)
        assert player.hand_count == 1

    def test_play_card(self):
        player = Player(0)
        cards = [Card(Company.APPY_FIZZ, 1), Card(Company.BEESWAX, 2)]
        player.set_hand(cards)
        played = player.play_card(0)
        assert played.company == Company.APPY_FIZZ
        assert player.hand_count == 1

    def test_play_card_invalid_index(self):
        player = Player(0)
        with pytest.raises(ValueError):
            player.play_card(0)


class TestPlayerTableau:
    """Test Player tableau management."""

    def test_add_to_tableau(self):
        player = Player(0)
        card = Card(Company.APPY_FIZZ, 1)
        player.add_to_tableau(card)
        assert player.get_company_count(Company.APPY_FIZZ) == 1

    def test_get_company_count(self):
        player = Player(0)
        assert player.get_company_count(Company.APPY_FIZZ) == 0


class TestPlayerCoins:
    """Test Player coin management."""

    def test_pay_coins(self):
        player = Player(0)
        player.pay_coins(5)
        assert player.coins == PlayerConsts.INITIAL_COINS - 5

    def test_pay_coins_not_enough(self):
        player = Player(0)
        with pytest.raises(ValueError):
            player.pay_coins(100)

    def test_receive_coins(self):
        player = Player(0)
        player.receive_coins(5)
        assert player.coins == PlayerConsts.INITIAL_COINS + 5


class TestPlayerReset:
    """Test Player reset."""

    def test_reset(self):
        player = Player(0)
        player.add_card_to_hand(Card(Company.APPY_FIZZ, 1))
        player.add_to_tableau(Card(Company.BEESWAX, 2))
        player.pay_coins(5)
        player.reset()
        assert player.hand_count == 0
        assert player.coins == PlayerConsts.INITIAL_COINS
