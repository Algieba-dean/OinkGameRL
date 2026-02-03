"""Tests for Doudizhu player module."""

import pytest

from games.doudizhu.card import Card
from games.doudizhu.enums import CardRank, CardSuit, PlayerRole
from games.doudizhu.player import Player


class TestPlayer:
    """Test Player class."""

    @pytest.fixture
    def player(self) -> Player:
        return Player(0)

    def test_player_creation(self, player):
        assert player.player_idx == 0
        assert player.hand_count == 0
        assert player.role == PlayerRole.PEASANT

    def test_set_role(self, player):
        player.set_role(PlayerRole.LANDLORD)
        assert player.role == PlayerRole.LANDLORD

    def test_set_hand(self, player):
        cards = [
            Card(CardRank.ACE, CardSuit.SPADE),
            Card(CardRank.THREE, CardSuit.HEART),
            Card(CardRank.KING, CardSuit.CLUB),
        ]
        player.set_hand(cards)
        assert player.hand_count == 3
        # Should be sorted by rank
        assert player.hand[0].rank == CardRank.THREE
        assert player.hand[1].rank == CardRank.KING
        assert player.hand[2].rank == CardRank.ACE

    def test_add_cards(self, player):
        initial_cards = [Card(CardRank.THREE, CardSuit.SPADE)]
        player.set_hand(initial_cards)

        new_cards = [
            Card(CardRank.ACE, CardSuit.HEART),
            Card(CardRank.FIVE, CardSuit.CLUB),
        ]
        player.add_cards(new_cards)

        assert player.hand_count == 3
        # Should be sorted
        assert player.hand[0].rank == CardRank.THREE
        assert player.hand[1].rank == CardRank.FIVE
        assert player.hand[2].rank == CardRank.ACE

    def test_play_cards(self, player):
        cards = [
            Card(CardRank.THREE, CardSuit.SPADE),
            Card(CardRank.THREE, CardSuit.HEART),
            Card(CardRank.FIVE, CardSuit.CLUB),
        ]
        player.set_hand(cards)

        to_play = [
            Card(CardRank.THREE, CardSuit.SPADE),
            Card(CardRank.THREE, CardSuit.HEART),
        ]
        played = player.play_cards(to_play)

        assert len(played) == 2
        assert player.hand_count == 1
        assert player.hand[0].rank == CardRank.FIVE

    def test_play_cards_not_in_hand(self, player):
        cards = [Card(CardRank.THREE, CardSuit.SPADE)]
        player.set_hand(cards)

        to_play = [Card(CardRank.ACE, CardSuit.HEART)]
        played = player.play_cards(to_play)

        assert len(played) == 0
        assert player.hand_count == 1

    def test_has_cards_true(self, player):
        cards = [
            Card(CardRank.THREE, CardSuit.SPADE),
            Card(CardRank.THREE, CardSuit.HEART),
            Card(CardRank.FIVE, CardSuit.CLUB),
        ]
        player.set_hand(cards)

        check = [
            Card(CardRank.THREE, CardSuit.SPADE),
            Card(CardRank.THREE, CardSuit.HEART),
        ]
        assert player.has_cards(check)

    def test_has_cards_false(self, player):
        cards = [
            Card(CardRank.THREE, CardSuit.SPADE),
            Card(CardRank.FIVE, CardSuit.CLUB),
        ]
        player.set_hand(cards)

        check = [
            Card(CardRank.THREE, CardSuit.SPADE),
            Card(CardRank.THREE, CardSuit.HEART),
        ]
        assert not player.has_cards(check)

    def test_reset(self, player):
        player.set_role(PlayerRole.LANDLORD)
        player.set_hand([Card(CardRank.ACE, CardSuit.SPADE)])

        player.reset()

        assert player.hand_count == 0
        assert player.role == PlayerRole.PEASANT

    def test_hand_property_returns_list(self, player):
        cards = [Card(CardRank.THREE, CardSuit.SPADE)]
        player.set_hand(cards)
        assert isinstance(player.hand, list)
