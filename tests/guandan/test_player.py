"""Tests for Guandan player module."""

import pytest

from games.guandan.card import Card
from games.guandan.enums import CardRank, CardSuit, Team
from games.guandan.player import Player


class TestPlayer:
    """Test Player class."""

    @pytest.fixture
    def player(self) -> Player:
        return Player(0)

    def test_player_creation(self, player):
        assert player.player_idx == 0
        assert player.hand_count == 0
        assert player.team == Team.TEAM_A
        assert not player.finished
        assert player.finish_order == -1

    def test_team_assignment(self):
        p0 = Player(0)
        p1 = Player(1)
        p2 = Player(2)
        p3 = Player(3)
        assert p0.team == Team.TEAM_A
        assert p1.team == Team.TEAM_B
        assert p2.team == Team.TEAM_A
        assert p3.team == Team.TEAM_B

    def test_set_hand(self, player):
        cards = [
            Card(CardRank.ACE, CardSuit.SPADE, 0),
            Card(CardRank.THREE, CardSuit.HEART, 0),
            Card(CardRank.KING, CardSuit.CLUB, 0),
        ]
        player.set_hand(cards)
        assert player.hand_count == 3

    def test_play_cards(self, player):
        cards = [
            Card(CardRank.THREE, CardSuit.SPADE, 0),
            Card(CardRank.THREE, CardSuit.HEART, 0),
            Card(CardRank.FIVE, CardSuit.CLUB, 0),
        ]
        player.set_hand(cards)

        to_play = [
            Card(CardRank.THREE, CardSuit.SPADE, 0),
            Card(CardRank.THREE, CardSuit.HEART, 0),
        ]
        played = player.play_cards(to_play)

        assert len(played) == 2
        assert player.hand_count == 1

    def test_has_cards_true(self, player):
        cards = [
            Card(CardRank.THREE, CardSuit.SPADE, 0),
            Card(CardRank.THREE, CardSuit.HEART, 0),
        ]
        player.set_hand(cards)
        assert player.has_cards([Card(CardRank.THREE, CardSuit.SPADE, 0)])

    def test_has_cards_false(self, player):
        cards = [Card(CardRank.THREE, CardSuit.SPADE, 0)]
        player.set_hand(cards)
        assert not player.has_cards([Card(CardRank.ACE, CardSuit.HEART, 0)])

    def test_mark_finished(self, player):
        player.mark_finished(1)
        assert player.finished
        assert player.finish_order == 1

    def test_reset(self, player):
        player.set_hand([Card(CardRank.ACE, CardSuit.SPADE, 0)])
        player.mark_finished(1)
        player.reset()
        assert player.hand_count == 0
        assert not player.finished
        assert player.finish_order == -1
