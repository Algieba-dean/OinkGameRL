"""Tests for In a Grove Player module."""

import pytest

from games.in_a_grove.card import SuspectCard
from games.in_a_grove.enums import TileType
from games.in_a_grove.player import Player


class TestPlayerInit:
    """Test Player initialization."""

    def test_create_player(self):
        player = Player(0)
        assert player.player_idx == 0
        assert player.hand_count == 0
        assert player.score == 0
        assert len(player.tiles) == 3


class TestPlayerHand:
    """Test Player hand management."""

    def test_set_hand(self):
        player = Player(0)
        cards = [SuspectCard(2), SuspectCard(3)]
        player.set_hand(cards)
        assert player.hand_count == 2

    def test_add_card(self):
        player = Player(0)
        player.add_card(SuspectCard(5))
        assert player.hand_count == 1

    def test_play_card(self):
        player = Player(0)
        cards = [SuspectCard(2), SuspectCard(3)]
        player.set_hand(cards)
        played = player.play_card(0)
        assert played.value == 2
        assert player.hand_count == 1

    def test_play_card_invalid_index(self):
        player = Player(0)
        with pytest.raises(ValueError):
            player.play_card(0)


class TestPlayerVoting:
    """Test Player voting functionality."""

    def test_vote(self):
        player = Player(0)
        player.vote(TileType.CULPRIT)
        assert player.current_vote == TileType.CULPRIT
        assert TileType.CULPRIT not in player.tiles

    def test_vote_invalid_tile(self):
        player = Player(0)
        player.vote(TileType.CULPRIT)
        with pytest.raises(ValueError):
            player.vote(TileType.CULPRIT)

    def test_clear_vote(self):
        player = Player(0)
        player.vote(TileType.CULPRIT)
        player.clear_vote()
        assert player.current_vote is None


class TestPlayerScore:
    """Test Player score management."""

    def test_add_score(self):
        player = Player(0)
        player.add_score(5)
        assert player.score == 5


class TestPlayerReset:
    """Test Player reset functionality."""

    def test_reset_tiles(self):
        player = Player(0)
        player.vote(TileType.CULPRIT)
        player.reset_tiles()
        assert len(player.tiles) == 3

    def test_reset(self):
        player = Player(0)
        player.add_card(SuspectCard(5))
        player.vote(TileType.CULPRIT)
        player.add_score(10)
        player.reset()
        assert player.hand_count == 0
        assert player.score == 0
        assert len(player.tiles) == 3
        assert player.current_vote is None
