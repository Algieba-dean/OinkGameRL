"""Tests for Mahjong hand checker module."""

from games.mahjong.enums import TileSuit
from games.mahjong.hand_checker import HandChecker
from games.mahjong.tile import Tile


class TestIsWinningHand:
    """Test winning hand detection."""

    def test_simple_winning_hand(self):
        # 4 triplets + 1 pair
        hand = [
            Tile(TileSuit.WAN, 1, 0),
            Tile(TileSuit.WAN, 1, 1),
            Tile(TileSuit.WAN, 1, 2),
            Tile(TileSuit.WAN, 2, 0),
            Tile(TileSuit.WAN, 2, 1),
            Tile(TileSuit.WAN, 2, 2),
            Tile(TileSuit.WAN, 3, 0),
            Tile(TileSuit.WAN, 3, 1),
            Tile(TileSuit.WAN, 3, 2),
            Tile(TileSuit.WAN, 4, 0),
            Tile(TileSuit.WAN, 4, 1),
            Tile(TileSuit.WAN, 4, 2),
            Tile(TileSuit.WAN, 5, 0),
            Tile(TileSuit.WAN, 5, 1),
        ]
        assert HandChecker.is_winning_hand(hand, [])

    def test_sequence_winning_hand(self):
        # 4 sequences + 1 pair
        hand = [
            Tile(TileSuit.WAN, 1, 0),
            Tile(TileSuit.WAN, 2, 0),
            Tile(TileSuit.WAN, 3, 0),
            Tile(TileSuit.WAN, 4, 0),
            Tile(TileSuit.WAN, 5, 0),
            Tile(TileSuit.WAN, 6, 0),
            Tile(TileSuit.TIAO, 1, 0),
            Tile(TileSuit.TIAO, 2, 0),
            Tile(TileSuit.TIAO, 3, 0),
            Tile(TileSuit.TONG, 7, 0),
            Tile(TileSuit.TONG, 8, 0),
            Tile(TileSuit.TONG, 9, 0),
            Tile(TileSuit.FENG, 1, 0),
            Tile(TileSuit.FENG, 1, 1),
        ]
        assert HandChecker.is_winning_hand(hand, [])

    def test_non_winning_hand(self):
        hand = [
            Tile(TileSuit.WAN, 1, 0),
            Tile(TileSuit.WAN, 2, 0),
            Tile(TileSuit.WAN, 4, 0),
            Tile(TileSuit.WAN, 5, 0),
            Tile(TileSuit.WAN, 6, 0),
            Tile(TileSuit.WAN, 7, 0),
            Tile(TileSuit.TIAO, 1, 0),
            Tile(TileSuit.TIAO, 2, 0),
            Tile(TileSuit.TIAO, 3, 0),
            Tile(TileSuit.TONG, 7, 0),
            Tile(TileSuit.TONG, 8, 0),
            Tile(TileSuit.TONG, 9, 0),
            Tile(TileSuit.FENG, 1, 0),
            Tile(TileSuit.FENG, 2, 1),
        ]
        assert not HandChecker.is_winning_hand(hand, [])

    def test_wrong_hand_size(self):
        hand = [Tile(TileSuit.WAN, 1, 0)] * 13
        assert not HandChecker.is_winning_hand(hand, [])


class TestCanChi:
    """Test chi detection."""

    def test_can_chi_middle(self):
        hand = [
            Tile(TileSuit.WAN, 1, 0),
            Tile(TileSuit.WAN, 3, 0),
        ]
        discard = Tile(TileSuit.WAN, 2, 0)
        result = HandChecker.can_chi(hand, discard)
        assert len(result) == 1

    def test_can_chi_left(self):
        hand = [
            Tile(TileSuit.WAN, 2, 0),
            Tile(TileSuit.WAN, 3, 0),
        ]
        discard = Tile(TileSuit.WAN, 1, 0)
        result = HandChecker.can_chi(hand, discard)
        assert len(result) == 1

    def test_can_chi_right(self):
        hand = [
            Tile(TileSuit.WAN, 7, 0),
            Tile(TileSuit.WAN, 8, 0),
        ]
        discard = Tile(TileSuit.WAN, 9, 0)
        result = HandChecker.can_chi(hand, discard)
        assert len(result) == 1

    def test_cannot_chi_honor(self):
        hand = [
            Tile(TileSuit.FENG, 1, 0),
            Tile(TileSuit.FENG, 2, 0),
        ]
        discard = Tile(TileSuit.FENG, 3, 0)
        result = HandChecker.can_chi(hand, discard)
        assert len(result) == 0

    def test_cannot_chi_no_tiles(self):
        hand = [Tile(TileSuit.WAN, 5, 0)]
        discard = Tile(TileSuit.WAN, 2, 0)
        result = HandChecker.can_chi(hand, discard)
        assert len(result) == 0


class TestCanPong:
    """Test pong detection."""

    def test_can_pong(self):
        hand = [
            Tile(TileSuit.WAN, 1, 0),
            Tile(TileSuit.WAN, 1, 1),
        ]
        discard = Tile(TileSuit.WAN, 1, 2)
        assert HandChecker.can_pong(hand, discard)

    def test_cannot_pong(self):
        hand = [Tile(TileSuit.WAN, 1, 0)]
        discard = Tile(TileSuit.WAN, 1, 1)
        assert not HandChecker.can_pong(hand, discard)


class TestCanGang:
    """Test gang detection."""

    def test_can_gang(self):
        hand = [
            Tile(TileSuit.WAN, 1, 0),
            Tile(TileSuit.WAN, 1, 1),
            Tile(TileSuit.WAN, 1, 2),
        ]
        discard = Tile(TileSuit.WAN, 1, 3)
        assert HandChecker.can_gang(hand, discard)

    def test_cannot_gang(self):
        hand = [
            Tile(TileSuit.WAN, 1, 0),
            Tile(TileSuit.WAN, 1, 1),
        ]
        discard = Tile(TileSuit.WAN, 1, 2)
        assert not HandChecker.can_gang(hand, discard)


class TestCanAnGang:
    """Test an_gang detection."""

    def test_can_an_gang(self):
        hand = [
            Tile(TileSuit.WAN, 1, 0),
            Tile(TileSuit.WAN, 1, 1),
            Tile(TileSuit.WAN, 1, 2),
            Tile(TileSuit.WAN, 1, 3),
        ]
        result = HandChecker.can_an_gang(hand)
        assert 0 in result  # WAN 1 type_id = 0

    def test_cannot_an_gang(self):
        hand = [
            Tile(TileSuit.WAN, 1, 0),
            Tile(TileSuit.WAN, 1, 1),
            Tile(TileSuit.WAN, 1, 2),
        ]
        result = HandChecker.can_an_gang(hand)
        assert len(result) == 0


class TestIsTenpai:
    """Test tenpai detection."""

    def test_tenpai_one_tile(self):
        # Missing one tile to complete
        hand = [
            Tile(TileSuit.WAN, 1, 0),
            Tile(TileSuit.WAN, 1, 1),
            Tile(TileSuit.WAN, 1, 2),
            Tile(TileSuit.WAN, 2, 0),
            Tile(TileSuit.WAN, 2, 1),
            Tile(TileSuit.WAN, 2, 2),
            Tile(TileSuit.WAN, 3, 0),
            Tile(TileSuit.WAN, 3, 1),
            Tile(TileSuit.WAN, 3, 2),
            Tile(TileSuit.WAN, 4, 0),
            Tile(TileSuit.WAN, 4, 1),
            Tile(TileSuit.WAN, 4, 2),
            Tile(TileSuit.WAN, 5, 0),
        ]
        result = HandChecker.is_tenpai(hand, [])
        assert 4 in result  # WAN 5 type_id = 4

    def test_not_tenpai(self):
        hand = [
            Tile(TileSuit.WAN, 1, 0),
            Tile(TileSuit.WAN, 3, 0),
            Tile(TileSuit.WAN, 5, 0),
            Tile(TileSuit.WAN, 7, 0),
            Tile(TileSuit.WAN, 9, 0),
            Tile(TileSuit.TIAO, 1, 0),
            Tile(TileSuit.TIAO, 3, 0),
            Tile(TileSuit.TIAO, 5, 0),
            Tile(TileSuit.TIAO, 7, 0),
            Tile(TileSuit.TIAO, 9, 0),
            Tile(TileSuit.TONG, 1, 0),
            Tile(TileSuit.TONG, 3, 0),
            Tile(TileSuit.TONG, 5, 0),
        ]
        result = HandChecker.is_tenpai(hand, [])
        assert len(result) == 0


class TestCanFormWinning:
    """Test _can_form_winning helper method."""

    def test_empty_tiles_returns_true(self):
        """Test that empty tiles list returns True."""
        result = HandChecker._can_form_winning([])
        assert result is True
