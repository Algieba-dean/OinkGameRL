"""Tests for Mahjong Meld module."""

from games.mahjong.enums import MeldType
from games.mahjong.meld import Meld
from games.mahjong.tile import Tile, TileSuit


class TestMeld:
    """Test Meld class."""

    def test_meld_str_chi(self):
        """Test string representation of chi meld."""
        tiles = (
            Tile(TileSuit.WAN, 1, 0),
            Tile(TileSuit.WAN, 2, 0),
            Tile(TileSuit.WAN, 3, 0),
        )
        meld = Meld(MeldType.CHI, tiles, from_player=0)
        result = str(meld)
        assert "吃" in result

    def test_meld_str_pong(self):
        """Test string representation of pong meld."""
        tiles = (
            Tile(TileSuit.WAN, 1, 0),
            Tile(TileSuit.WAN, 1, 1),
            Tile(TileSuit.WAN, 1, 2),
        )
        meld = Meld(MeldType.PONG, tiles, from_player=1)
        result = str(meld)
        assert "碰" in result

    def test_meld_str_ming_gang(self):
        """Test string representation of ming gang meld."""
        tiles = (
            Tile(TileSuit.WAN, 1, 0),
            Tile(TileSuit.WAN, 1, 1),
            Tile(TileSuit.WAN, 1, 2),
            Tile(TileSuit.WAN, 1, 3),
        )
        meld = Meld(MeldType.MING_GANG, tiles, from_player=2)
        result = str(meld)
        assert "明杠" in result

    def test_meld_str_an_gang(self):
        """Test string representation of an gang meld."""
        tiles = (
            Tile(TileSuit.WAN, 1, 0),
            Tile(TileSuit.WAN, 1, 1),
            Tile(TileSuit.WAN, 1, 2),
            Tile(TileSuit.WAN, 1, 3),
        )
        meld = Meld(MeldType.AN_GANG, tiles, from_player=None)
        result = str(meld)
        assert "暗杠" in result

    def test_is_concealed_an_gang(self):
        """Test that an gang is concealed."""
        tiles = (
            Tile(TileSuit.WAN, 1, 0),
            Tile(TileSuit.WAN, 1, 1),
            Tile(TileSuit.WAN, 1, 2),
            Tile(TileSuit.WAN, 1, 3),
        )
        meld = Meld(MeldType.AN_GANG, tiles, from_player=None)
        assert meld.is_concealed is True

    def test_is_concealed_ming_gang(self):
        """Test that ming gang is not concealed."""
        tiles = (
            Tile(TileSuit.WAN, 1, 0),
            Tile(TileSuit.WAN, 1, 1),
            Tile(TileSuit.WAN, 1, 2),
            Tile(TileSuit.WAN, 1, 3),
        )
        meld = Meld(MeldType.MING_GANG, tiles, from_player=0)
        assert meld.is_concealed is False

    def test_is_concealed_pong(self):
        """Test that pong is not concealed."""
        tiles = (
            Tile(TileSuit.WAN, 1, 0),
            Tile(TileSuit.WAN, 1, 1),
            Tile(TileSuit.WAN, 1, 2),
        )
        meld = Meld(MeldType.PONG, tiles, from_player=1)
        assert meld.is_concealed is False

    def test_is_concealed_chi(self):
        """Test that chi is not concealed."""
        tiles = (
            Tile(TileSuit.WAN, 1, 0),
            Tile(TileSuit.WAN, 2, 0),
            Tile(TileSuit.WAN, 3, 0),
        )
        meld = Meld(MeldType.CHI, tiles, from_player=0)
        assert meld.is_concealed is False
