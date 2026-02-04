"""Tests for Mahjong tile module."""

import pytest

from games.mahjong.enums import TileSuit
from games.mahjong.tile import Tile, create_full_tileset


class TestTile:
    """Test Tile class."""

    def test_tile_creation(self):
        tile = Tile(TileSuit.WAN, 1, 0)
        assert tile.suit == TileSuit.WAN
        assert tile.rank == 1
        assert tile.copy == 0

    def test_tile_str_wan(self):
        tile = Tile(TileSuit.WAN, 5, 0)
        assert str(tile) == "5万"

    def test_tile_str_tiao(self):
        tile = Tile(TileSuit.TIAO, 3, 0)
        assert str(tile) == "3条"

    def test_tile_str_tong(self):
        tile = Tile(TileSuit.TONG, 9, 0)
        assert str(tile) == "9筒"

    def test_tile_str_feng(self):
        assert str(Tile(TileSuit.FENG, 1, 0)) == "东"
        assert str(Tile(TileSuit.FENG, 2, 0)) == "南"
        assert str(Tile(TileSuit.FENG, 3, 0)) == "西"
        assert str(Tile(TileSuit.FENG, 4, 0)) == "北"

    def test_tile_str_jian(self):
        assert str(Tile(TileSuit.JIAN, 1, 0)) == "中"
        assert str(Tile(TileSuit.JIAN, 2, 0)) == "发"
        assert str(Tile(TileSuit.JIAN, 3, 0)) == "白"

    def test_tile_type_id(self):
        # 万: 0-8
        assert Tile(TileSuit.WAN, 1, 0).tile_type_id == 0
        assert Tile(TileSuit.WAN, 9, 0).tile_type_id == 8
        # 条: 9-17
        assert Tile(TileSuit.TIAO, 1, 0).tile_type_id == 9
        assert Tile(TileSuit.TIAO, 9, 0).tile_type_id == 17
        # 筒: 18-26
        assert Tile(TileSuit.TONG, 1, 0).tile_type_id == 18
        assert Tile(TileSuit.TONG, 9, 0).tile_type_id == 26
        # 风: 27-30
        assert Tile(TileSuit.FENG, 1, 0).tile_type_id == 27
        assert Tile(TileSuit.FENG, 4, 0).tile_type_id == 30
        # 箭: 31-33
        assert Tile(TileSuit.JIAN, 1, 0).tile_type_id == 31
        assert Tile(TileSuit.JIAN, 3, 0).tile_type_id == 33

    def test_tile_id(self):
        # tile_id = type_id * 4 + copy
        tile = Tile(TileSuit.WAN, 1, 0)
        assert tile.tile_id == 0
        tile = Tile(TileSuit.WAN, 1, 3)
        assert tile.tile_id == 3
        tile = Tile(TileSuit.JIAN, 3, 3)
        assert tile.tile_id == 33 * 4 + 3

    def test_tile_from_id(self):
        for tile_id in range(136):
            tile = Tile.from_id(tile_id)
            assert tile.tile_id == tile_id

    def test_tile_from_type_id(self):
        tile = Tile.from_type_id(0, 2)
        assert tile.suit == TileSuit.WAN
        assert tile.rank == 1
        assert tile.copy == 2

    def test_tile_ordering(self):
        t1 = Tile(TileSuit.WAN, 1, 0)
        t2 = Tile(TileSuit.WAN, 2, 0)
        assert t1 < t2

    def test_tile_frozen(self):
        tile = Tile(TileSuit.WAN, 1, 0)
        with pytest.raises(AttributeError):
            tile.rank = 2

    def test_is_numbered(self):
        assert Tile(TileSuit.WAN, 1, 0).is_numbered()
        assert Tile(TileSuit.TIAO, 5, 0).is_numbered()
        assert Tile(TileSuit.TONG, 9, 0).is_numbered()
        assert not Tile(TileSuit.FENG, 1, 0).is_numbered()
        assert not Tile(TileSuit.JIAN, 1, 0).is_numbered()

    def test_is_honor(self):
        assert not Tile(TileSuit.WAN, 1, 0).is_honor()
        assert Tile(TileSuit.FENG, 1, 0).is_honor()
        assert Tile(TileSuit.JIAN, 1, 0).is_honor()

    def test_is_terminal(self):
        assert Tile(TileSuit.WAN, 1, 0).is_terminal()
        assert Tile(TileSuit.WAN, 9, 0).is_terminal()
        assert not Tile(TileSuit.WAN, 5, 0).is_terminal()
        assert not Tile(TileSuit.FENG, 1, 0).is_terminal()

    def test_same_type(self):
        t1 = Tile(TileSuit.WAN, 1, 0)
        t2 = Tile(TileSuit.WAN, 1, 1)
        t3 = Tile(TileSuit.WAN, 2, 0)
        assert t1.same_type(t2)
        assert not t1.same_type(t3)


class TestCreateFullTileset:
    """Test create_full_tileset function."""

    def test_tileset_size(self):
        tiles = create_full_tileset()
        assert len(tiles) == 136

    def test_tileset_unique_ids(self):
        tiles = create_full_tileset()
        ids = [t.tile_id for t in tiles]
        assert len(set(ids)) == 136

    def test_tileset_four_copies(self):
        tiles = create_full_tileset()
        from collections import Counter

        type_counts = Counter(t.tile_type_id for t in tiles)
        for count in type_counts.values():
            assert count == 4

    def test_tileset_34_types(self):
        tiles = create_full_tileset()
        types = set(t.tile_type_id for t in tiles)
        assert len(types) == 34


class TestTileEdgeCases:
    """Test edge cases for Tile class."""

    def test_tile_str_unknown_suit(self):
        """Test string representation for unknown suit (defensive code)."""
        tile = Tile(TileSuit.WAN, 1, 0)
        # Use object.__setattr__ to modify frozen dataclass
        object.__setattr__(tile, "suit", 99)  # Invalid suit value
        result = str(tile)
        assert "?" in result

    def test_tile_type_id_unknown_suit(self):
        """Test tile_type_id for unknown suit (defensive code)."""
        tile = Tile(TileSuit.WAN, 1, 0)
        # Use object.__setattr__ to modify frozen dataclass
        object.__setattr__(tile, "suit", 99)  # Invalid suit value
        assert tile.tile_type_id == -1

    def test_tile_type_id_all_suits(self):
        """Test tile_type_id for all valid suits."""
        # WAN
        for rank in range(1, 10):
            assert Tile(TileSuit.WAN, rank, 0).tile_type_id == rank - 1
        # TIAO
        for rank in range(1, 10):
            assert Tile(TileSuit.TIAO, rank, 0).tile_type_id == 9 + rank - 1
        # TONG
        for rank in range(1, 10):
            assert Tile(TileSuit.TONG, rank, 0).tile_type_id == 18 + rank - 1
        # FENG
        for rank in range(1, 5):
            assert Tile(TileSuit.FENG, rank, 0).tile_type_id == 27 + rank - 1
        # JIAN
        for rank in range(1, 4):
            assert Tile(TileSuit.JIAN, rank, 0).tile_type_id == 31 + rank - 1
