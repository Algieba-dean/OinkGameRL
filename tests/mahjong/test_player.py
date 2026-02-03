"""Tests for Mahjong player module."""

import pytest

from games.mahjong.enums import MeldType, TileSuit
from games.mahjong.meld import Meld
from games.mahjong.player import Player
from games.mahjong.tile import Tile


class TestPlayer:
    """Test Player class."""

    @pytest.fixture
    def player(self) -> Player:
        return Player(0)

    def test_player_creation(self, player):
        assert player.player_idx == 0
        assert player.hand_count == 0
        assert len(player.melds) == 0
        assert len(player.discards) == 0
        assert not player.is_winner

    def test_set_hand(self, player):
        tiles = [
            Tile(TileSuit.WAN, 1, 0),
            Tile(TileSuit.WAN, 2, 0),
            Tile(TileSuit.WAN, 3, 0),
        ]
        player.set_hand(tiles)
        assert player.hand_count == 3

    def test_add_tile(self, player):
        player.set_hand([Tile(TileSuit.WAN, 1, 0)])
        player.add_tile(Tile(TileSuit.WAN, 2, 0))
        assert player.hand_count == 2

    def test_remove_tile(self, player):
        tile = Tile(TileSuit.WAN, 1, 0)
        player.set_hand([tile, Tile(TileSuit.WAN, 2, 0)])
        assert player.remove_tile(tile)
        assert player.hand_count == 1

    def test_remove_tile_not_found(self, player):
        player.set_hand([Tile(TileSuit.WAN, 1, 0)])
        assert not player.remove_tile(Tile(TileSuit.WAN, 2, 0))

    def test_remove_tiles(self, player):
        tiles = [
            Tile(TileSuit.WAN, 1, 0),
            Tile(TileSuit.WAN, 2, 0),
            Tile(TileSuit.WAN, 3, 0),
        ]
        player.set_hand(tiles)
        assert player.remove_tiles([tiles[0], tiles[1]])
        assert player.hand_count == 1

    def test_remove_tiles_fail(self, player):
        player.set_hand([Tile(TileSuit.WAN, 1, 0)])
        assert not player.remove_tiles(
            [Tile(TileSuit.WAN, 1, 0), Tile(TileSuit.WAN, 2, 0)]
        )
        assert player.hand_count == 1  # Unchanged

    def test_discard_tile(self, player):
        tile = Tile(TileSuit.WAN, 1, 0)
        player.set_hand([tile])
        assert player.discard_tile(tile)
        assert player.hand_count == 0
        assert len(player.discards) == 1

    def test_discard_tile_not_found(self, player):
        player.set_hand([Tile(TileSuit.WAN, 1, 0)])
        assert not player.discard_tile(Tile(TileSuit.WAN, 2, 0))

    def test_add_meld(self, player):
        meld = Meld(
            MeldType.PONG,
            (
                Tile(TileSuit.WAN, 1, 0),
                Tile(TileSuit.WAN, 1, 1),
                Tile(TileSuit.WAN, 1, 2),
            ),
            1,
        )
        player.add_meld(meld)
        assert len(player.melds) == 1

    def test_has_tile(self, player):
        tile = Tile(TileSuit.WAN, 1, 0)
        player.set_hand([tile])
        assert player.has_tile(tile)
        assert not player.has_tile(Tile(TileSuit.WAN, 2, 0))

    def test_has_tile_type(self, player):
        player.set_hand([Tile(TileSuit.WAN, 1, 0)])
        assert player.has_tile_type(0)  # WAN 1
        assert not player.has_tile_type(1)  # WAN 2

    def test_count_tile_type(self, player):
        player.set_hand(
            [
                Tile(TileSuit.WAN, 1, 0),
                Tile(TileSuit.WAN, 1, 1),
                Tile(TileSuit.WAN, 2, 0),
            ]
        )
        assert player.count_tile_type(0) == 2
        assert player.count_tile_type(1) == 1

    def test_get_tiles_of_type(self, player):
        player.set_hand(
            [
                Tile(TileSuit.WAN, 1, 0),
                Tile(TileSuit.WAN, 1, 1),
                Tile(TileSuit.WAN, 2, 0),
            ]
        )
        tiles = player.get_tiles_of_type(0)
        assert len(tiles) == 2

    def test_mark_winner(self, player):
        player.mark_winner()
        assert player.is_winner

    def test_reset(self, player):
        player.set_hand([Tile(TileSuit.WAN, 1, 0)])
        player.mark_winner()
        player.reset()
        assert player.hand_count == 0
        assert not player.is_winner
