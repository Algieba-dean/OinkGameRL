"""Player representation for Mahjong (麻将) game."""

from __future__ import annotations

from games.mahjong.meld import Meld
from games.mahjong.tile import Tile


class Player:
    """A player in Mahjong game."""

    def __init__(self, player_idx: int) -> None:
        self._player_idx = player_idx
        self._hand: list[Tile] = []
        self._melds: list[Meld] = []
        self._discards: list[Tile] = []
        self._is_winner: bool = False

    @property
    def player_idx(self) -> int:
        return self._player_idx

    @property
    def hand(self) -> list[Tile]:
        return self._hand

    @property
    def hand_count(self) -> int:
        return len(self._hand)

    @property
    def melds(self) -> list[Meld]:
        return self._melds

    @property
    def discards(self) -> list[Tile]:
        return self._discards

    @property
    def is_winner(self) -> bool:
        return self._is_winner

    def set_hand(self, tiles: list[Tile]) -> None:
        """Set player's hand."""
        self._hand = sorted(tiles)

    def add_tile(self, tile: Tile) -> None:
        """Add a tile to hand (摸牌)."""
        self._hand.append(tile)
        self._hand = sorted(self._hand)

    def remove_tile(self, tile: Tile) -> bool:
        """Remove a tile from hand. Returns True if successful."""
        if tile in self._hand:
            self._hand.remove(tile)
            return True
        return False

    def remove_tiles(self, tiles: list[Tile]) -> bool:
        """Remove multiple tiles from hand."""
        hand_copy = self._hand.copy()
        for tile in tiles:
            if tile in hand_copy:
                hand_copy.remove(tile)
            else:
                return False
        self._hand = hand_copy
        return True

    def discard_tile(self, tile: Tile) -> bool:
        """Discard a tile (出牌)."""
        if self.remove_tile(tile):
            self._discards.append(tile)
            return True
        return False

    def add_meld(self, meld: Meld) -> None:
        """Add a meld."""
        self._melds.append(meld)

    def has_tile(self, tile: Tile) -> bool:
        """Check if player has a specific tile."""
        return tile in self._hand

    def has_tile_type(self, tile_type_id: int) -> bool:
        """Check if player has a tile of given type."""
        return any(t.tile_type_id == tile_type_id for t in self._hand)

    def count_tile_type(self, tile_type_id: int) -> int:
        """Count tiles of given type in hand."""
        return sum(1 for t in self._hand if t.tile_type_id == tile_type_id)

    def get_tiles_of_type(self, tile_type_id: int) -> list[Tile]:
        """Get all tiles of given type from hand."""
        return [t for t in self._hand if t.tile_type_id == tile_type_id]

    def mark_winner(self) -> None:
        """Mark player as winner."""
        self._is_winner = True

    def reset(self) -> None:
        """Reset player state."""
        self._hand = []
        self._melds = []
        self._discards = []
        self._is_winner = False
