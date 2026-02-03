"""Tile representation for Mahjong (麻将) game."""

from __future__ import annotations

from dataclasses import dataclass

from games.mahjong.enums import TileSuit


@dataclass(frozen=True)
class Tile:
    """A single tile in Mahjong."""

    suit: TileSuit
    rank: int  # 1-9 for numbered, 1-4 for feng, 1-3 for jian
    copy: int = 0  # 0-3, which copy of this tile

    def __lt__(self, other: Tile) -> bool:
        if self.suit != other.suit:
            return self.suit < other.suit
        if self.rank != other.rank:
            return self.rank < other.rank
        return self.copy < other.copy

    def __str__(self) -> str:
        suit_names = {
            TileSuit.WAN: "万",
            TileSuit.TIAO: "条",
            TileSuit.TONG: "筒",
        }
        feng_names = {1: "东", 2: "南", 3: "西", 4: "北"}
        jian_names = {1: "中", 2: "发", 3: "白"}

        if self.suit in (TileSuit.WAN, TileSuit.TIAO, TileSuit.TONG):
            return f"{self.rank}{suit_names[self.suit]}"
        if self.suit == TileSuit.FENG:
            return feng_names[self.rank]
        if self.suit == TileSuit.JIAN:
            return jian_names[self.rank]
        return f"?{self.suit}:{self.rank}"

    @property
    def tile_type_id(self) -> int:
        """Get tile type ID (0-33), ignoring copy number."""
        if self.suit == TileSuit.WAN:
            return self.rank - 1  # 0-8
        if self.suit == TileSuit.TIAO:
            return 9 + self.rank - 1  # 9-17
        if self.suit == TileSuit.TONG:
            return 18 + self.rank - 1  # 18-26
        if self.suit == TileSuit.FENG:
            return 27 + self.rank - 1  # 27-30
        if self.suit == TileSuit.JIAN:
            return 31 + self.rank - 1  # 31-33
        return -1

    @property
    def tile_id(self) -> int:
        """Get unique tile ID (0-135)."""
        return self.tile_type_id * 4 + self.copy

    @classmethod
    def from_id(cls, tile_id: int) -> Tile:
        """Create tile from unique ID."""
        type_id = tile_id // 4
        copy = tile_id % 4
        return cls.from_type_id(type_id, copy)

    @classmethod
    def from_type_id(cls, type_id: int, copy: int = 0) -> Tile:
        """Create tile from type ID and copy number."""
        if type_id < 9:
            return cls(TileSuit.WAN, type_id + 1, copy)
        if type_id < 18:
            return cls(TileSuit.TIAO, type_id - 9 + 1, copy)
        if type_id < 27:
            return cls(TileSuit.TONG, type_id - 18 + 1, copy)
        if type_id < 31:
            return cls(TileSuit.FENG, type_id - 27 + 1, copy)
        return cls(TileSuit.JIAN, type_id - 31 + 1, copy)

    def is_numbered(self) -> bool:
        """Check if tile is a numbered tile (万条筒)."""
        return self.suit in (TileSuit.WAN, TileSuit.TIAO, TileSuit.TONG)

    def is_honor(self) -> bool:
        """Check if tile is an honor tile (风箭)."""
        return self.suit in (TileSuit.FENG, TileSuit.JIAN)

    def is_terminal(self) -> bool:
        """Check if tile is a terminal (1 or 9)."""
        return self.is_numbered() and self.rank in (1, 9)

    def same_type(self, other: Tile) -> bool:
        """Check if two tiles are the same type (ignoring copy)."""
        return self.suit == other.suit and self.rank == other.rank


def create_full_tileset() -> list[Tile]:
    """Create a full set of 136 tiles."""
    tiles: list[Tile] = []

    # Numbered suits (万条筒): 1-9, 4 copies each
    for suit in (TileSuit.WAN, TileSuit.TIAO, TileSuit.TONG):
        for rank in range(1, 10):
            for copy in range(4):
                tiles.append(Tile(suit, rank, copy))

    # 风 (东南西北): 4 copies each
    for rank in range(1, 5):
        for copy in range(4):
            tiles.append(Tile(TileSuit.FENG, rank, copy))

    # 箭 (中发白): 4 copies each
    for rank in range(1, 4):
        for copy in range(4):
            tiles.append(Tile(TileSuit.JIAN, rank, copy))

    return tiles
