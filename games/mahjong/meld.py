"""Meld representation for Mahjong (麻将) game."""

from __future__ import annotations

from dataclasses import dataclass

from games.mahjong.enums import MeldType
from games.mahjong.tile import Tile


@dataclass(frozen=True)
class Meld:
    """A meld (面子) in Mahjong."""

    meld_type: MeldType
    tiles: tuple[Tile, ...]
    from_player: int | None = (
        None  # Player index who discarded the tile (for chi/pong/ming_gang)
    )

    def __str__(self) -> str:
        tiles_str = " ".join(str(t) for t in self.tiles)
        type_names = {
            MeldType.CHI: "吃",
            MeldType.PONG: "碰",
            MeldType.MING_GANG: "明杠",
            MeldType.AN_GANG: "暗杠",
        }
        return f"[{type_names[self.meld_type]}: {tiles_str}]"

    @property
    def is_concealed(self) -> bool:
        """Check if meld is concealed (暗)."""
        return self.meld_type == MeldType.AN_GANG
