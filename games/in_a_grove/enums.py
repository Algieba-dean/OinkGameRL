"""Enums for In a Grove game."""

from enum import IntEnum


class TileType(IntEnum):
    """Types of voting tiles."""

    CULPRIT = 0
    WITNESS = 1
    ACCOMPLICE = 2


class GamePhase(IntEnum):
    """Game phases."""

    DEALING = 0
    VOTING = 1
    REVEAL = 2
    SCORING = 3
