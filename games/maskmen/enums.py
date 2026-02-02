"""Enums for Maskmen game."""

from enum import IntEnum


class CardColor(IntEnum):
    """Card colors in Maskmen."""

    RED = 0
    BLUE = 1
    YELLOW = 2
    GREEN = 3
    PURPLE = 4
    ORANGE = 5


class GamePhase(IntEnum):
    """Game phases."""

    SETUP = 0
    PLAYING = 1
    FINISHED = 2


class ActionType(IntEnum):
    """Types of actions in Maskmen."""

    PLAY_CARD = 0
    PASS = 1
