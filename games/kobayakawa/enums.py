"""Enums for Kobayakawa game."""

from enum import IntEnum


class GamePhase(IntEnum):
    """Game phases."""

    DRAW_OR_SWAP = 0
    BETTING = 1
    SHOWDOWN = 2
    ROUND_END = 3


class ActionType(IntEnum):
    """Types of actions."""

    DRAW = 0
    SWAP = 1
    BET = 2
    PASS = 3
