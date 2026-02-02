"""Enums for Startups game."""

from enum import IntEnum


class Company(IntEnum):
    """Company types in Startups."""

    APPY_FIZZ = 0
    BEESWAX = 1
    CRABWALK = 2
    DRIFTWOOD = 3
    ELEPHUN = 4
    FISHTANK = 5


class ActionType(IntEnum):
    """Types of actions in Startups."""

    PLAY_CARD = 0
    TAKE_CARD = 1
    PASS = 2
