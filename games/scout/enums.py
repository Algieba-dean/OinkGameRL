from enum import Enum, IntEnum, auto


class CardPattern(Enum):
    SAME_RANK = auto()
    SEQUENCE = auto()
    INVALID_PATTERN = auto()


class GamePhase(Enum): ...


class GameAction(IntEnum):
    PLAY = 1
    SCOUT = 2
    SCOUT_PLAY = 3


class ScoutPosition(IntEnum):
    LEFT = 1
    RIGHT = 2
