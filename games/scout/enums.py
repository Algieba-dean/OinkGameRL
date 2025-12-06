from enum import Enum, auto


class CardPattern(Enum):
    SAME_RANK = auto()
    SEQUENCE = auto()
    INVALID_PATTERN = auto()
