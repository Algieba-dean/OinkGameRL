"""Card module for In a Grove game."""

from __future__ import annotations


class SuspectCard:
    """Represents a suspect card in In a Grove.

    Value 1 is the accomplice, values 2-8 are suspects.
    The highest value card is the culprit.
    """

    def __init__(self, value: int) -> None:
        if value < 1 or value > 8:
            raise ValueError(f"Invalid card value: {value}, must be 1-8")
        self.__value = value

    @property
    def value(self) -> int:
        return self.__value

    @property
    def is_accomplice(self) -> bool:
        return self.__value == 1

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SuspectCard):
            return False
        return self.__value == other.value

    def __lt__(self, other: SuspectCard) -> bool:
        return self.__value < other.value

    def __hash__(self) -> int:
        return hash(self.__value)

    def __repr__(self) -> str:
        return f"SuspectCard({self.__value})"

    def __str__(self) -> str:
        if self.__value == 1:
            return "A"
        return str(self.__value)
