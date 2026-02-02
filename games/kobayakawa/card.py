"""Card module for Kobayakawa game."""

from __future__ import annotations

from games.kobayakawa.constants import CardConsts


class Card:
    """Represents a card in Kobayakawa (value 1-15)."""

    def __init__(self, value: int) -> None:
        if value < CardConsts.MIN_VALUE or value > CardConsts.MAX_VALUE:
            raise ValueError(
                f"Invalid card value: {value}, "
                f"must be {CardConsts.MIN_VALUE}-{CardConsts.MAX_VALUE}"
            )
        self.__value = value

    @property
    def value(self) -> int:
        return self.__value

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Card):
            return False
        return self.__value == other.value

    def __lt__(self, other: Card) -> bool:
        return self.__value < other.value

    def __hash__(self) -> int:
        return hash(self.__value)

    def __repr__(self) -> str:
        return f"Card({self.__value})"

    def __str__(self) -> str:
        return str(self.__value)
