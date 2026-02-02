"""Card module for Maskmen game."""

from __future__ import annotations

from games.maskmen.enums import CardColor


class Card:
    """Represents a card in Maskmen.

    Each card has a color and a value (1-3).
    """

    def __init__(self, color: CardColor, value: int) -> None:
        if value not in (1, 2, 3):
            raise ValueError(f"Invalid card value: {value}, must be 1, 2, or 3")
        self.__color = color
        self.__value = value

    @property
    def color(self) -> CardColor:
        return self.__color

    @property
    def value(self) -> int:
        return self.__value

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Card):
            return False
        return self.__color == other.color and self.__value == other.value

    def __hash__(self) -> int:
        return hash((self.__color, self.__value))

    def __repr__(self) -> str:
        return f"Card({self.__color.name}, {self.__value})"

    def __str__(self) -> str:
        return f"{self.__color.name[0]}{self.__value}"
