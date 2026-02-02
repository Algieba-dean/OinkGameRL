"""Card module for Startups game."""

from __future__ import annotations

from games.startups.enums import Company


class Card:
    """Represents a card in Startups.

    Each card belongs to a company (0-5) and has a value (1-6).
    """

    def __init__(self, company: Company, value: int) -> None:
        if value < 1 or value > 6:
            raise ValueError(f"Invalid card value: {value}, must be 1-6")
        self.__company = company
        self.__value = value

    @property
    def company(self) -> Company:
        return self.__company

    @property
    def value(self) -> int:
        return self.__value

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Card):
            return False
        return self.__company == other.company and self.__value == other.value

    def __hash__(self) -> int:
        return hash((self.__company, self.__value))

    def __repr__(self) -> str:
        return f"Card({self.__company.name}, {self.__value})"

    def __str__(self) -> str:
        return f"{self.__company.name[0]}{self.__value}"
