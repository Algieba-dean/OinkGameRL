from functools import total_ordering
from typing import Self


@total_ordering
class Card:
    def __init__(self, idx: int, top: int, bottom: int, supported_players: list[int]):
        self.__idx: int = idx
        self.__top: int = top
        self.__bottom: int = bottom
        self.__supported_players: list[int] = supported_players

    def __str__(self):
        return f"[{self.top}]/{self.bottom}"

    def __repr__(self):
        return self.__str__()

    def __lt__(self, value):
        return self.top < value.top

    def __eq__(self, value):
        return self.top == value.top

    @property
    def idx(self):
        return self.__idx

    @property
    def top(self):
        return self.__top

    @property
    def bottom(self):
        return self.__bottom

    @property
    def supported_players(self):
        return self.__supported_players

    def flip(self) -> Self:
        self.__top, self.__bottom = self.__bottom, self.__top
        return self


class Cards: ...


class CardInitilizer: ...
