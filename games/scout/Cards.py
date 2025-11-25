from typing import List, Self


class Card:
    def __init__(self, top: int, bottom: int, supported_players: List[int]):
        self.__top = top
        self.__bottom = bottom
        self.__supported_players = supported_players

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

    def __str__(self):
        return f"[{self.top}]/{self.bottom}"

    def __repr__(self):
        return self.__str__()


class Cards: ...


class CardInitilizer: ...
