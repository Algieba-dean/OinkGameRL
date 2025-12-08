from typing import Self


class Card:
    def __init__(
        self, idx: int, top: int, bottom: int, supported_players: list[int]
    ) -> None:
        self.__idx: int = idx
        self.__top: int = top
        self.__bottom: int = bottom
        self.__supported_players: list[int] = supported_players

    def __str__(self) -> str:
        return f"[{self.top}]/{self.bottom}"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, value) -> bool:
        if self.idx != value.idx:
            return False
        return {self.top, self.bottom} == {value.top, value.bottom}

    @property
    def idx(self) -> int:
        return self.__idx

    @property
    def top(self) -> int:
        return self.__top

    @property
    def bottom(self) -> int:
        return self.__bottom

    @property
    def supported_players(self) -> list[int]:
        return self.__supported_players

    def flip(self) -> Self:
        self.__top, self.__bottom = self.__bottom, self.__top
        return self
