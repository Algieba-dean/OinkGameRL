"""Player module for Kobayakawa game."""

from __future__ import annotations

from games.kobayakawa.card import Card
from games.kobayakawa.constants import PlayerConsts


class Player:
    """Represents a player in Kobayakawa game."""

    def __init__(self, player_idx: int) -> None:
        self.__player_idx = player_idx
        self.__card: Card | None = None
        self.__coins = PlayerConsts.INITIAL_COINS
        self.__has_bet = False

    @property
    def player_idx(self) -> int:
        return self.__player_idx

    @property
    def card(self) -> Card | None:
        return self.__card

    @property
    def coins(self) -> int:
        return self.__coins

    @property
    def has_bet(self) -> bool:
        return self.__has_bet

    @property
    def is_eliminated(self) -> bool:
        return self.__coins <= 0

    def set_card(self, card: Card) -> None:
        """Set player's card."""
        self.__card = card

    def swap_card(self, new_card: Card) -> Card | None:
        """Swap current card with new card, return old card."""
        old_card = self.__card
        self.__card = new_card
        return old_card

    def place_bet(self) -> None:
        """Place a bet (1 coin)."""
        if self.__coins <= 0:
            raise ValueError("Cannot bet with no coins")
        self.__coins -= 1
        self.__has_bet = True

    def win_pot(self, amount: int) -> None:
        """Win coins from pot."""
        self.__coins += amount

    def reset_bet(self) -> None:
        """Reset bet status for new round."""
        self.__has_bet = False

    def reset(self) -> None:
        """Reset player for new game."""
        self.__card = None
        self.__coins = PlayerConsts.INITIAL_COINS
        self.__has_bet = False
