"""Player module for Startups game."""

from __future__ import annotations

from games.startups.card import Card
from games.startups.constants import PlayerConsts
from games.startups.enums import Company


class Player:
    """Represents a player in Startups game."""

    def __init__(self, player_idx: int) -> None:
        self.__player_idx = player_idx
        self.__hand: list[Card] = []
        self.__tableau: dict[Company, list[Card]] = {c: [] for c in Company}
        self.__coins = PlayerConsts.INITIAL_COINS

    @property
    def player_idx(self) -> int:
        return self.__player_idx

    @property
    def hand(self) -> tuple[Card, ...]:
        return tuple(self.__hand)

    @property
    def hand_count(self) -> int:
        return len(self.__hand)

    @property
    def tableau(self) -> dict[Company, tuple[Card, ...]]:
        return {c: tuple(cards) for c, cards in self.__tableau.items()}

    @property
    def coins(self) -> int:
        return self.__coins

    def set_hand(self, cards: list[Card]) -> None:
        """Set player's hand."""
        self.__hand = list(cards)

    def add_card_to_hand(self, card: Card) -> None:
        """Add a card to hand."""
        self.__hand.append(card)

    def play_card(self, card_idx: int) -> Card:
        """Play a card from hand by index."""
        if card_idx < 0 or card_idx >= len(self.__hand):
            raise ValueError(f"Invalid card index: {card_idx}")
        return self.__hand.pop(card_idx)

    def add_to_tableau(self, card: Card) -> None:
        """Add a card to tableau."""
        self.__tableau[card.company].append(card)

    def pay_coins(self, amount: int) -> None:
        """Pay coins."""
        if amount > self.__coins:
            raise ValueError(f"Not enough coins: have {self.__coins}, need {amount}")
        self.__coins -= amount

    def receive_coins(self, amount: int) -> None:
        """Receive coins."""
        self.__coins += amount

    def get_company_count(self, company: Company) -> int:
        """Get number of cards for a company in tableau."""
        return len(self.__tableau[company])

    def reset(self) -> None:
        """Reset player for new game."""
        self.__hand = []
        self.__tableau = {c: [] for c in Company}
        self.__coins = PlayerConsts.INITIAL_COINS
