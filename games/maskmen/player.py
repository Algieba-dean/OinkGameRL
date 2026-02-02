"""Player module for Maskmen game."""

from __future__ import annotations

from games.maskmen.card import Card
from games.maskmen.enums import CardColor


class Player:
    """Represents a player in Maskmen game."""

    def __init__(self, player_idx: int, cards: list[Card]) -> None:
        self.__player_idx = player_idx
        self.__hand: list[Card] = list(cards)
        self.__collected_sets: list[CardColor] = []

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
    def collected_sets(self) -> tuple[CardColor, ...]:
        return tuple(self.__collected_sets)

    @property
    def set_count(self) -> int:
        return len(self.__collected_sets)

    def play_card(self, card_idx: int) -> Card:
        """Play a card from hand by index.

        Args:
            card_idx: Index of card to play

        Returns:
            The played card

        Raises:
            ValueError: If index is invalid
        """
        if card_idx < 0 or card_idx >= len(self.__hand):
            raise ValueError(
                f"Invalid card index: {card_idx}, hand size: {len(self.__hand)}"
            )
        return self.__hand.pop(card_idx)

    def add_card(self, card: Card) -> None:
        """Add a card to hand."""
        self.__hand.append(card)

    def collect_set(self, color: CardColor) -> None:
        """Collect a complete set of a color."""
        self.__collected_sets.append(color)

    def has_color(self, color: CardColor) -> bool:
        """Check if player has any card of the given color."""
        return any(card.color == color for card in self.__hand)

    def get_cards_of_color(self, color: CardColor) -> list[Card]:
        """Get all cards of a specific color from hand."""
        return [card for card in self.__hand if card.color == color]

    def reset(self, cards: list[Card]) -> None:
        """Reset player with new cards."""
        self.__hand = list(cards)
        self.__collected_sets = []
