"""Dealer module for Startups game."""

from __future__ import annotations

import numpy as np

from games.startups.card import Card
from games.startups.enums import Company


class Dealer:
    """Handles card dealing for Startups game."""

    def __init__(self, random_generator: np.random.Generator | None = None) -> None:
        self._rng = random_generator or np.random.default_rng()
        self._deck: list[Card] = []

    def reset(self, random_generator: np.random.Generator) -> None:
        """Reset dealer with new random generator."""
        self._rng = random_generator
        self._deck = []

    def create_and_shuffle_deck(self) -> None:
        """Create and shuffle a deck of 36 cards (6 companies x 6 values)."""
        self._deck = []
        for company in Company:
            for value in range(1, 7):
                self._deck.append(Card(company=company, value=value))
        self._rng.shuffle(self._deck)

    def deal_one(self) -> Card | None:
        """Deal one card from deck."""
        if not self._deck:
            return None
        return self._deck.pop()

    def deal_to_players(
        self, player_num: int, cards_per_player: int
    ) -> list[list[Card]]:
        """Deal cards to each player."""
        hands = []
        for _ in range(player_num):
            hand = [self._deck.pop() for _ in range(cards_per_player)]
            hands.append(hand)
        return hands

    @property
    def deck_count(self) -> int:
        return len(self._deck)
