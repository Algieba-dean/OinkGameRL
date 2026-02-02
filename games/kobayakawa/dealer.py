"""Dealer module for Kobayakawa game."""

from __future__ import annotations

import numpy as np

from games.kobayakawa.card import Card
from games.kobayakawa.constants import CardConsts


class Dealer:
    """Handles card dealing for Kobayakawa game."""

    def __init__(self, random_generator: np.random.Generator | None = None) -> None:
        self._rng = random_generator or np.random.default_rng()
        self._deck: list[Card] = []

    def reset(self, random_generator: np.random.Generator) -> None:
        """Reset dealer with new random generator."""
        self._rng = random_generator
        self._deck = []

    def create_and_shuffle_deck(self) -> None:
        """Create and shuffle a deck of 15 cards (1-15)."""
        self._deck = [Card(value=v) for v in range(1, CardConsts.TOTAL_CARDS + 1)]
        self._rng.shuffle(self._deck)

    def deal_one(self) -> Card | None:
        """Deal one card from deck."""
        if not self._deck:
            return None
        return self._deck.pop()

    def deal_to_players(self, player_num: int) -> list[Card]:
        """Deal one card to each player.

        Returns:
            List of cards, one per player
        """
        return [self._deck.pop() for _ in range(player_num)]

    @property
    def deck_count(self) -> int:
        return len(self._deck)
