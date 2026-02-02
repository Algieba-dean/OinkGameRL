"""Dealer module for Maskmen game."""

from __future__ import annotations

import numpy as np

from games.maskmen.card import Card
from games.maskmen.constants import GameConsts
from games.maskmen.enums import CardColor


class Dealer:
    """Handles card dealing for Maskmen game."""

    def __init__(self, random_generator: np.random.Generator | None = None) -> None:
        self._rng = random_generator or np.random.default_rng()

    def reset(self, random_generator: np.random.Generator) -> None:
        """Reset dealer with new random generator."""
        self._rng = random_generator

    def create_deck(self) -> list[Card]:
        """Create a full deck of 18 cards (6 colors x 3 values)."""
        deck = []
        for color in CardColor:
            for value in (1, 2, 3):
                deck.append(Card(color=color, value=value))
        return deck

    def shuffle_deck(self, deck: list[Card]) -> list[Card]:
        """Shuffle the deck and return a new shuffled list."""
        shuffled = deck.copy()
        self._rng.shuffle(shuffled)
        return shuffled

    def deal_cards(self, player_num: int) -> tuple[list[list[Card]], list[Card]]:
        """Deal cards to players.

        Args:
            player_num: Number of players (2-6)

        Returns:
            Tuple of (player_hands, remaining_deck)
        """
        deck = self.create_deck()
        shuffled = self.shuffle_deck(deck)

        hand_size = GameConsts.INITIAL_HAND_SIZE[player_num]
        player_hands: list[list[Card]] = []

        idx = 0
        for _ in range(player_num):
            hand = shuffled[idx : idx + hand_size]
            player_hands.append(hand)
            idx += hand_size

        remaining = shuffled[idx:]
        return player_hands, remaining
