"""Player representation for Doudizhu (斗地主) game."""

from __future__ import annotations

from games.doudizhu.card import Card
from games.doudizhu.enums import PlayerRole


class Player:
    """A player in Doudizhu game."""

    def __init__(self, player_idx: int) -> None:
        self._player_idx = player_idx
        self._hand: list[Card] = []
        self._role: PlayerRole = PlayerRole.PEASANT

    @property
    def player_idx(self) -> int:
        return self._player_idx

    @property
    def hand(self) -> list[Card]:
        return self._hand

    @property
    def hand_count(self) -> int:
        return len(self._hand)

    @property
    def role(self) -> PlayerRole:
        return self._role

    def set_role(self, role: PlayerRole) -> None:
        """Set player's role."""
        self._role = role

    def set_hand(self, cards: list[Card]) -> None:
        """Set player's hand."""
        self._hand = sorted(cards, key=lambda c: (c.rank, c.suit))

    def add_cards(self, cards: list[Card]) -> None:
        """Add cards to hand (for landlord getting bottom cards)."""
        self._hand.extend(cards)
        self._hand = sorted(self._hand, key=lambda c: (c.rank, c.suit))

    def play_cards(self, cards: list[Card]) -> list[Card]:
        """Remove and return the specified cards from hand."""
        played = []
        for card in cards:
            if card in self._hand:
                self._hand.remove(card)
                played.append(card)
        return played

    def has_cards(self, cards: list[Card]) -> bool:
        """Check if player has all the specified cards."""
        hand_copy = self._hand.copy()
        for card in cards:
            if card in hand_copy:
                hand_copy.remove(card)
            else:
                return False
        return True

    def reset(self) -> None:
        """Reset player state."""
        self._hand = []
        self._role = PlayerRole.PEASANT
