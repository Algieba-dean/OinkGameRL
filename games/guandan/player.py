"""Player representation for Guandan (掼蛋) game."""

from __future__ import annotations

from games.guandan.card import Card
from games.guandan.enums import Team


class Player:
    """A player in Guandan game."""

    def __init__(self, player_idx: int) -> None:
        self._player_idx = player_idx
        self._hand: list[Card] = []
        self._team: Team = Team.TEAM_A if player_idx % 2 == 0 else Team.TEAM_B
        self._finished: bool = False
        self._finish_order: int = -1  # 1st, 2nd, 3rd, 4th

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
    def team(self) -> Team:
        return self._team

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def finish_order(self) -> int:
        return self._finish_order

    def set_hand(self, cards: list[Card]) -> None:
        """Set player's hand."""
        self._hand = sorted(cards)

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

    def mark_finished(self, order: int) -> None:
        """Mark player as finished with given order."""
        self._finished = True
        self._finish_order = order

    def reset(self) -> None:
        """Reset player state."""
        self._hand = []
        self._finished = False
        self._finish_order = -1
