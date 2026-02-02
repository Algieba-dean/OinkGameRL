"""GameState module for Maskmen game."""

from __future__ import annotations

from games.maskmen.card import Card
from games.maskmen.constants import GameConsts, PlayerConsts
from games.maskmen.enums import CardColor
from games.maskmen.player import Player


class GameState:
    """Manages the complete state of a Maskmen game."""

    def __init__(
        self,
        player_num: int,
        player_hands: list[list[Card]],
        deck: list[Card],
    ) -> None:
        if player_num not in PlayerConsts.ALLOWED_PLAYER_NUM:
            raise ValueError(
                f"Invalid player num {player_num}, "
                f"allowed: {PlayerConsts.ALLOWED_PLAYER_NUM}"
            )
        if len(player_hands) != player_num:
            raise ValueError(f"Expected {player_num} hands, got {len(player_hands)}")

        self.__player_num = player_num
        self.__players = [
            Player(player_idx=i, cards=player_hands[i]) for i in range(player_num)
        ]
        self.__deck = list(deck)
        self.__discard_pile: list[Card] = []
        self.__current_player_idx = 0
        self.__table: dict[CardColor, list[Card]] = {color: [] for color in CardColor}

    @property
    def player_num(self) -> int:
        return self.__player_num

    @property
    def players(self) -> tuple[Player, ...]:
        return tuple(self.__players)

    @property
    def current_player_idx(self) -> int:
        return self.__current_player_idx

    @property
    def deck_count(self) -> int:
        return len(self.__deck)

    @property
    def table(self) -> dict[CardColor, tuple[Card, ...]]:
        return {color: tuple(cards) for color, cards in self.__table.items()}

    @property
    def is_terminated(self) -> bool:
        """Game ends when a player collects 3 sets or deck is empty."""
        for player in self.__players:
            if player.set_count >= GameConsts.WINNING_SETS:
                return True
        return self.__deck_count_zero_and_all_passed()

    def __deck_count_zero_and_all_passed(self) -> bool:
        """Check if deck is empty and no one can play."""
        if len(self.__deck) > 0:
            return False
        return all(player.hand_count == 0 for player in self.__players)

    def get_player(self, player_idx: int) -> Player:
        return self.__players[player_idx]

    def get_current_player(self) -> Player:
        return self.__players[self.__current_player_idx]

    def next_player(self) -> None:
        self.__current_player_idx = (self.__current_player_idx + 1) % self.__player_num

    def play_card_to_table(self, card: Card) -> None:
        """Play a card to the table."""
        self.__table[card.color].append(card)
        self.__check_and_collect_set(card.color)

    def __check_and_collect_set(self, color: CardColor) -> None:
        """Check if a color set is complete (3 cards) and award it."""
        table_cards = self.__table[color]
        if len(table_cards) >= 3:
            values = {c.value for c in table_cards}
            if values == {1, 2, 3}:
                self.get_current_player().collect_set(color)
                self.__discard_pile.extend(table_cards)
                self.__table[color] = []

    def draw_card(self) -> Card | None:
        """Draw a card from deck."""
        if not self.__deck:
            return None
        return self.__deck.pop()

    def get_winner(self) -> int | None:
        """Get winner player index or None if no winner yet."""
        for player in self.__players:
            if player.set_count >= GameConsts.WINNING_SETS:
                return player.player_idx

        if self.is_terminated:
            max_sets = max(p.set_count for p in self.__players)
            winners = [p for p in self.__players if p.set_count == max_sets]
            if len(winners) == 1:
                return winners[0].player_idx
        return None

    def reset(self, player_hands: list[list[Card]], deck: list[Card]) -> None:
        """Reset game state."""
        for i, player in enumerate(self.__players):
            player.reset(player_hands[i])
        self.__deck = list(deck)
        self.__discard_pile = []
        self.__current_player_idx = 0
        self.__table = {color: [] for color in CardColor}
