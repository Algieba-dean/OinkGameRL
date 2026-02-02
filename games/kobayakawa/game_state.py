"""GameState module for Kobayakawa game."""

from __future__ import annotations

from games.kobayakawa.card import Card
from games.kobayakawa.constants import GameConsts, PlayerConsts
from games.kobayakawa.dealer import Dealer
from games.kobayakawa.enums import GamePhase
from games.kobayakawa.player import Player


class GameState:
    """Manages the complete state of a Kobayakawa game."""

    def __init__(self, player_num: int) -> None:
        if player_num not in PlayerConsts.ALLOWED_PLAYER_NUM:
            raise ValueError(
                f"Invalid player num {player_num}, "
                f"allowed: {PlayerConsts.ALLOWED_PLAYER_NUM}"
            )

        self.__player_num = player_num
        self.__players = [Player(player_idx=i) for i in range(player_num)]
        self.__kobayakawa_card: Card | None = None
        self.__current_player_idx = 0
        self.__phase = GamePhase.DRAW_OR_SWAP
        self.__round = 1
        self.__pot = 0
        self.__dealer = Dealer()

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
    def kobayakawa_card(self) -> Card | None:
        return self.__kobayakawa_card

    @property
    def phase(self) -> GamePhase:
        return self.__phase

    @property
    def round(self) -> int:
        return self.__round

    @property
    def pot(self) -> int:
        return self.__pot

    @property
    def is_terminated(self) -> bool:
        """Game ends after 7 rounds or only 1 player has coins."""
        active_players = sum(1 for p in self.__players if not p.is_eliminated)
        return self.__round > GameConsts.ROUNDS_PER_GAME or active_players <= 1

    @property
    def dealer(self) -> Dealer:
        return self.__dealer

    def get_player(self, player_idx: int) -> Player:
        return self.__players[player_idx]

    def get_current_player(self) -> Player:
        return self.__players[self.__current_player_idx]

    def set_kobayakawa(self, card: Card) -> None:
        """Set the Kobayakawa card."""
        self.__kobayakawa_card = card

    def add_to_pot(self, amount: int) -> None:
        """Add coins to pot."""
        self.__pot += amount

    def next_player(self) -> None:
        """Advance to next non-eliminated player."""
        for _ in range(self.__player_num):
            self.__current_player_idx = (
                self.__current_player_idx + 1
            ) % self.__player_num
            if not self.__players[self.__current_player_idx].is_eliminated:
                break

    def set_phase(self, phase: GamePhase) -> None:
        """Set game phase."""
        self.__phase = phase

    def start_new_round(self, rng) -> None:
        """Start a new round."""
        self.__round += 1
        self.__pot = 0
        self.__current_player_idx = 0

        while self.__players[self.__current_player_idx].is_eliminated:
            self.__current_player_idx = (
                self.__current_player_idx + 1
            ) % self.__player_num

        for player in self.__players:
            player.reset_bet()

        self.__dealer.reset(random_generator=rng)
        self.__dealer.create_and_shuffle_deck()

        for player in self.__players:
            if not player.is_eliminated:
                card = self.__dealer.deal_one()
                if card:
                    player.set_card(card)

        kobayakawa = self.__dealer.deal_one()
        if kobayakawa:
            self.__kobayakawa_card = kobayakawa

        self.__phase = GamePhase.DRAW_OR_SWAP

    def resolve_showdown(self) -> int | None:
        """Resolve showdown and return winner index."""
        betting_players = [p for p in self.__players if p.has_bet and p.card]

        if not betting_players:
            return None

        if len(betting_players) == 1:
            winner = betting_players[0]
            winner.win_pot(self.__pot)
            return winner.player_idx

        min_card_player = min(
            betting_players, key=lambda p: p.card.value if p.card else 0
        )

        effective_values: dict[int, int] = {}
        for p in betting_players:
            assert p.card is not None  # guaranteed by betting_players filter
            val = p.card.value
            if p == min_card_player and self.__kobayakawa_card:
                val += self.__kobayakawa_card.value
            effective_values[p.player_idx] = val

        winner_idx = max(effective_values, key=lambda k: effective_values[k])
        self.__players[winner_idx].win_pot(self.__pot)
        return winner_idx

    def get_winner(self) -> int | None:
        """Get overall game winner."""
        if not self.is_terminated:
            return None

        active_players = [p for p in self.__players if not p.is_eliminated]
        if len(active_players) == 1:
            return active_players[0].player_idx

        max_coins = max(p.coins for p in self.__players)
        winners = [p for p in self.__players if p.coins == max_coins]
        if len(winners) == 1:
            return winners[0].player_idx
        return None

    def reset(self, rng) -> None:
        """Reset game state."""
        for player in self.__players:
            player.reset()
        self.__kobayakawa_card = None
        self.__current_player_idx = 0
        self.__phase = GamePhase.DRAW_OR_SWAP
        self.__round = 1
        self.__pot = 0
        self.start_new_round(rng)
