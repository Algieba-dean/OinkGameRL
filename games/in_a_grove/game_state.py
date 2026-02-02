"""GameState module for In a Grove game."""

from __future__ import annotations

import numpy as np

from games.in_a_grove.card import SuspectCard
from games.in_a_grove.constants import GameConsts, PlayerConsts
from games.in_a_grove.enums import GamePhase, TileType
from games.in_a_grove.player import Player


class GameState:
    """Manages the complete state of an In a Grove game."""

    def __init__(self, player_num: int) -> None:
        if player_num not in PlayerConsts.ALLOWED_PLAYER_NUM:
            raise ValueError(
                f"Invalid player num {player_num}, "
                f"allowed: {PlayerConsts.ALLOWED_PLAYER_NUM}"
            )

        self.__player_num = player_num
        self.__players = [Player(player_idx=i) for i in range(player_num)]
        self.__center_card: SuspectCard | None = None
        self.__current_player_idx = 0
        self.__phase = GamePhase.DEALING
        self.__round = 1

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
    def center_card(self) -> SuspectCard | None:
        return self.__center_card

    @property
    def phase(self) -> GamePhase:
        return self.__phase

    @property
    def round(self) -> int:
        return self.__round

    @property
    def is_terminated(self) -> bool:
        """Game ends after 3 rounds."""
        return self.__round > GameConsts.ROUNDS_PER_GAME

    def get_player(self, player_idx: int) -> Player:
        return self.__players[player_idx]

    def get_current_player(self) -> Player:
        return self.__players[self.__current_player_idx]

    def next_player(self) -> None:
        self.__current_player_idx = (self.__current_player_idx + 1) % self.__player_num

    def set_phase(self, phase: GamePhase) -> None:
        """Set game phase."""
        self.__phase = phase

    def set_center_card(self, card: SuspectCard) -> None:
        """Set the center card (culprit indicator)."""
        self.__center_card = card

    def resolve_round(self) -> None:
        """Resolve the current round and calculate scores."""
        if self.__center_card is None:
            return

        culprit_value = self.__center_card.value
        accomplice_value = 1

        for player in self.__players:
            vote = player.current_vote
            if vote is None:
                continue

            if vote == TileType.CULPRIT:
                for card in player.hand:
                    if card.value == culprit_value:
                        player.add_score(culprit_value)
                        break

            elif vote == TileType.ACCOMPLICE:
                for card in player.hand:
                    if card.value == accomplice_value:
                        player.add_score(1)
                        break

            elif vote == TileType.WITNESS:
                has_neither = True
                for card in player.hand:
                    if card.value in (culprit_value, accomplice_value):
                        has_neither = False
                        break
                if has_neither:
                    player.add_score(1)

            player.clear_vote()

    def start_new_round(self, rng: np.random.Generator) -> None:
        """Start a new round."""
        self.__round += 1
        self.__current_player_idx = 0
        self.__center_card = None

        for player in self.__players:
            player.reset_tiles()

        cards = [SuspectCard(value=v) for v in range(1, 9)]
        rng.shuffle(cards)

        cards_per_player = 8 // self.__player_num
        idx = 0
        for player in self.__players:
            hand = cards[idx : idx + cards_per_player]
            player.set_hand(hand)
            idx += cards_per_player

        if idx < 8:
            self.__center_card = cards[idx]

        self.__phase = GamePhase.VOTING

    def get_winner(self) -> int | None:
        """Get winner (highest score)."""
        if not self.is_terminated:
            return None
        max_score = max(p.score for p in self.__players)
        winners = [p.player_idx for p in self.__players if p.score == max_score]
        return winners[0] if len(winners) == 1 else None

    def reset(self, rng: np.random.Generator) -> None:
        """Reset game state."""
        for player in self.__players:
            player.reset()
        self.__center_card = None
        self.__current_player_idx = 0
        self.__phase = GamePhase.DEALING
        self.__round = 0
        self.start_new_round(rng)
