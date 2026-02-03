"""Game state for Doudizhu (斗地主) game."""

from __future__ import annotations

import numpy as np

from games.doudizhu.card import Card, create_full_deck
from games.doudizhu.constants import GameConsts
from games.doudizhu.enums import GamePhase, PlayerRole
from games.doudizhu.hand_detector import HandDetector, HandInfo
from games.doudizhu.player import Player


class GameState:
    """Manages the state of a Doudizhu game."""

    def __init__(self) -> None:
        self._players: list[Player] = [Player(i) for i in range(GameConsts.NUM_PLAYERS)]
        self._current_player_idx: int = 0
        self._phase: GamePhase = GamePhase.BIDDING
        self._landlord_idx: int = -1
        self._bottom_cards: list[Card] = []
        self._last_play: list[Card] = []
        self._last_play_info: HandInfo | None = None
        self._last_player_idx: int = -1
        self._pass_count: int = 0
        self._bid_history: list[int] = []  # 0=pass, 1=bid
        self._current_bidder_idx: int = 0

    @property
    def current_player_idx(self) -> int:
        return self._current_player_idx

    @property
    def phase(self) -> GamePhase:
        return self._phase

    @property
    def landlord_idx(self) -> int:
        return self._landlord_idx

    @property
    def bottom_cards(self) -> list[Card]:
        return self._bottom_cards

    @property
    def last_play(self) -> list[Card]:
        return self._last_play

    @property
    def last_play_info(self) -> HandInfo | None:
        return self._last_play_info

    @property
    def last_player_idx(self) -> int:
        return self._last_player_idx

    @property
    def is_terminated(self) -> bool:
        return self._phase == GamePhase.FINISHED

    def get_player(self, player_idx: int) -> Player:
        """Get player by index."""
        return self._players[player_idx]

    def reset(self, rng: np.random.Generator) -> None:
        """Reset game state and deal cards."""
        # Reset players
        for player in self._players:
            player.reset()

        # Shuffle and deal
        deck = create_full_deck()
        deck_indices = list(range(len(deck)))
        rng.shuffle(deck_indices)
        shuffled_deck = [deck[i] for i in deck_indices]

        # Deal 17 cards to each player
        for i in range(GameConsts.NUM_PLAYERS):
            start = i * GameConsts.PEASANT_CARDS
            end = start + GameConsts.PEASANT_CARDS
            self._players[i].set_hand(shuffled_deck[start:end])

        # Set bottom cards (last 3)
        self._bottom_cards = shuffled_deck[-GameConsts.BOTTOM_CARDS :]

        # Reset game state
        self._current_player_idx = 0
        self._phase = GamePhase.BIDDING
        self._landlord_idx = -1
        self._last_play = []
        self._last_play_info = None
        self._last_player_idx = -1
        self._pass_count = 0
        self._bid_history = []
        self._current_bidder_idx = 0

    def bid(self, want_landlord: bool) -> None:
        """Process a bid action during bidding phase."""
        if self._phase != GamePhase.BIDDING:
            return

        self._bid_history.append(1 if want_landlord else 0)

        if want_landlord:
            # Player becomes landlord
            self._landlord_idx = self._current_player_idx
            self._players[self._landlord_idx].set_role(PlayerRole.LANDLORD)
            self._players[self._landlord_idx].add_cards(self._bottom_cards)
            self._phase = GamePhase.PLAYING
            # Landlord plays first
            self._current_player_idx = self._landlord_idx
        else:
            # Move to next bidder
            self._current_bidder_idx += 1
            if self._current_bidder_idx >= GameConsts.NUM_PLAYERS:
                # No one wants to be landlord, force first player
                self._landlord_idx = 0
                self._players[0].set_role(PlayerRole.LANDLORD)
                self._players[0].add_cards(self._bottom_cards)
                self._phase = GamePhase.PLAYING
                self._current_player_idx = 0
            else:
                self._current_player_idx = self._current_bidder_idx

    def play(self, cards: list[Card]) -> bool:
        """Play cards. Returns True if valid play."""
        if self._phase != GamePhase.PLAYING:
            return False

        player = self._players[self._current_player_idx]

        # Pass (不出)
        if not cards:
            if self._last_player_idx == self._current_player_idx:
                # Can't pass if you played last
                return False
            self._pass_count += 1
            if self._pass_count >= 2:
                # Two passes, reset the round
                self._last_play = []
                self._last_play_info = None
                self._pass_count = 0
            self._advance_player()
            return True

        # Check if player has the cards
        if not player.has_cards(cards):
            return False

        # Detect hand type
        hand_info = HandDetector.detect(cards)
        if hand_info.hand_type.value >= 15:  # INVALID
            return False

        # Check if can beat last play
        if self._last_play_info is not None and not hand_info.can_beat(
            self._last_play_info
        ):
            return False

        # Valid play
        player.play_cards(cards)
        self._last_play = cards
        self._last_play_info = hand_info
        self._last_player_idx = self._current_player_idx
        self._pass_count = 0

        # Check win condition
        if player.hand_count == 0:
            self._phase = GamePhase.FINISHED
            return True

        self._advance_player()
        return True

    def _advance_player(self) -> None:
        """Move to next player."""
        self._current_player_idx = (
            self._current_player_idx + 1
        ) % GameConsts.NUM_PLAYERS

    def get_winner(self) -> int | None:
        """Get winner index, or None if game not finished."""
        if self._phase != GamePhase.FINISHED:
            return None
        for player in self._players:
            if player.hand_count == 0:
                return player.player_idx
        return None

    def get_winner_team(self) -> PlayerRole | None:
        """Get winning team (LANDLORD or PEASANT)."""
        winner = self.get_winner()
        if winner is None:
            return None
        return self._players[winner].role
