"""Game state for Guandan (掼蛋) game."""

from __future__ import annotations

import numpy as np

from games.guandan.card import Card, create_double_deck
from games.guandan.constants import GameConsts
from games.guandan.enums import CardRank, GamePhase, Team
from games.guandan.hand_detector import HandDetector, HandInfo
from games.guandan.player import Player


class GameState:
    """Manages the state of a Guandan game."""

    def __init__(self) -> None:
        self._players: list[Player] = [Player(i) for i in range(GameConsts.NUM_PLAYERS)]
        self._current_player_idx: int = 0
        self._phase: GamePhase = GamePhase.PLAYING
        self._level_rank: CardRank = CardRank.TWO  # 当前级牌
        self._last_play: list[Card] = []
        self._last_play_info: HandInfo | None = None
        self._last_player_idx: int = -1
        self._pass_count: int = 0
        self._finish_order: int = 0  # Next finish order to assign
        self._team_scores: dict[Team, int] = {Team.TEAM_A: 0, Team.TEAM_B: 0}

    @property
    def current_player_idx(self) -> int:
        return self._current_player_idx

    @property
    def phase(self) -> GamePhase:
        return self._phase

    @property
    def level_rank(self) -> CardRank:
        return self._level_rank

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
        deck = create_double_deck()
        deck_indices = list(range(len(deck)))
        rng.shuffle(deck_indices)
        shuffled_deck = [deck[i] for i in deck_indices]

        # Deal 27 cards to each player
        for i in range(GameConsts.NUM_PLAYERS):
            start = i * GameConsts.CARDS_PER_PLAYER
            end = start + GameConsts.CARDS_PER_PLAYER
            self._players[i].set_hand(shuffled_deck[start:end])

        # Reset game state
        self._current_player_idx = 0
        self._phase = GamePhase.PLAYING
        self._last_play = []
        self._last_play_info = None
        self._last_player_idx = -1
        self._pass_count = 0
        self._finish_order = 0
        self._team_scores = {Team.TEAM_A: 0, Team.TEAM_B: 0}

    def play(self, cards: list[Card]) -> bool:
        """Play cards. Returns True if valid play."""
        if self._phase != GamePhase.PLAYING:
            return False

        player = self._players[self._current_player_idx]

        # Skip finished players
        if player.finished:
            self._advance_player()
            return True

        # Pass (不出)
        if not cards:
            if self._last_player_idx == self._current_player_idx:
                # Can't pass if you played last (new round)
                return False
            self._pass_count += 1
            # Check if all other active players passed
            active_players = sum(1 for p in self._players if not p.finished)
            if self._pass_count >= active_players - 1:
                # Reset round
                self._last_play = []
                self._last_play_info = None
                self._pass_count = 0
            self._advance_player()
            return True

        # Check if player has the cards
        if not player.has_cards(cards):
            return False

        # Detect hand type
        hand_info = HandDetector.detect(cards, self._level_rank)
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

        # Check if player finished
        if player.hand_count == 0:
            self._finish_order += 1
            player.mark_finished(self._finish_order)
            self._check_game_end()

        if not self.is_terminated:
            self._advance_player()

        return True

    def _advance_player(self) -> None:
        """Move to next active player."""
        for _ in range(GameConsts.NUM_PLAYERS):
            self._current_player_idx = (
                self._current_player_idx + 1
            ) % GameConsts.NUM_PLAYERS
            if not self._players[self._current_player_idx].finished:
                break

    def _check_game_end(self) -> None:
        """Check if game should end."""
        finished_count = sum(1 for p in self._players if p.finished)

        # Game ends when 3 players finish (one left)
        if finished_count >= 3:
            # Mark last player as 4th
            for p in self._players:
                if not p.finished:
                    p.mark_finished(4)
            self._phase = GamePhase.FINISHED
            self._calculate_scores()

    def _calculate_scores(self) -> None:
        """Calculate team scores based on finish order."""
        # Get finish orders by team
        team_a_orders = [p.finish_order for p in self._players if p.team == Team.TEAM_A]
        team_b_orders = [p.finish_order for p in self._players if p.team == Team.TEAM_B]

        # Scoring rules:
        # - 双上 (both teammates 1st and 2nd): +3
        # - 头游+三游 (1st and 3rd): +2
        # - 头游+末游 (1st and 4th): +1
        # - 二游+三游 (2nd and 3rd): +1

        def get_team_score(orders: list[int]) -> int:
            orders = sorted(orders)
            if orders == [1, 2]:
                return 3
            if orders == [1, 3]:
                return 2
            if orders == [1, 4]:
                return 1
            if orders == [2, 3]:
                return 1
            return 0

        self._team_scores[Team.TEAM_A] = get_team_score(team_a_orders)
        self._team_scores[Team.TEAM_B] = get_team_score(team_b_orders)

    def get_winner_team(self) -> Team | None:
        """Get winning team."""
        if self._phase != GamePhase.FINISHED:
            return None
        if self._team_scores[Team.TEAM_A] > self._team_scores[Team.TEAM_B]:
            return Team.TEAM_A
        if self._team_scores[Team.TEAM_B] > self._team_scores[Team.TEAM_A]:
            return Team.TEAM_B
        return None  # Tie

    def get_team_score(self, team: Team) -> int:
        """Get score for a team."""
        return self._team_scores[team]
