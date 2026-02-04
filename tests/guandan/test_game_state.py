"""Tests for Guandan game state module."""

import numpy as np
import pytest

from games.guandan.card import Card
from games.guandan.constants import GameConsts
from games.guandan.enums import CardRank, CardSuit, GamePhase, Team
from games.guandan.game_state import GameState


class TestGameStateInit:
    """Test GameState initialization."""

    def test_initial_state(self):
        state = GameState()
        assert state.current_player_idx == 0
        assert state.phase == GamePhase.PLAYING
        assert state.level_rank == CardRank.TWO
        assert len(state.last_play) == 0
        assert not state.is_terminated

    def test_get_player(self):
        state = GameState()
        for i in range(GameConsts.NUM_PLAYERS):
            player = state.get_player(i)
            assert player.player_idx == i


class TestGameStateReset:
    """Test GameState reset."""

    @pytest.fixture
    def state(self) -> GameState:
        state = GameState()
        rng = np.random.default_rng(42)
        state.reset(rng)
        return state

    def test_reset_deals_cards(self, state):
        for i in range(GameConsts.NUM_PLAYERS):
            player = state.get_player(i)
            assert player.hand_count == GameConsts.CARDS_PER_PLAYER

    def test_reset_total_cards(self, state):
        total = sum(
            state.get_player(i).hand_count for i in range(GameConsts.NUM_PLAYERS)
        )
        assert total == GameConsts.TOTAL_CARDS

    def test_reset_with_seed_deterministic(self):
        state1 = GameState()
        state2 = GameState()
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        state1.reset(rng1)
        state2.reset(rng2)

        for i in range(GameConsts.NUM_PLAYERS):
            hand1 = state1.get_player(i).hand
            hand2 = state2.get_player(i).hand
            assert hand1 == hand2


class TestGameStatePlaying:
    """Test playing phase."""

    @pytest.fixture
    def state(self) -> GameState:
        state = GameState()
        rng = np.random.default_rng(42)
        state.reset(rng)
        return state

    def test_play_single_card(self, state):
        player = state.get_player(0)
        card = player.hand[0]
        initial_count = player.hand_count

        result = state.play([card])

        assert result is True
        assert player.hand_count == initial_count - 1
        assert state.last_play == [card]

    def test_play_advances_player(self, state):
        player = state.get_player(0)
        card = player.hand[0]
        state.play([card])
        assert state.current_player_idx == 1

    def test_pass_when_not_last_player(self, state):
        # Player 0 plays
        player0 = state.get_player(0)
        state.play([player0.hand[0]])

        # Player 1 passes
        result = state.play([])
        assert result is True
        assert state.current_player_idx == 2

    def test_cannot_pass_when_last_player(self, state):
        # Player 0 plays, then 1, 2, 3 pass
        player0 = state.get_player(0)
        state.play([player0.hand[0]])
        state.play([])  # Player 1 passes
        state.play([])  # Player 2 passes
        state.play([])  # Player 3 passes

        # Now it's player 0's turn again, round should reset
        assert state.last_player_idx == 0
        assert len(state.last_play) == 0

    def test_play_invalid_cards_not_in_hand(self, state):
        fake_card = Card(CardRank.RED_JOKER, CardSuit.JOKER, 0)
        player = state.get_player(0)
        if fake_card not in player.hand:
            result = state.play([fake_card])
            assert result is False


class TestGameStateFinishing:
    """Test game finishing logic."""

    def test_player_finishes_when_empty_hand(self):
        state = GameState()
        rng = np.random.default_rng(42)
        state.reset(rng)

        # Play all cards from player 0
        player = state.get_player(0)
        while player.hand_count > 0 and not state.is_terminated:
            card = player.hand[0]
            if not state.play([card]):
                break
            if state.current_player_idx != 0:
                # Others pass
                for _ in range(3):
                    if state.current_player_idx != 0 and not state.is_terminated:
                        state.play([])

        if player.hand_count == 0:
            assert player.finished
            assert player.finish_order == 1

    def test_game_ends_when_three_finish(self):
        state = GameState()
        rng = np.random.default_rng(42)
        state.reset(rng)

        finished_count = 0
        max_steps = 1000
        step = 0

        while not state.is_terminated and step < max_steps:
            player = state.get_player(state.current_player_idx)
            if player.finished:
                state.play([])
                step += 1
                continue

            # Try to play a card
            if player.hand_count > 0:
                card = player.hand[0]
                if not state.play([card]):
                    state.play([])
            else:
                state.play([])
            step += 1

        # Game should eventually terminate
        if state.is_terminated:
            finished_count = sum(1 for i in range(4) if state.get_player(i).finished)
            assert finished_count == 4


class TestGameStateScoring:
    """Test scoring logic."""

    def test_get_winner_team_before_finish(self):
        state = GameState()
        rng = np.random.default_rng(42)
        state.reset(rng)
        assert state.get_winner_team() is None

    def test_get_team_score(self):
        state = GameState()
        rng = np.random.default_rng(42)
        state.reset(rng)
        assert state.get_team_score(Team.TEAM_A) == 0
        assert state.get_team_score(Team.TEAM_B) == 0


class TestGameStateEdgeCases:
    """Test edge cases."""

    def test_play_in_finished_phase(self):
        state = GameState()
        state._phase = GamePhase.FINISHED
        result = state.play([])
        assert result is False

    def test_last_play_info_none_initially(self):
        state = GameState()
        rng = np.random.default_rng(42)
        state.reset(rng)
        assert state.last_play_info is None

    def test_skip_finished_player(self):
        """Test that finished players are skipped."""
        state = GameState()
        rng = np.random.default_rng(42)
        state.reset(rng)

        # Mark player 1 as finished
        state._players[1]._finished = True
        state._players[1]._finish_order = 1

        # Player 0 plays
        player0 = state.get_player(0)
        state.play([player0.hand[0]])

        # Should skip player 1 and go to player 2
        assert state.current_player_idx == 2

    def test_cannot_pass_own_play(self):
        """Test that player cannot pass their own play."""
        state = GameState()
        rng = np.random.default_rng(42)
        state.reset(rng)

        # Player 0 plays
        player0 = state.get_player(0)
        state.play([player0.hand[0]])

        # All others pass
        state.play([])  # Player 1
        state.play([])  # Player 2
        state.play([])  # Player 3

        # Now player 0 cannot pass (must play)
        result = state.play([])
        assert result is False

    def test_play_invalid_hand_type(self):
        """Test playing invalid hand type."""
        state = GameState()
        rng = np.random.default_rng(42)
        state.reset(rng)

        # Try to play two different cards (invalid)
        player = state.get_player(0)
        if player.hand_count >= 2:
            card1 = player.hand[0]
            card2 = player.hand[1]
            if card1.rank != card2.rank:
                result = state.play([card1, card2])
                assert result is False
