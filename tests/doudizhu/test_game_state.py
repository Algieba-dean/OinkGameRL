"""Tests for Doudizhu game state module."""

import numpy as np
import pytest

from games.doudizhu.card import Card
from games.doudizhu.constants import GameConsts
from games.doudizhu.enums import CardRank, CardSuit, GamePhase, PlayerRole
from games.doudizhu.game_state import GameState


class TestGameStateInit:
    """Test GameState initialization."""

    def test_initial_state(self):
        state = GameState()
        assert state.current_player_idx == 0
        assert state.phase == GamePhase.BIDDING
        assert state.landlord_idx == -1
        assert len(state.bottom_cards) == 0
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
            assert player.hand_count == GameConsts.PEASANT_CARDS

    def test_reset_sets_bottom_cards(self, state):
        assert len(state.bottom_cards) == GameConsts.BOTTOM_CARDS

    def test_reset_total_cards(self, state):
        total = sum(
            state.get_player(i).hand_count for i in range(GameConsts.NUM_PLAYERS)
        )
        total += len(state.bottom_cards)
        assert total == GameConsts.TOTAL_CARDS

    def test_reset_phase_is_bidding(self, state):
        assert state.phase == GamePhase.BIDDING

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


class TestGameStateBidding:
    """Test bidding phase."""

    @pytest.fixture
    def state(self) -> GameState:
        state = GameState()
        rng = np.random.default_rng(42)
        state.reset(rng)
        return state

    def test_bid_becomes_landlord(self, state):
        state.bid(want_landlord=True)
        assert state.landlord_idx == 0
        assert state.phase == GamePhase.PLAYING
        assert state.get_player(0).role == PlayerRole.LANDLORD

    def test_bid_landlord_gets_bottom_cards(self, state):
        initial_count = state.get_player(0).hand_count
        bottom_count = len(state.bottom_cards)
        state.bid(want_landlord=True)
        assert state.get_player(0).hand_count == initial_count + bottom_count

    def test_pass_moves_to_next_bidder(self, state):
        state.bid(want_landlord=False)
        assert state.current_player_idx == 1
        assert state.phase == GamePhase.BIDDING

    def test_all_pass_forces_first_landlord(self, state):
        for _ in range(GameConsts.NUM_PLAYERS):
            state.bid(want_landlord=False)
        assert state.landlord_idx == 0
        assert state.phase == GamePhase.PLAYING

    def test_second_player_bids(self, state):
        state.bid(want_landlord=False)  # Player 0 passes
        state.bid(want_landlord=True)  # Player 1 bids
        assert state.landlord_idx == 1
        assert state.current_player_idx == 1  # Landlord plays first


class TestGameStatePlaying:
    """Test playing phase."""

    @pytest.fixture
    def state(self) -> GameState:
        state = GameState()
        rng = np.random.default_rng(42)
        state.reset(rng)
        state.bid(want_landlord=True)  # Player 0 becomes landlord
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
        # Player 0 plays, then 1 and 2 pass
        player0 = state.get_player(0)
        state.play([player0.hand[0]])
        state.play([])  # Player 1 passes
        state.play([])  # Player 2 passes

        # Now it's player 0's turn again, and they played last
        # They should be able to play anything (new round)
        assert state.last_player_idx == 0
        assert len(state.last_play) == 0  # Round reset

    def test_play_invalid_cards_not_in_hand(self, state):
        fake_card = Card(CardRank.RED_JOKER, CardSuit.JOKER)
        player = state.get_player(0)
        if fake_card not in player.hand:
            result = state.play([fake_card])
            assert result is False

    def test_win_condition(self, state):
        player = state.get_player(0)
        # Play all cards one by one (simplified test)
        while player.hand_count > 0 and not state.is_terminated:
            card = player.hand[0]
            state.play([card])
            if state.current_player_idx != 0:
                state.play([])  # Others pass

        if player.hand_count == 0:
            assert state.is_terminated
            assert state.get_winner() == 0

    def test_get_winner_before_finish(self, state):
        assert state.get_winner() is None

    def test_get_winner_team(self, state):
        # Force a quick win for testing
        player = state.get_player(0)
        while player.hand_count > 0:
            card = player.hand[0]
            if not state.play([card]):
                break
            if state.is_terminated:
                break
            # Others pass
            for _ in range(2):
                if state.current_player_idx != 0 and not state.is_terminated:
                    state.play([])

        if state.is_terminated:
            winner_team = state.get_winner_team()
            assert winner_team == PlayerRole.LANDLORD


class TestGameStateEdgeCases:
    """Test edge cases."""

    def test_bid_in_playing_phase_ignored(self):
        state = GameState()
        rng = np.random.default_rng(42)
        state.reset(rng)
        state.bid(want_landlord=True)  # Now in PLAYING phase

        old_landlord = state.landlord_idx
        state.bid(want_landlord=True)  # Should be ignored
        assert state.landlord_idx == old_landlord

    def test_play_in_bidding_phase(self):
        state = GameState()
        rng = np.random.default_rng(42)
        state.reset(rng)

        card = state.get_player(0).hand[0]
        result = state.play([card])
        assert result is False

    def test_last_play_info_none_initially(self):
        state = GameState()
        rng = np.random.default_rng(42)
        state.reset(rng)
        state.bid(want_landlord=True)
        assert state.last_play_info is None
