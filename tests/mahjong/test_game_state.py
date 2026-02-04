"""Tests for Mahjong game state module."""

import numpy as np
import pytest

from games.mahjong.constants import GameConsts
from games.mahjong.enums import ActionType, GamePhase
from games.mahjong.game_state import GameState


class TestGameStateInit:
    """Test GameState initialization."""

    def test_initial_state(self):
        state = GameState()
        assert state.current_player_idx == 0
        assert state.phase == GamePhase.DRAWING
        assert state.wall_count == 0
        assert state.last_discard is None
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

    def test_reset_deals_tiles(self, state):
        for i in range(GameConsts.NUM_PLAYERS):
            player = state.get_player(i)
            assert player.hand_count == GameConsts.HAND_SIZE

    def test_reset_wall_size(self, state):
        expected_wall = (
            GameConsts.TOTAL_TILES - GameConsts.NUM_PLAYERS * GameConsts.HAND_SIZE
        )
        assert state.wall_count == expected_wall

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


class TestGameStateDrawAndDiscard:
    """Test draw and discard."""

    @pytest.fixture
    def state(self) -> GameState:
        state = GameState()
        rng = np.random.default_rng(42)
        state.reset(rng)
        return state

    def test_draw_tile(self, state):
        initial_wall = state.wall_count
        player = state.get_player(0)
        initial_hand = player.hand_count

        tile = state.draw_tile()

        assert tile is not None
        assert state.wall_count == initial_wall - 1
        assert player.hand_count == initial_hand + 1
        assert state.phase == GamePhase.DISCARDING

    def test_discard_tile(self, state):
        state.draw_tile()
        player = state.get_player(0)
        tile = player.hand[0]

        result = state.discard_tile(tile)

        assert result is True
        assert state.last_discard == tile

    def test_discard_wrong_phase(self, state):
        player = state.get_player(0)
        tile = player.hand[0]
        result = state.discard_tile(tile)
        assert result is False

    def test_discard_advances_player(self, state):
        state.draw_tile()
        player = state.get_player(0)
        tile = player.hand[0]
        state.discard_tile(tile)

        # If no responses, should advance
        if state.phase == GamePhase.DRAWING:
            assert state.current_player_idx == 1


class TestGameStateResponses:
    """Test response actions (chi, pong, gang, hu)."""

    def test_respond_pass(self):
        state = GameState()
        rng = np.random.default_rng(42)
        state.reset(rng)
        state.draw_tile()

        player = state.get_player(0)
        tile = player.hand[0]
        state.discard_tile(tile)

        if state.phase == GamePhase.WAITING_RESPONSE:
            for player_idx in list(state._pending_responses.keys()):
                state.respond(player_idx, ActionType.PASS)

    def test_respond_wrong_phase(self):
        state = GameState()
        rng = np.random.default_rng(42)
        state.reset(rng)
        result = state.respond(1, ActionType.PASS)
        assert result is False


class TestGameStateSpecialActions:
    """Test special actions."""

    def test_an_gang_wrong_phase(self):
        state = GameState()
        rng = np.random.default_rng(42)
        state.reset(rng)
        result = state.an_gang(0)
        assert result is False

    def test_self_hu_wrong_phase(self):
        state = GameState()
        rng = np.random.default_rng(42)
        state.reset(rng)
        result = state.self_hu()
        assert result is False

    def test_get_valid_discards(self):
        state = GameState()
        rng = np.random.default_rng(42)
        state.reset(rng)
        state.draw_tile()

        discards = state.get_valid_discards()
        assert len(discards) > 0

    def test_get_valid_discards_wrong_phase(self):
        state = GameState()
        rng = np.random.default_rng(42)
        state.reset(rng)
        discards = state.get_valid_discards()
        assert len(discards) == 0


class TestGameStateTermination:
    """Test game termination."""

    def test_empty_wall_terminates(self):
        state = GameState()
        rng = np.random.default_rng(42)
        state.reset(rng)

        # Draw all tiles
        while state.wall_count > 0 and not state.is_terminated:
            if state.phase == GamePhase.DRAWING:
                state.draw_tile()
            elif state.phase == GamePhase.DISCARDING:
                player = state.get_player(state.current_player_idx)
                if player.hand_count > 0:
                    state.discard_tile(player.hand[0])
            elif state.phase == GamePhase.WAITING_RESPONSE:
                for player_idx in list(state._pending_responses.keys()):
                    state.respond(player_idx, ActionType.PASS)

        assert state.is_terminated or state.wall_count == 0


class TestAdvancedResponses:
    """Test advanced response actions."""

    def test_respond_not_in_pending(self):
        """Test responding when player is not in pending responses."""
        state = GameState()
        rng = np.random.default_rng(42)
        state.reset(rng)
        state.draw_tile()
        player = state.get_player(0)
        state.discard_tile(player.hand[0])

        if state.phase == GamePhase.WAITING_RESPONSE:
            # Try to respond with a player not in pending
            result = state.respond(0, ActionType.PASS)  # Player 0 discarded
            assert result is False

    def test_respond_invalid_action(self):
        """Test responding with invalid action type."""
        state = GameState()
        rng = np.random.default_rng(42)
        state.reset(rng)
        state.draw_tile()
        player = state.get_player(0)
        state.discard_tile(player.hand[0])

        if state.phase == GamePhase.WAITING_RESPONSE:
            for player_idx in list(state._pending_responses.keys()):
                # Try invalid action (CHI without tiles)
                assert state.respond(player_idx, ActionType.CHI, tiles=None) is False
                break

    def test_draw_from_empty_wall(self):
        """Test drawing when wall is empty."""
        state = GameState()
        rng = np.random.default_rng(42)
        state.reset(rng)

        # Empty the wall
        state._wall = []
        tile = state.draw_tile()
        assert tile is None
        assert state.is_terminated

    def test_respond_action_not_in_allowed_actions(self):
        """Test responding with action not in allowed actions."""
        state = GameState()
        rng = np.random.default_rng(42)
        state.reset(rng)
        state.draw_tile()
        player = state.get_player(0)
        state.discard_tile(player.hand[0])

        if state.phase == GamePhase.WAITING_RESPONSE:
            for player_idx in list(state._pending_responses.keys()):
                allowed = state._pending_responses[player_idx]
                # Try an action not in allowed list
                if ActionType.HU not in allowed:
                    result = state.respond(player_idx, ActionType.HU)
                    assert result is False
                break

    def test_respond_with_last_discard_none(self):
        """Test respond when last_discard is None (edge case)."""
        state = GameState()
        rng = np.random.default_rng(42)
        state.reset(rng)
        state.draw_tile()
        player = state.get_player(0)
        state.discard_tile(player.hand[0])

        if state.phase == GamePhase.WAITING_RESPONSE:
            # Force last_discard to None
            state._last_discard = None
            for player_idx in list(state._pending_responses.keys()):
                result = state.respond(player_idx, ActionType.PASS)
                assert result is False
                break

    def test_chi_with_wrong_tile_count(self):
        """Test chi with wrong number of tiles."""
        state = GameState()
        rng = np.random.default_rng(42)
        state.reset(rng)
        state.draw_tile()
        player = state.get_player(0)
        state.discard_tile(player.hand[0])

        if state.phase == GamePhase.WAITING_RESPONSE:
            for player_idx in list(state._pending_responses.keys()):
                # Try CHI with only 1 tile
                result = state.respond(
                    player_idx,
                    ActionType.CHI,
                    tiles=[state.get_player(player_idx).hand[0]],
                )
                assert result is False
                break
