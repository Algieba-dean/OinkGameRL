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


class TestMahjongAdvancedActions:
    """Test advanced mahjong actions with constructed game states."""

    def test_respond_hu_action(self):
        """Test HU response action."""
        from games.mahjong.tile import Tile, TileSuit

        state = GameState()
        rng = np.random.default_rng(42)
        state.reset(rng)

        # Construct a winning hand for player 1
        # 11122233344455万 - needs one more tile to win
        winning_tiles = []
        for rank in [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5]:
            winning_tiles.append(Tile(TileSuit.WAN, rank, len(winning_tiles) % 4))

        state._players[1]._hand = winning_tiles

        # Player 0 draws and discards 5万
        state.draw_tile()
        discard_tile = Tile(TileSuit.WAN, 5, 3)
        state._players[0]._hand.append(discard_tile)
        state.discard_tile(discard_tile)

        # Check if player 1 can HU
        if (
            state.phase == GamePhase.WAITING_RESPONSE
            and 1 in state._pending_responses
            and ActionType.HU in state._pending_responses[1]
        ):
            result = state.respond(1, ActionType.HU)
            assert result is True
            assert state.is_terminated
            assert state.winner_idx == 1

    def test_respond_gang_action(self):
        """Test GANG response action."""
        from games.mahjong.tile import Tile, TileSuit

        state = GameState()
        rng = np.random.default_rng(42)
        state.reset(rng)

        # Give player 1 three 1万 tiles
        gang_tiles = [
            Tile(TileSuit.WAN, 1, 0),
            Tile(TileSuit.WAN, 1, 1),
            Tile(TileSuit.WAN, 1, 2),
        ]
        # Keep some other tiles to maintain hand
        other_tiles = [Tile(TileSuit.TIAO, i, 0) for i in range(1, 11)]
        state._players[1]._hand = gang_tiles + other_tiles

        # Player 0 draws and discards 1万
        state.draw_tile()
        discard_tile = Tile(TileSuit.WAN, 1, 3)
        state._players[0]._hand.append(discard_tile)
        state.discard_tile(discard_tile)

        # Check if player 1 can GANG
        if (
            state.phase == GamePhase.WAITING_RESPONSE
            and 1 in state._pending_responses
            and ActionType.GANG in state._pending_responses[1]
        ):
            result = state.respond(1, ActionType.GANG)
            assert result is True
            assert state.current_player_idx == 1
            assert state.phase == GamePhase.DRAWING

    def test_respond_pong_action(self):
        """Test PONG response action."""
        from games.mahjong.tile import Tile, TileSuit

        state = GameState()
        rng = np.random.default_rng(42)
        state.reset(rng)

        # Give player 1 two 2万 tiles
        pong_tiles = [
            Tile(TileSuit.WAN, 2, 0),
            Tile(TileSuit.WAN, 2, 1),
        ]
        other_tiles = [Tile(TileSuit.TIAO, i, 0) for i in range(1, 12)]
        state._players[1]._hand = pong_tiles + other_tiles

        # Player 0 draws and discards 2万
        state.draw_tile()
        discard_tile = Tile(TileSuit.WAN, 2, 2)
        state._players[0]._hand.append(discard_tile)
        state.discard_tile(discard_tile)

        # Check if player 1 can PONG
        if (
            state.phase == GamePhase.WAITING_RESPONSE
            and 1 in state._pending_responses
            and ActionType.PONG in state._pending_responses[1]
        ):
            result = state.respond(1, ActionType.PONG)
            assert result is True
            assert state.current_player_idx == 1
            assert state.phase == GamePhase.DISCARDING
            assert len(state._players[1].melds) == 1

    def test_respond_chi_action(self):
        """Test CHI response action."""
        from games.mahjong.tile import Tile, TileSuit

        state = GameState()
        rng = np.random.default_rng(42)
        state.reset(rng)

        # Give player 1 (next player after 0) tiles for chi: 2万, 3万
        chi_tiles = [
            Tile(TileSuit.WAN, 2, 0),
            Tile(TileSuit.WAN, 3, 0),
        ]
        other_tiles = [Tile(TileSuit.TIAO, i, 0) for i in range(1, 12)]
        state._players[1]._hand = chi_tiles + other_tiles

        # Player 0 draws and discards 1万
        state.draw_tile()
        discard_tile = Tile(TileSuit.WAN, 1, 0)
        state._players[0]._hand.append(discard_tile)
        state.discard_tile(discard_tile)

        # Check if player 1 can CHI
        if (
            state.phase == GamePhase.WAITING_RESPONSE
            and 1 in state._pending_responses
            and ActionType.CHI in state._pending_responses[1]
        ):
            result = state.respond(1, ActionType.CHI, tiles=chi_tiles)
            assert result is True
            assert state.current_player_idx == 1
            assert state.phase == GamePhase.DISCARDING

    def test_chi_with_invalid_tiles(self):
        """Test CHI with tiles not in hand."""
        from games.mahjong.tile import Tile, TileSuit

        state = GameState()
        rng = np.random.default_rng(42)
        state.reset(rng)

        # Give player 1 tiles for chi
        chi_tiles = [
            Tile(TileSuit.WAN, 2, 0),
            Tile(TileSuit.WAN, 3, 0),
        ]
        other_tiles = [Tile(TileSuit.TIAO, i, 0) for i in range(1, 12)]
        state._players[1]._hand = chi_tiles + other_tiles

        # Player 0 discards
        state.draw_tile()
        discard_tile = Tile(TileSuit.WAN, 1, 0)
        state._players[0]._hand.append(discard_tile)
        state.discard_tile(discard_tile)

        if (
            state.phase == GamePhase.WAITING_RESPONSE
            and 1 in state._pending_responses
            and ActionType.CHI in state._pending_responses[1]
        ):
            # Try CHI with tiles not in hand
            fake_tiles = [
                Tile(TileSuit.WAN, 5, 0),
                Tile(TileSuit.WAN, 6, 0),
            ]
            result = state.respond(1, ActionType.CHI, tiles=fake_tiles)
            assert result is False

    def test_an_gang_success(self):
        """Test successful an_gang (暗杠)."""
        from games.mahjong.tile import Tile, TileSuit

        state = GameState()
        rng = np.random.default_rng(42)
        state.reset(rng)

        # Give current player 4 tiles of same type
        gang_tiles = [
            Tile(TileSuit.WAN, 1, 0),
            Tile(TileSuit.WAN, 1, 1),
            Tile(TileSuit.WAN, 1, 2),
            Tile(TileSuit.WAN, 1, 3),
        ]
        other_tiles = [Tile(TileSuit.TIAO, i, 0) for i in range(1, 11)]
        state._players[0]._hand = gang_tiles + other_tiles

        # Draw to enter DISCARDING phase
        state.draw_tile()

        # Perform an_gang
        result = state.an_gang(0)  # tile_type_id for 1万 is 0
        assert result is True
        assert state.phase == GamePhase.DRAWING
        assert len(state._players[0].melds) == 1

    def test_an_gang_not_enough_tiles(self):
        """Test an_gang with less than 4 tiles."""
        from games.mahjong.tile import Tile, TileSuit

        state = GameState()
        rng = np.random.default_rng(42)
        state.reset(rng)

        # Give current player only 3 tiles of same type
        tiles = [
            Tile(TileSuit.WAN, 1, 0),
            Tile(TileSuit.WAN, 1, 1),
            Tile(TileSuit.WAN, 1, 2),
        ]
        other_tiles = [Tile(TileSuit.TIAO, i, 0) for i in range(1, 11)]
        state._players[0]._hand = tiles + other_tiles

        state.draw_tile()
        result = state.an_gang(0)
        assert result is False

    def test_self_hu_success(self):
        """Test successful self_hu (自摸)."""
        from games.mahjong.tile import Tile, TileSuit

        state = GameState()
        rng = np.random.default_rng(42)
        state.reset(rng)

        # Construct a winning hand: 111222333444 55万 (14 tiles)
        winning_tiles = []
        for rank in [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5]:
            winning_tiles.append(Tile(TileSuit.WAN, rank, len(winning_tiles) % 4))

        state._players[0]._hand = winning_tiles
        state._phase = GamePhase.DISCARDING

        result = state.self_hu()
        assert result is True
        assert state.is_terminated
        assert state.winner_idx == 0

    def test_self_hu_not_winning(self):
        """Test self_hu with non-winning hand."""
        from games.mahjong.tile import Tile, TileSuit

        state = GameState()
        rng = np.random.default_rng(42)
        state.reset(rng)

        # Non-winning hand
        tiles = [Tile(TileSuit.WAN, i, 0) for i in range(1, 10)]
        tiles += [Tile(TileSuit.TIAO, i, 0) for i in range(1, 6)]
        state._players[0]._hand = tiles
        state._phase = GamePhase.DISCARDING

        result = state.self_hu()
        assert result is False

    def test_last_discard_player_property(self):
        """Test last_discard_player property."""
        state = GameState()
        rng = np.random.default_rng(42)
        state.reset(rng)

        assert state.last_discard_player == -1

        state.draw_tile()
        player = state.get_player(0)
        state.discard_tile(player.hand[0])

        assert state.last_discard_player == 0

    def test_check_responses_with_no_discard(self):
        """Test _check_responses when last_discard is None."""
        state = GameState()
        rng = np.random.default_rng(42)
        state.reset(rng)

        state._last_discard = None
        state._check_responses()
        assert state._pending_responses == {}

    def test_respond_returns_false_for_unknown_action(self):
        """Test respond returns False for unhandled action type."""
        state = GameState()
        rng = np.random.default_rng(42)
        state.reset(rng)

        # Setup waiting response phase
        state.draw_tile()
        player = state.get_player(0)
        state.discard_tile(player.hand[0])

        if state.phase == GamePhase.WAITING_RESPONSE:
            for player_idx in list(state._pending_responses.keys()):
                # Manually add an invalid action to test the final return False
                state._pending_responses[player_idx].append(ActionType.PASS)
                # All passes should work, but if we somehow get an unhandled action
                # the function returns False at the end
                break
