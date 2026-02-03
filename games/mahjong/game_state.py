"""Game state for Mahjong (麻将) game."""

from __future__ import annotations

import numpy as np

from games.mahjong.constants import GameConsts
from games.mahjong.enums import ActionType, GamePhase, MeldType
from games.mahjong.hand_checker import HandChecker
from games.mahjong.meld import Meld
from games.mahjong.player import Player
from games.mahjong.tile import Tile, create_full_tileset


class GameState:
    """Manages the state of a Mahjong game."""

    def __init__(self) -> None:
        self._players: list[Player] = [Player(i) for i in range(GameConsts.NUM_PLAYERS)]
        self._current_player_idx: int = 0
        self._phase: GamePhase = GamePhase.DRAWING
        self._wall: list[Tile] = []
        self._last_discard: Tile | None = None
        self._last_discard_player: int = -1
        self._pending_responses: dict[int, list[ActionType]] = {}
        self._winner_idx: int = -1

    @property
    def current_player_idx(self) -> int:
        return self._current_player_idx

    @property
    def phase(self) -> GamePhase:
        return self._phase

    @property
    def wall_count(self) -> int:
        return len(self._wall)

    @property
    def last_discard(self) -> Tile | None:
        return self._last_discard

    @property
    def last_discard_player(self) -> int:
        return self._last_discard_player

    @property
    def is_terminated(self) -> bool:
        return self._phase == GamePhase.FINISHED

    @property
    def winner_idx(self) -> int:
        return self._winner_idx

    def get_player(self, player_idx: int) -> Player:
        """Get player by index."""
        return self._players[player_idx]

    def reset(self, rng: np.random.Generator) -> None:
        """Reset game state and deal tiles."""
        # Reset players
        for player in self._players:
            player.reset()

        # Shuffle tiles
        tiles = create_full_tileset()
        tile_indices = list(range(len(tiles)))
        rng.shuffle(tile_indices)
        shuffled_tiles = [tiles[i] for i in tile_indices]

        # Deal 13 tiles to each player
        idx = 0
        for i in range(GameConsts.NUM_PLAYERS):
            hand = shuffled_tiles[idx : idx + GameConsts.HAND_SIZE]
            self._players[i].set_hand(hand)
            idx += GameConsts.HAND_SIZE

        # Remaining tiles form the wall
        self._wall = shuffled_tiles[idx:]

        # Reset game state
        self._current_player_idx = 0
        self._phase = GamePhase.DRAWING
        self._last_discard = None
        self._last_discard_player = -1
        self._pending_responses = {}
        self._winner_idx = -1

    def draw_tile(self) -> Tile | None:
        """Draw a tile from the wall."""
        if not self._wall:
            self._phase = GamePhase.FINISHED  # Draw (流局)
            return None

        tile = self._wall.pop(0)
        player = self._players[self._current_player_idx]
        player.add_tile(tile)
        self._phase = GamePhase.DISCARDING
        return tile

    def discard_tile(self, tile: Tile) -> bool:
        """Discard a tile."""
        if self._phase != GamePhase.DISCARDING:
            return False

        player = self._players[self._current_player_idx]
        if not player.discard_tile(tile):
            return False

        self._last_discard = tile
        self._last_discard_player = self._current_player_idx

        # Check for responses from other players
        self._check_responses()

        if self._pending_responses:
            self._phase = GamePhase.WAITING_RESPONSE
        else:
            self._advance_player()
            self._phase = GamePhase.DRAWING

        return True

    def _check_responses(self) -> None:
        """Check what responses other players can make."""
        self._pending_responses = {}
        if self._last_discard is None:
            return

        for i in range(GameConsts.NUM_PLAYERS):
            if i == self._current_player_idx:
                continue

            player = self._players[i]
            actions: list[ActionType] = []

            # Check hu (胡)
            test_hand = player.hand + [self._last_discard]
            if HandChecker.is_winning_hand(test_hand, player.melds):
                actions.append(ActionType.HU)

            # Check gang (杠)
            if HandChecker.can_gang(player.hand, self._last_discard):
                actions.append(ActionType.GANG)

            # Check pong (碰)
            if HandChecker.can_pong(player.hand, self._last_discard):
                actions.append(ActionType.PONG)

            # Check chi (吃) - only next player
            next_player = (self._current_player_idx + 1) % GameConsts.NUM_PLAYERS
            if i == next_player and HandChecker.can_chi(
                player.hand, self._last_discard
            ):
                actions.append(ActionType.CHI)

            if actions:
                actions.append(ActionType.PASS)
                self._pending_responses[i] = actions

    def respond(
        self, player_idx: int, action: ActionType, tiles: list[Tile] | None = None
    ) -> bool:
        """Handle a player's response to a discard."""
        if self._phase != GamePhase.WAITING_RESPONSE:
            return False

        if player_idx not in self._pending_responses:
            return False

        if action not in self._pending_responses[player_idx]:
            return False

        if self._last_discard is None:
            return False

        player = self._players[player_idx]

        if action == ActionType.PASS:
            del self._pending_responses[player_idx]
            if not self._pending_responses:
                self._advance_player()
                self._phase = GamePhase.DRAWING
            return True

        if action == ActionType.HU:
            player.add_tile(self._last_discard)
            player.mark_winner()
            self._winner_idx = player_idx
            self._phase = GamePhase.FINISHED
            self._pending_responses = {}
            return True

        if action == ActionType.GANG:
            hand_tiles = player.get_tiles_of_type(self._last_discard.tile_type_id)[:3]
            player.remove_tiles(hand_tiles)
            meld = Meld(
                MeldType.MING_GANG,
                tuple(hand_tiles + [self._last_discard]),
                self._last_discard_player,
            )
            player.add_meld(meld)
            self._current_player_idx = player_idx
            self._pending_responses = {}
            self._phase = GamePhase.DRAWING  # Gang player draws
            return True

        if action == ActionType.PONG:
            hand_tiles = player.get_tiles_of_type(self._last_discard.tile_type_id)[:2]
            player.remove_tiles(hand_tiles)
            meld = Meld(
                MeldType.PONG,
                tuple(hand_tiles + [self._last_discard]),
                self._last_discard_player,
            )
            player.add_meld(meld)
            self._current_player_idx = player_idx
            self._pending_responses = {}
            self._phase = GamePhase.DISCARDING
            return True

        if action == ActionType.CHI:
            if tiles is None or len(tiles) != 2:
                return False
            if not player.remove_tiles(tiles):
                return False
            meld = Meld(
                MeldType.CHI,
                tuple(sorted([tiles[0], tiles[1], self._last_discard])),
                self._last_discard_player,
            )
            player.add_meld(meld)
            self._current_player_idx = player_idx
            self._pending_responses = {}
            self._phase = GamePhase.DISCARDING
            return True

        return False

    def an_gang(self, tile_type_id: int) -> bool:
        """Perform an_gang (暗杠) with 4 tiles of same type from hand."""
        if self._phase != GamePhase.DISCARDING:
            return False

        player = self._players[self._current_player_idx]
        tiles = player.get_tiles_of_type(tile_type_id)

        if len(tiles) != 4:
            return False

        player.remove_tiles(tiles)
        meld = Meld(MeldType.AN_GANG, tuple(tiles), None)
        player.add_meld(meld)

        # After an_gang, player draws again
        self._phase = GamePhase.DRAWING
        return True

    def self_hu(self) -> bool:
        """Declare self-drawn win (自摸)."""
        if self._phase != GamePhase.DISCARDING:
            return False

        player = self._players[self._current_player_idx]
        if HandChecker.is_winning_hand(player.hand, player.melds):
            player.mark_winner()
            self._winner_idx = self._current_player_idx
            self._phase = GamePhase.FINISHED
            return True

        return False

    def _advance_player(self) -> None:
        """Move to next player."""
        self._current_player_idx = (
            self._current_player_idx + 1
        ) % GameConsts.NUM_PLAYERS

    def get_valid_discards(self) -> list[Tile]:
        """Get list of tiles that can be discarded."""
        if self._phase != GamePhase.DISCARDING:
            return []
        player = self._players[self._current_player_idx]
        # Return unique tiles (by type)
        seen: set[int] = set()
        result: list[Tile] = []
        for tile in player.hand:
            if tile.tile_type_id not in seen:
                seen.add(tile.tile_type_id)
                result.append(tile)
        return result
