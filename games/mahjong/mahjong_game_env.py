"""Mahjong (麻将) game environment."""

from typing import Any

import numpy as np
from gymnasium import spaces

from games.board_game import BoardGameEnv
from games.mahjong.constants import GameConsts, TileConsts
from games.mahjong.enums import ActionType, GamePhase
from games.mahjong.game_state import GameState
from games.mahjong.hand_checker import HandChecker
from games.mahjong.tile import Tile
from games.registry import register_game


@register_game("mahjong")
class MahjongGameEnv(BoardGameEnv):
    """Mahjong (麻将) game environment.

    A simplified Chinese Mahjong for 4 players.
    136 tiles, 13 tiles per player initially.

    Action space:
    - 0: Pass
    - 1-136: Discard tile (by tile_id)
    - 137: Draw
    - 138: Self Hu (自摸)
    - 139-172: Chi (with specific tile combination)
    - 173: Pong
    - 174: Gang (明杠)
    - 175-208: An Gang (暗杠, by tile type)
    - 209: Hu (点炮)
    """

    OBSERVATION_SIZE = 500
    MAX_ACTIONS = 250

    # Action indices
    ACTION_PASS = 0
    ACTION_DISCARD_START = 1
    ACTION_DISCARD_END = 136
    ACTION_DRAW = 137
    ACTION_SELF_HU = 138
    ACTION_CHI_START = 139
    ACTION_CHI_END = 172
    ACTION_PONG = 173
    ACTION_GANG = 174
    ACTION_AN_GANG_START = 175
    ACTION_AN_GANG_END = 208
    ACTION_HU = 209

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__(render_mode=render_mode)
        self._num_players = GameConsts.NUM_PLAYERS
        self._current_player_idx = 0

        self._rng: np.random.Generator = np.random.default_rng()
        self._game_state: GameState | None = None

        self.action_space = spaces.Discrete(self.MAX_ACTIONS)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.OBSERVATION_SIZE,), dtype=np.float32
        )

    def _reset_logic(
        self, seed: int | None, options: dict[str, Any] | None = None
    ) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        else:
            self._rng = np.random.default_rng()

        self._game_state = GameState()
        self._game_state.reset(rng=self._rng)
        self._current_player_idx = self._game_state.current_player_idx

    def _get_observation(self, player_idx: int) -> np.ndarray:
        obs = np.zeros(self.OBSERVATION_SIZE, dtype=np.float32)

        if self._game_state is None:
            return obs

        idx = 0

        # Player's hand (136 binary values)
        player = self._game_state.get_player(player_idx)
        for tile in player.hand:
            obs[tile.tile_id] = 1.0
        idx = 136

        # Last discard (136 binary values)
        if self._game_state.last_discard is not None:
            obs[idx + self._game_state.last_discard.tile_id] = 1.0
        idx = 272

        # Game phase (4 values)
        obs[idx + self._game_state.phase] = 1.0
        idx += 4

        # Current player (4 values)
        obs[idx + self._current_player_idx] = 1.0
        idx += 4

        # Wall count (normalized)
        obs[idx] = self._game_state.wall_count / GameConsts.TOTAL_TILES
        idx += 1

        # Discards for all players (34 * 4 = 136 values, count per type)
        for i in range(GameConsts.NUM_PLAYERS):
            p = self._game_state.get_player(i)
            for tile in p.discards:
                type_idx = tile.tile_type_id
                obs[idx + i * TileConsts.TOTAL_TILE_TYPES + type_idx] += 0.25
        idx += 136

        return obs

    def _get_global_state(self) -> dict[str, Any]:
        if self._game_state is None:
            return {}

        players_info = []
        for i in range(GameConsts.NUM_PLAYERS):
            p = self._game_state.get_player(i)
            players_info.append(
                {
                    "hand": [str(t) for t in p.hand],
                    "hand_count": p.hand_count,
                    "melds": [str(m) for m in p.melds],
                    "discards": [str(t) for t in p.discards],
                    "is_winner": p.is_winner,
                }
            )

        return {
            "current_player": self._current_player_idx,
            "num_players": self._num_players,
            "players": players_info,
            "phase": self._game_state.phase.name,
            "wall_count": self._game_state.wall_count,
            "last_discard": (
                str(self._game_state.last_discard)
                if self._game_state.last_discard
                else None
            ),
        }

    def _get_action_mask(self, player_idx: int) -> list[int]:
        mask = [0] * self.MAX_ACTIONS

        if self._game_state is None:
            return mask

        phase = self._game_state.phase
        player = self._game_state.get_player(player_idx)

        if phase == GamePhase.DRAWING:
            if player_idx == self._game_state.current_player_idx:
                mask[self.ACTION_DRAW] = 1

        elif phase == GamePhase.DISCARDING:
            if player_idx == self._game_state.current_player_idx:
                # Can discard any tile in hand
                for tile in player.hand:
                    mask[self.ACTION_DISCARD_START + tile.tile_id] = 1

                # Check self hu
                if HandChecker.is_winning_hand(player.hand, player.melds):
                    mask[self.ACTION_SELF_HU] = 1

                # Check an_gang
                an_gang_types = HandChecker.can_an_gang(player.hand)
                for type_id in an_gang_types:
                    mask[self.ACTION_AN_GANG_START + type_id] = 1

        elif (
            phase == GamePhase.WAITING_RESPONSE
            and player_idx in self._game_state._pending_responses
        ):
            actions = self._game_state._pending_responses[player_idx]

            if ActionType.PASS in actions:
                mask[self.ACTION_PASS] = 1

            if ActionType.HU in actions:
                mask[self.ACTION_HU] = 1

            if ActionType.GANG in actions:
                mask[self.ACTION_GANG] = 1

            if ActionType.PONG in actions:
                mask[self.ACTION_PONG] = 1

            if ActionType.CHI in actions:
                # Mark chi actions available
                discard = self._game_state.last_discard
                if discard is not None:
                    chi_options = HandChecker.can_chi(player.hand, discard)
                    for i, _ in enumerate(chi_options):
                        if self.ACTION_CHI_START + i < self.ACTION_CHI_END:
                            mask[self.ACTION_CHI_START + i] = 1

        return mask

    def _apply_action(self, action: Any) -> tuple[float, bool]:
        if self._game_state is None:
            return 0.0, True

        phase = self._game_state.phase
        reward = 0.0

        if phase == GamePhase.DRAWING:
            if action == self.ACTION_DRAW:
                self._game_state.draw_tile()

        elif phase == GamePhase.DISCARDING:
            if self.ACTION_DISCARD_START <= action <= self.ACTION_DISCARD_END:
                tile_id = action - self.ACTION_DISCARD_START
                tile = Tile.from_id(tile_id)
                self._game_state.discard_tile(tile)

            elif action == self.ACTION_SELF_HU:
                if self._game_state.self_hu():
                    reward = 1.0

            elif self.ACTION_AN_GANG_START <= action <= self.ACTION_AN_GANG_END:
                type_id = action - self.ACTION_AN_GANG_START
                self._game_state.an_gang(type_id)

        elif phase == GamePhase.WAITING_RESPONSE:
            player_idx = self._current_player_idx

            if action == self.ACTION_PASS:
                self._game_state.respond(player_idx, ActionType.PASS)

            elif action == self.ACTION_HU:
                if self._game_state.respond(player_idx, ActionType.HU):
                    reward = 1.0

            elif action == self.ACTION_GANG:
                self._game_state.respond(player_idx, ActionType.GANG)

            elif action == self.ACTION_PONG:
                self._game_state.respond(player_idx, ActionType.PONG)

            elif self.ACTION_CHI_START <= action <= self.ACTION_CHI_END:
                chi_idx = action - self.ACTION_CHI_START
                discard = self._game_state.last_discard
                if discard is not None:
                    player = self._game_state.get_player(player_idx)
                    chi_options = HandChecker.can_chi(player.hand, discard)
                    if chi_idx < len(chi_options):
                        tiles = list(chi_options[chi_idx])
                        self._game_state.respond(player_idx, ActionType.CHI, tiles)

        self._current_player_idx = self._game_state.current_player_idx
        terminated = self._game_state.is_terminated

        return reward, terminated

    def _render_text(self) -> str:
        if self._game_state is None:
            return "Game not initialized"

        lines = []
        lines.append("=== 麻将 (Mahjong) ===")
        lines.append(f"Phase: {self._game_state.phase.name}")
        lines.append(f"Current Player: {self._current_player_idx}")
        lines.append(f"Wall: {self._game_state.wall_count} tiles")
        lines.append("")

        lines.append("Players:")
        for i in range(GameConsts.NUM_PLAYERS):
            p = self._game_state.get_player(i)
            marker = " <--" if i == self._current_player_idx else ""
            hand_str = " ".join(str(t) for t in p.hand)
            melds_str = " | ".join(str(m) for m in p.melds) if p.melds else ""
            lines.append(f"  P{i}: [{p.hand_count}] {hand_str}{marker}")
            if melds_str:
                lines.append(f"      Melds: {melds_str}")

        if self._game_state.last_discard:
            lines.append(
                f"\nLast Discard (P{self._game_state.last_discard_player}): "
                f"{self._game_state.last_discard}"
            )

        if self._game_state.winner_idx >= 0:
            lines.append(f"\nWinner: Player {self._game_state.winner_idx}")

        return "\n".join(lines)
