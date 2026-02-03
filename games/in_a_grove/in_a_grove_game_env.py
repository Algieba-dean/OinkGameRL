"""In a Grove game environment."""

from typing import Any

import numpy as np
from gymnasium import spaces

from games.board_game import BoardGameEnv
from games.in_a_grove.constants import PlayerConsts
from games.in_a_grove.enums import GamePhase, TileType
from games.in_a_grove.game_state import GameState
from games.registry import register_game


@register_game("in_a_grove")
class InAGroveGameEnv(BoardGameEnv):
    """In a Grove deduction game environment.

    Players deduce who is the culprit, witness, or accomplice.
    """

    OBSERVATION_SIZE = 50

    def __init__(self, player_num: int = 4, render_mode: str | None = None) -> None:
        if player_num not in PlayerConsts.ALLOWED_PLAYER_NUM:
            raise ValueError(
                f"Invalid player_num {player_num}, "
                f"allowed: {PlayerConsts.ALLOWED_PLAYER_NUM}"
            )

        super().__init__(render_mode=render_mode)
        self._num_players = player_num
        self._current_player_idx = 0

        self._rng: np.random.Generator = np.random.default_rng()
        self._game_state: GameState | None = None

        self.action_space = spaces.Discrete(len(TileType))
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

        self._game_state = GameState(player_num=self._num_players)
        self._game_state.reset(rng=self._rng)
        self._current_player_idx = self._game_state.current_player_idx

    def _get_observation(self, player_idx: int) -> np.ndarray:
        obs = np.zeros(self.OBSERVATION_SIZE, dtype=np.float32)

        if self._game_state is None:
            return obs

        idx = 0
        player = self._game_state.get_player(player_idx)

        for card in player.hand:
            if idx < self.OBSERVATION_SIZE:
                obs[idx] = card.value / 8
                idx += 1

        idx = 10
        for tile in player.tiles:
            if idx < self.OBSERVATION_SIZE:
                obs[idx] = (tile + 1) / len(TileType)
                idx += 1

        idx = 20
        if self._game_state.center_card:
            obs[idx] = self._game_state.center_card.value / 8

        idx = 25
        obs[idx] = self._game_state.round / 3
        obs[idx + 1] = self._game_state.phase / len(GamePhase)

        idx = 30
        for i in range(self._num_players):
            p = self._game_state.get_player(i)
            if idx < self.OBSERVATION_SIZE:
                obs[idx] = p.score / 20
                idx += 1

        return obs

    def _get_global_state(self) -> dict[str, Any]:
        if self._game_state is None:
            return {}

        players_info = []
        for i in range(self._num_players):
            p = self._game_state.get_player(i)
            players_info.append(
                {
                    "hand": [c.value for c in p.hand],
                    "tiles": [t.name for t in p.tiles],
                    "score": p.score,
                    "vote": p.current_vote.name if p.current_vote else None,
                }
            )

        return {
            "current_player": self._current_player_idx,
            "num_players": self._num_players,
            "players": players_info,
            "center_card": self._game_state.center_card.value
            if self._game_state.center_card
            else None,
            "phase": self._game_state.phase.name,
            "round": self._game_state.round,
        }

    def _get_action_mask(self, player_idx: int) -> list[int]:
        mask = [0] * len(TileType)

        if self._game_state is None:
            return mask

        player = self._game_state.get_player(player_idx)

        for tile in player.tiles:
            mask[tile] = 1

        return mask

    def _apply_action(self, action: Any) -> tuple[float, bool]:
        if self._game_state is None:
            return 0.0, True

        player = self._game_state.get_current_player()
        tile_type = TileType(action)

        if tile_type in player.tiles:
            player.vote(tile_type)

        self._game_state.next_player()

        if self._game_state.current_player_idx == 0:
            self._game_state.resolve_round()
            if not self._game_state.is_terminated:
                self._game_state.start_new_round(self._rng)

        terminated = self._game_state.is_terminated

        reward = 0.0
        if terminated:
            winner = self._game_state.get_winner()
            if winner == self._current_player_idx:
                reward = 1.0
            elif winner is not None:
                reward = -1.0

        if not terminated:
            self._current_player_idx = self._game_state.current_player_idx

        return reward, terminated

    def _render_text(self) -> str:
        if self._game_state is None:
            return "Game not initialized"

        lines = []
        lines.append(f"=== In a Grove (Players: {self._num_players}) ===")
        lines.append(f"Round: {self._game_state.round}/3")
        lines.append(f"Phase: {self._game_state.phase.name}")

        if self._game_state.center_card:
            lines.append(f"Center Card: {self._game_state.center_card}")
        lines.append("")

        lines.append("Players:")
        for i in range(self._num_players):
            p = self._game_state.get_player(i)
            marker = " <--" if i == self._current_player_idx else ""
            tiles_str = ", ".join(t.name for t in p.tiles)
            vote_str = f" [voted: {p.current_vote.name}]" if p.current_vote else ""
            lines.append(
                f"  Player {i}: {p.hand_count} cards, "
                f"score={p.score}, tiles=[{tiles_str}]{vote_str}{marker}"
            )

        return "\n".join(lines)
