"""Kobayakawa game environment."""

from typing import Any

import numpy as np
from gymnasium import spaces

from games.kobayakawa.constants import CardConsts, PlayerConsts
from games.kobayakawa.enums import ActionType, GamePhase
from games.kobayakawa.game_state import GameState
from games.oink_game import OinkGameEnv
from games.registry import register_game


@register_game("kobayakawa")
class KobayakawaGameEnv(OinkGameEnv):
    """Kobayakawa card game environment.

    A minimalist betting game where players have one card.
    The lowest card gets bonus from the Kobayakawa card.
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

        self.action_space = spaces.Discrete(len(ActionType))
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
        if player.card:
            obs[idx] = player.card.value / CardConsts.MAX_VALUE
        idx += 1

        if self._game_state.kobayakawa_card:
            obs[idx] = self._game_state.kobayakawa_card.value / CardConsts.MAX_VALUE
        idx += 1

        obs[idx] = self._game_state.phase / len(GamePhase)
        idx += 1

        obs[idx] = self._game_state.round / 7
        idx += 1

        obs[idx] = self._game_state.pot / 20
        idx += 1

        for i in range(self._num_players):
            p = self._game_state.get_player(i)
            if idx < self.OBSERVATION_SIZE:
                obs[idx] = p.coins / 10
                idx += 1
            if idx < self.OBSERVATION_SIZE:
                obs[idx] = 1.0 if p.has_bet else 0.0
                idx += 1
            if idx < self.OBSERVATION_SIZE:
                obs[idx] = 0.0 if p.is_eliminated else 1.0
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
                    "card": p.card.value if p.card else None,
                    "coins": p.coins,
                    "has_bet": p.has_bet,
                    "eliminated": p.is_eliminated,
                }
            )

        return {
            "current_player": self._current_player_idx,
            "num_players": self._num_players,
            "players": players_info,
            "kobayakawa": (
                self._game_state.kobayakawa_card.value
                if self._game_state.kobayakawa_card
                else None
            ),
            "phase": self._game_state.phase.name,
            "round": self._game_state.round,
            "pot": self._game_state.pot,
        }

    def _get_action_mask(self, player_idx: int) -> list[int]:
        mask = [0] * len(ActionType)

        if self._game_state is None:
            return mask

        player = self._game_state.get_player(player_idx)
        phase = self._game_state.phase

        if player.is_eliminated:
            return mask

        if phase == GamePhase.DRAW_OR_SWAP:
            if self._game_state.dealer.deck_count > 0:
                mask[ActionType.DRAW] = 1
                mask[ActionType.SWAP] = 1
            mask[ActionType.PASS] = 1

        elif phase == GamePhase.BETTING:
            if player.coins > 0 and not player.has_bet:
                mask[ActionType.BET] = 1
            mask[ActionType.PASS] = 1

        return mask

    def _apply_action(self, action: Any) -> tuple[float, bool]:
        if self._game_state is None:
            return 0.0, True

        player = self._game_state.get_current_player()
        phase = self._game_state.phase

        if phase == GamePhase.DRAW_OR_SWAP:
            self._handle_draw_or_swap(action, player)
        elif phase == GamePhase.BETTING:
            self._handle_betting(action, player)

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

    def _handle_draw_or_swap(self, action: int, player) -> None:
        if self._game_state is None:
            return

        if action == ActionType.DRAW:
            new_card = self._game_state.dealer.deal_one()
            if new_card:
                player.set_card(new_card)

        elif action == ActionType.SWAP:
            new_card = self._game_state.dealer.deal_one()
            if new_card:
                old_card = player.swap_card(new_card)
                if old_card:
                    self._game_state.set_kobayakawa(old_card)

        self._advance_draw_phase()

    def _handle_betting(self, action: int, player) -> None:
        if self._game_state is None:
            return

        if action == ActionType.BET:
            player.place_bet()
            self._game_state.add_to_pot(1)

        self._advance_betting_phase()

    def _advance_draw_phase(self) -> None:
        if self._game_state is None:
            return

        self._game_state.next_player()

        if self._game_state.current_player_idx == 0:
            self._game_state.set_phase(GamePhase.BETTING)

    def _advance_betting_phase(self) -> None:
        if self._game_state is None:
            return

        self._game_state.next_player()

        if self._game_state.current_player_idx == 0:
            self._game_state.resolve_showdown()
            if not self._game_state.is_terminated:
                self._game_state.start_new_round(self._rng)

    def _render_text(self) -> str:
        if self._game_state is None:
            return "Game not initialized"

        lines = []
        lines.append(f"=== Kobayakawa (Players: {self._num_players}) ===")
        lines.append(f"Round: {self._game_state.round}/7")
        lines.append(f"Phase: {self._game_state.phase.name}")
        lines.append(f"Pot: {self._game_state.pot} coins")
        lines.append("")

        if self._game_state.kobayakawa_card:
            lines.append(f"Kobayakawa Card: {self._game_state.kobayakawa_card}")
        lines.append("")

        lines.append("Players:")
        for i in range(self._num_players):
            p = self._game_state.get_player(i)
            status = " (eliminated)" if p.is_eliminated else ""
            bet = " [BET]" if p.has_bet else ""
            marker = " <--" if i == self._current_player_idx else ""
            card_str = str(p.card.value) if p.card else "?"

            if i == self._current_player_idx:
                lines.append(
                    f"  Player {i}: Card={card_str}, "
                    f"Coins={p.coins}{bet}{status}{marker}"
                )
            else:
                lines.append(f"  Player {i}: Card=?, Coins={p.coins}{bet}{status}")

        return "\n".join(lines)
