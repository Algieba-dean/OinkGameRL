"""Maskmen game environment."""

from typing import Any

import numpy as np
from gymnasium import spaces

from games.board_game import BoardGameEnv
from games.maskmen.constants import PlayerConsts
from games.maskmen.dealer import Dealer
from games.maskmen.enums import CardColor
from games.maskmen.game_state import GameState
from games.registry import register_game


@register_game("maskmen")
class MaskmenGameEnv(BoardGameEnv):
    """Maskmen card game environment.

    Players collect sets of colored cards. First to collect 3 sets wins.
    """

    MAX_HAND_SIZE = 6
    NUM_COLORS = 6
    NUM_VALUES = 3
    OBSERVATION_SIZE = 100

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
        self._dealer = Dealer(random_generator=self._rng)
        self._game_state: GameState | None = None

        self._build_action_space()
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.OBSERVATION_SIZE,), dtype=np.float32
        )

    def _build_action_space(self) -> None:
        """Build action space: play card (index 0-5) or pass."""
        self.action_space = spaces.Discrete(self.MAX_HAND_SIZE + 1)

    def _reset_logic(
        self, seed: int | None, options: dict[str, Any] | None = None
    ) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        else:
            self._rng = np.random.default_rng()

        self._dealer.reset(random_generator=self._rng)
        player_hands, deck = self._dealer.deal_cards(player_num=self._num_players)

        self._game_state = GameState(
            player_num=self._num_players,
            player_hands=player_hands,
            deck=deck,
        )
        self._current_player_idx = 0

    def _get_observation(self, player_idx: int) -> np.ndarray:
        obs = np.zeros(self.OBSERVATION_SIZE, dtype=np.float32)

        if self._game_state is None:
            return obs

        idx = 0
        player = self._game_state.get_player(player_idx)
        for card in player.hand:
            if idx + 2 < self.OBSERVATION_SIZE:
                obs[idx] = card.color / self.NUM_COLORS
                obs[idx + 1] = card.value / self.NUM_VALUES
                idx += 2

        idx = 20
        for color in CardColor:
            table_cards = self._game_state.table[color]
            obs[idx] = len(table_cards) / 3
            idx += 1

        idx = 30
        for i in range(self._num_players):
            p = self._game_state.get_player(i)
            if idx < self.OBSERVATION_SIZE:
                obs[idx] = p.set_count / 3
                idx += 1

        idx = 40
        obs[idx] = self._game_state.deck_count / 18

        idx = 50
        for i in range(self._num_players):
            p = self._game_state.get_player(i)
            if idx < self.OBSERVATION_SIZE:
                obs[idx] = p.hand_count / self.MAX_HAND_SIZE
                idx += 1

        return obs

    def _get_global_state(self) -> dict[str, Any]:
        if self._game_state is None:
            return {}

        hands = []
        for i in range(self._num_players):
            player = self._game_state.get_player(i)
            hand_repr = [{"color": c.color.name, "value": c.value} for c in player.hand]
            hands.append(hand_repr)

        table_repr = {}
        for color in CardColor:
            cards = self._game_state.table[color]
            table_repr[color.name] = [{"value": c.value} for c in cards]

        return {
            "current_player": self._current_player_idx,
            "num_players": self._num_players,
            "hands": hands,
            "table": table_repr,
            "deck_count": self._game_state.deck_count,
            "sets": [
                self._game_state.get_player(i).set_count
                for i in range(self._num_players)
            ],
        }

    def _get_action_mask(self, player_idx: int) -> list[int]:
        mask = [0] * (self.MAX_HAND_SIZE + 1)

        if self._game_state is None:
            return mask

        player = self._game_state.get_player(player_idx)

        for i in range(player.hand_count):
            mask[i] = 1

        mask[self.MAX_HAND_SIZE] = 1

        return mask

    def _apply_action(self, action: Any) -> tuple[float, bool]:
        if self._game_state is None:
            return 0.0, True

        player = self._game_state.get_current_player()

        if action < player.hand_count:
            card = player.play_card(action)
            self._game_state.play_card_to_table(card)

            drawn = self._game_state.draw_card()
            if drawn:
                player.add_card(drawn)

        terminated = self._game_state.is_terminated

        reward = 0.0
        if terminated:
            winner = self._game_state.get_winner()
            if winner == self._current_player_idx:
                reward = 1.0
            elif winner is not None:
                reward = -1.0

        if not terminated:
            self._game_state.next_player()
            self._current_player_idx = self._game_state.current_player_idx

        return reward, terminated

    def _render_text(self) -> str:
        if self._game_state is None:
            return "Game not initialized"

        lines = []
        lines.append(f"=== Maskmen (Players: {self._num_players}) ===")
        lines.append(f"Current Player: {self._current_player_idx}")
        lines.append(f"Deck: {self._game_state.deck_count} cards")
        lines.append("")

        lines.append("Table:")
        for color in CardColor:
            cards = self._game_state.table[color]
            if cards:
                cards_str = " ".join(str(c) for c in cards)
                lines.append(f"  {color.name}: {cards_str}")
        lines.append("")

        lines.append("Players:")
        for i in range(self._num_players):
            player = self._game_state.get_player(i)
            sets_str = ", ".join(c.name for c in player.collected_sets)
            marker = " (current)" if i == self._current_player_idx else ""
            lines.append(
                f"  Player {i}{marker}: {player.hand_count} cards, "
                f"{player.set_count} sets [{sets_str}]"
            )

        return "\n".join(lines)
