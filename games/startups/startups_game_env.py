"""Startups game environment."""

from typing import Any

import numpy as np
from gymnasium import spaces

from games.board_game import BoardGameEnv
from games.registry import register_game
from games.startups.constants import CardConsts, PlayerConsts
from games.startups.enums import Company
from games.startups.game_state import GameState


@register_game("startups")
class StartupsGameEnv(BoardGameEnv):
    """Startups card game environment.

    Players invest in startup companies by playing and taking cards.
    """

    MAX_HAND_SIZE = 6
    MAX_MARKET_SIZE = 7
    OBSERVATION_SIZE = 150

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

        self.action_space = spaces.Discrete(
            self.MAX_HAND_SIZE + self.MAX_MARKET_SIZE + 1
        )
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
            if idx + 2 < self.OBSERVATION_SIZE:
                obs[idx] = card.company / len(Company)
                obs[idx + 1] = card.value / 6
                idx += 2

        idx = 20
        for card in self._game_state.market:
            if idx + 2 < self.OBSERVATION_SIZE:
                obs[idx] = card.company / len(Company)
                obs[idx + 1] = card.value / 6
                idx += 2

        idx = 40
        for i in range(self._num_players):
            p = self._game_state.get_player(i)
            if idx < self.OBSERVATION_SIZE:
                obs[idx] = p.coins / 20
                idx += 1
            for company in Company:
                if idx < self.OBSERVATION_SIZE:
                    obs[idx] = p.get_company_count(company) / 6
                    idx += 1

        idx = 100
        obs[idx] = self._game_state.dealer.deck_count / CardConsts.TOTAL_CARDS

        return obs

    def _get_global_state(self) -> dict[str, Any]:
        if self._game_state is None:
            return {}

        players_info = []
        for i in range(self._num_players):
            p = self._game_state.get_player(i)
            players_info.append(
                {
                    "hand": [
                        {"company": c.company.name, "value": c.value} for c in p.hand
                    ],
                    "tableau": {
                        co.name: [c.value for c in cards]
                        for co, cards in p.tableau.items()
                    },
                    "coins": p.coins,
                }
            )

        return {
            "current_player": self._current_player_idx,
            "num_players": self._num_players,
            "players": players_info,
            "market": [
                {"company": c.company.name, "value": c.value}
                for c in self._game_state.market
            ],
            "deck_count": self._game_state.dealer.deck_count,
        }

    def _get_action_mask(self, player_idx: int) -> list[int]:
        mask = [0] * (self.MAX_HAND_SIZE + self.MAX_MARKET_SIZE + 1)

        if self._game_state is None:
            return mask

        player = self._game_state.get_player(player_idx)

        for i in range(player.hand_count):
            mask[i] = 1

        market_start = self.MAX_HAND_SIZE
        for i in range(min(len(self._game_state.market), self.MAX_MARKET_SIZE)):
            card = self._game_state.market[i]
            cost = self._get_take_cost(card.company)
            if player.coins >= cost:
                mask[market_start + i] = 1

        mask[-1] = 1

        return mask

    def _get_take_cost(self, company: Company) -> int:
        """Get cost to take a card of given company from market."""
        if self._game_state is None:
            return 0
        count = sum(1 for c in self._game_state.market if c.company == company)
        return max(0, count - 1)

    def _apply_action(self, action: Any) -> tuple[float, bool]:
        if self._game_state is None:
            return 0.0, True

        player = self._game_state.get_current_player()

        if action < self.MAX_HAND_SIZE and action < player.hand_count:
            card = player.play_card(action)
            self._game_state.add_to_market(card)
            self._draw_card_if_available(player)

        elif action < self.MAX_HAND_SIZE + self.MAX_MARKET_SIZE:
            market_idx = action - self.MAX_HAND_SIZE
            if market_idx < len(self._game_state.market):
                card = self._game_state.market[market_idx]
                cost = self._get_take_cost(card.company)
                if player.coins >= cost:
                    player.pay_coins(cost)
                    taken_card = self._game_state.take_from_market(market_idx)
                    player.add_to_tableau(taken_card)
                    self._draw_card_if_available(player)

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

    def _draw_card_if_available(self, player) -> None:
        """Draw a card from deck if available."""
        if self._game_state is None:
            return
        card = self._game_state.dealer.deal_one()
        if card:
            player.add_card_to_hand(card)

    def _render_text(self) -> str:
        if self._game_state is None:
            return "Game not initialized"

        lines = []
        lines.append(f"=== Startups (Players: {self._num_players}) ===")
        lines.append(f"Deck: {self._game_state.dealer.deck_count} cards")
        lines.append("")

        lines.append("Market:")
        market_str = " ".join(str(c) for c in self._game_state.market)
        lines.append(f"  {market_str}")
        lines.append("")

        lines.append("Players:")
        for i in range(self._num_players):
            p = self._game_state.get_player(i)
            marker = " <--" if i == self._current_player_idx else ""
            tableau_parts = []
            for co in Company:
                count = p.get_company_count(co)
                if count > 0:
                    tableau_parts.append(f"{co.name[0]}:{count}")
            tableau_str = ", ".join(tableau_parts) if tableau_parts else "none"
            lines.append(
                f"  Player {i}: {p.hand_count} cards, "
                f"${p.coins}, tableau=[{tableau_str}]{marker}"
            )

        return "\n".join(lines)
