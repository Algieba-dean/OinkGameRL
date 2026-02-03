"""Doudizhu (斗地主) game environment."""

from typing import Any

import numpy as np
from gymnasium import spaces

from games.board_game import BoardGameEnv
from games.doudizhu.card import Card
from games.doudizhu.constants import GameConsts
from games.doudizhu.enums import GamePhase, HandType
from games.doudizhu.game_state import GameState
from games.doudizhu.hand_detector import HandDetector
from games.registry import register_game


@register_game("doudizhu")
class DoudizhuGameEnv(BoardGameEnv):
    """Doudizhu (斗地主) card game environment.

    A popular Chinese card game for 3 players where one player (landlord)
    plays against two peasants.

    Action space:
    - During bidding: 0=pass, 1=bid for landlord
    - During playing: action index maps to card combinations
    """

    OBSERVATION_SIZE = 300
    MAX_ACTIONS = 500

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__(render_mode=render_mode)
        self._num_players = GameConsts.NUM_PLAYERS
        self._current_player_idx = 0

        self._rng: np.random.Generator = np.random.default_rng()
        self._game_state: GameState | None = None
        self._action_mapping: list[list[int]] = []

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
        self._build_action_mapping()

    def _build_action_mapping(self) -> None:
        """Build action mapping for current player's hand."""
        self._action_mapping = []
        # Action 0: Pass
        self._action_mapping.append([])

        if self._game_state is None:
            return

        # In bidding phase, action 1 = bid for landlord
        if self._game_state.phase == GamePhase.BIDDING:
            self._action_mapping.append([-1])  # Special marker for bid
            return

        # In playing phase, generate all valid card combinations
        player = self._game_state.get_player(self._current_player_idx)
        hand = player.hand

        # Generate all possible combinations
        combinations = self._generate_all_combinations(hand)

        for combo in combinations:
            card_ids = [card.card_id for card in combo]
            self._action_mapping.append(card_ids)

    def _generate_all_combinations(self, hand: list[Card]) -> list[list[Card]]:
        """Generate all valid card combinations from hand."""
        combinations: list[list[Card]] = []

        # Singles
        for card in hand:
            combinations.append([card])

        # Pairs
        rank_cards: dict[int, list[Card]] = {}
        for card in hand:
            rank = int(card.rank)
            if rank not in rank_cards:
                rank_cards[rank] = []
            rank_cards[rank].append(card)

        for _, cards in rank_cards.items():
            if len(cards) >= 2:
                combinations.append(cards[:2])
            if len(cards) >= 3:
                combinations.append(cards[:3])
            if len(cards) >= 4:
                combinations.append(cards[:4])

        # Triples with single/pair
        for rank, cards in rank_cards.items():
            if len(cards) >= 3:
                triple = cards[:3]
                # Triple with single
                for other_rank, other_cards in rank_cards.items():
                    if other_rank != rank and len(other_cards) >= 1:
                        combinations.append(triple + [other_cards[0]])
                # Triple with pair
                for other_rank, other_cards in rank_cards.items():
                    if other_rank != rank and len(other_cards) >= 2:
                        combinations.append(triple + other_cards[:2])

        # Straights (5+ consecutive singles)
        sorted_ranks = sorted(rank_cards.keys())
        for length in range(5, 13):
            for start_idx in range(len(sorted_ranks) - length + 1):
                ranks_slice = sorted_ranks[start_idx : start_idx + length]
                if self._is_consecutive_ranks(ranks_slice) and max(ranks_slice) < 12:
                    straight = [rank_cards[r][0] for r in ranks_slice]
                    combinations.append(straight)

        # Straight pairs (3+ consecutive pairs)
        pair_ranks = [r for r, cards in rank_cards.items() if len(cards) >= 2]
        pair_ranks.sort()
        for length in range(3, 11):
            for start_idx in range(len(pair_ranks) - length + 1):
                ranks_slice = pair_ranks[start_idx : start_idx + length]
                if self._is_consecutive_ranks(ranks_slice) and max(ranks_slice) < 12:
                    straight_pair = []
                    for r in ranks_slice:
                        straight_pair.extend(rank_cards[r][:2])
                    combinations.append(straight_pair)

        # Bombs (4 of same rank) - already added above

        # Rocket (both jokers)
        joker_cards = [c for c in hand if c.rank >= 13]
        if len(joker_cards) == 2:
            combinations.append(joker_cards)

        # Four with two singles
        for rank, cards in rank_cards.items():
            if len(cards) == 4:
                four = cards[:4]
                other_ranks = [r for r in rank_cards if r != rank]
                if len(other_ranks) >= 2:
                    for i in range(len(other_ranks)):
                        for j in range(i + 1, len(other_ranks)):
                            r1, r2 = other_ranks[i], other_ranks[j]
                            combinations.append(
                                four + [rank_cards[r1][0], rank_cards[r2][0]]
                            )

        return combinations

    def _is_consecutive_ranks(self, ranks: list[int]) -> bool:
        """Check if ranks are consecutive."""
        return all(ranks[i] - ranks[i - 1] == 1 for i in range(1, len(ranks)))

    def _get_observation(self, player_idx: int) -> np.ndarray:
        obs = np.zeros(self.OBSERVATION_SIZE, dtype=np.float32)

        if self._game_state is None:
            return obs

        idx = 0

        # Player's hand (54 binary values)
        player = self._game_state.get_player(player_idx)
        for card in player.hand:
            obs[card.card_id] = 1.0
        idx = 54

        # Last played cards (54 binary values)
        for card in self._game_state.last_play:
            obs[idx + card.card_id] = 1.0
        idx = 108

        # Game phase (3 values)
        obs[idx + self._game_state.phase] = 1.0
        idx += 3

        # Player role (2 values)
        obs[idx + player.role] = 1.0
        idx += 2

        # Landlord index (3 values)
        if self._game_state.landlord_idx >= 0:
            obs[idx + self._game_state.landlord_idx] = 1.0
        idx += 3

        # Hand counts (3 values, normalized)
        for i in range(GameConsts.NUM_PLAYERS):
            p = self._game_state.get_player(i)
            obs[idx + i] = p.hand_count / 20.0
        idx += 3

        # Bottom cards (if landlord, 54 binary values)
        if player_idx == self._game_state.landlord_idx:
            for card in self._game_state.bottom_cards:
                obs[idx + card.card_id] = 1.0
        idx += 54

        return obs

    def _get_global_state(self) -> dict[str, Any]:
        if self._game_state is None:
            return {}

        players_info = []
        for i in range(GameConsts.NUM_PLAYERS):
            p = self._game_state.get_player(i)
            players_info.append(
                {
                    "hand": [str(c) for c in p.hand],
                    "hand_count": p.hand_count,
                    "role": p.role.name,
                }
            )

        return {
            "current_player": self._current_player_idx,
            "num_players": self._num_players,
            "players": players_info,
            "phase": self._game_state.phase.name,
            "landlord_idx": self._game_state.landlord_idx,
            "last_play": [str(c) for c in self._game_state.last_play],
            "bottom_cards": [str(c) for c in self._game_state.bottom_cards],
        }

    def _get_action_mask(self, player_idx: int) -> list[int]:
        mask = [0] * self.MAX_ACTIONS

        if self._game_state is None:
            return mask

        # Rebuild action mapping for current player
        if player_idx == self._current_player_idx:
            self._build_action_mapping()

        phase = self._game_state.phase

        if phase == GamePhase.BIDDING:
            mask[0] = 1  # Pass (don't bid)
            mask[1] = 1  # Bid for landlord
            return mask

        if phase == GamePhase.PLAYING:
            last_info = self._game_state.last_play_info

            # Can always pass unless you played last
            if self._game_state.last_player_idx != player_idx:
                mask[0] = 1

            # Check each action
            for action_idx in range(
                1, min(len(self._action_mapping), self.MAX_ACTIONS)
            ):
                card_ids = self._action_mapping[action_idx]
                if not card_ids:
                    continue

                cards = [Card.from_id(cid) for cid in card_ids]
                hand_info = HandDetector.detect(cards)

                if hand_info.hand_type == HandType.INVALID:
                    continue

                # If no last play, any valid hand is OK; otherwise must beat
                if last_info is None or hand_info.can_beat(last_info):
                    mask[action_idx] = 1

        return mask

    def _apply_action(self, action: Any) -> tuple[float, bool]:
        if self._game_state is None:
            return 0.0, True

        phase = self._game_state.phase

        if phase == GamePhase.BIDDING:
            want_landlord = action == 1
            self._game_state.bid(want_landlord)
            self._current_player_idx = self._game_state.current_player_idx
            self._build_action_mapping()
            return 0.0, False

        if phase == GamePhase.PLAYING:
            if action == 0:
                # Pass
                self._game_state.play([])
            else:
                # Play cards
                if action < len(self._action_mapping):
                    card_ids = self._action_mapping[action]
                    cards = [Card.from_id(cid) for cid in card_ids]
                    self._game_state.play(cards)

            self._current_player_idx = self._game_state.current_player_idx
            self._build_action_mapping()

            terminated = self._game_state.is_terminated
            reward = 0.0

            if terminated:
                winner = self._game_state.get_winner()
                if winner == self._current_player_idx:
                    reward = 1.0
                elif winner is not None:
                    # Check if same team
                    winner_role = self._game_state.get_player(winner).role
                    current_role = self._game_state.get_player(
                        self._current_player_idx
                    ).role
                    reward = 1.0 if winner_role == current_role else -1.0

            return reward, terminated

        return 0.0, True

    def _render_text(self) -> str:
        if self._game_state is None:
            return "Game not initialized"

        lines = []
        lines.append("=== 斗地主 (Doudizhu) ===")
        lines.append(f"Phase: {self._game_state.phase.name}")
        lines.append(f"Current Player: {self._current_player_idx}")

        if self._game_state.landlord_idx >= 0:
            lines.append(f"Landlord: Player {self._game_state.landlord_idx}")
        lines.append("")

        lines.append("Players:")
        for i in range(GameConsts.NUM_PLAYERS):
            p = self._game_state.get_player(i)
            role = "地主" if p.role.value == 1 else "农民"
            marker = " <--" if i == self._current_player_idx else ""
            hand_str = " ".join(str(c) for c in p.hand)
            lines.append(f"  Player {i} ({role}): [{p.hand_count}] {hand_str}{marker}")

        if self._game_state.last_play:
            last_str = " ".join(str(c) for c in self._game_state.last_play)
            lines.append(
                f"\nLast Play (P{self._game_state.last_player_idx}): {last_str}"
            )

        if self._game_state.bottom_cards:
            bottom_str = " ".join(str(c) for c in self._game_state.bottom_cards)
            lines.append(f"Bottom Cards: {bottom_str}")

        return "\n".join(lines)
