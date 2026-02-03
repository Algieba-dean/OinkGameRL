"""Guandan (掼蛋) game environment."""

from typing import Any

import numpy as np
from gymnasium import spaces

from games.board_game import BoardGameEnv
from games.guandan.card import Card
from games.guandan.constants import GameConsts
from games.guandan.enums import CardRank, GamePhase, HandType, Team
from games.guandan.game_state import GameState
from games.guandan.hand_detector import HandDetector
from games.registry import register_game


@register_game("guandan")
class GuandanGameEnv(BoardGameEnv):
    """Guandan (掼蛋) card game environment.

    A popular Chinese card game for 4 players in 2 teams.
    Uses two decks (108 cards), 27 cards per player.
    """

    OBSERVATION_SIZE = 450
    MAX_ACTIONS = 1000

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

        player = self._game_state.get_player(self._current_player_idx)
        if player.finished:
            return

        hand = player.hand
        combinations = self._generate_all_combinations(hand)

        for combo in combinations:
            card_ids = [card.card_id for card in combo]
            self._action_mapping.append(card_ids)

    def _generate_all_combinations(self, hand: list[Card]) -> list[list[Card]]:
        """Generate all valid card combinations from hand."""
        combinations: list[list[Card]] = []
        level_rank = self._game_state.level_rank if self._game_state else CardRank.TWO

        # Group cards by effective rank
        rank_cards: dict[int, list[Card]] = {}
        for card in hand:
            rank = card.get_effective_rank(level_rank)
            if rank not in rank_cards:
                rank_cards[rank] = []
            rank_cards[rank].append(card)

        # Singles
        for card in hand:
            combinations.append([card])

        # Pairs, Triples, and Bombs
        for _, cards in rank_cards.items():
            if len(cards) >= 2:
                combinations.append(cards[:2])
            if len(cards) >= 3:
                combinations.append(cards[:3])
            if len(cards) >= 4:
                combinations.append(cards[:4])  # Bomb 4
            if len(cards) >= 5:
                combinations.append(cards[:5])  # Bomb 5
            if len(cards) >= 6:
                combinations.append(cards[:6])  # Bomb 6
            if len(cards) >= 7:
                combinations.append(cards[:7])  # Bomb 7
            if len(cards) >= 8:
                combinations.append(cards[:8])  # Bomb 8

        # Triple with two
        for rank, cards in rank_cards.items():
            if len(cards) >= 3:
                triple = cards[:3]
                for other_rank, other_cards in rank_cards.items():
                    if other_rank != rank and len(other_cards) >= 2:
                        combinations.append(triple + other_cards[:2])

        # Straights (5+ consecutive singles)
        sorted_ranks = sorted([r for r in rank_cards if r < 98])
        for length in range(5, min(13, len(sorted_ranks) + 1)):
            for start_idx in range(len(sorted_ranks) - length + 1):
                ranks_slice = sorted_ranks[start_idx : start_idx + length]
                if self._is_consecutive(ranks_slice):
                    straight = [rank_cards[r][0] for r in ranks_slice]
                    combinations.append(straight)

        # Tubes (3+ consecutive pairs)
        pair_ranks = [
            r for r, cards in rank_cards.items() if len(cards) >= 2 and r < 98
        ]
        pair_ranks.sort()
        for length in range(3, min(11, len(pair_ranks) + 1)):
            for start_idx in range(len(pair_ranks) - length + 1):
                ranks_slice = pair_ranks[start_idx : start_idx + length]
                if self._is_consecutive(ranks_slice):
                    tube = []
                    for r in ranks_slice:
                        tube.extend(rank_cards[r][:2])
                    combinations.append(tube)

        # Plates (2+ consecutive triples)
        triple_ranks = [
            r for r, cards in rank_cards.items() if len(cards) >= 3 and r < 98
        ]
        triple_ranks.sort()
        for length in range(2, min(6, len(triple_ranks) + 1)):
            for start_idx in range(len(triple_ranks) - length + 1):
                ranks_slice = triple_ranks[start_idx : start_idx + length]
                if self._is_consecutive(ranks_slice):
                    plate = []
                    for r in ranks_slice:
                        plate.extend(rank_cards[r][:3])
                    combinations.append(plate)

        # Rocket (4 jokers)
        jokers = [
            c for c in hand if c.rank in (CardRank.BLACK_JOKER, CardRank.RED_JOKER)
        ]
        if len(jokers) == 4:
            combinations.append(jokers)

        return combinations

    def _is_consecutive(self, ranks: list[int]) -> bool:
        """Check if ranks are consecutive."""
        if not ranks:
            return False
        return all(ranks[i] - ranks[i - 1] == 1 for i in range(1, len(ranks)))

    def _get_observation(self, player_idx: int) -> np.ndarray:
        obs = np.zeros(self.OBSERVATION_SIZE, dtype=np.float32)

        if self._game_state is None:
            return obs

        idx = 0

        # Player's hand (108 binary values)
        player = self._game_state.get_player(player_idx)
        for card in player.hand:
            obs[card.card_id] = 1.0
        idx = 108

        # Last played cards (108 binary values)
        for card in self._game_state.last_play:
            obs[idx + card.card_id] = 1.0
        idx = 216

        # Game phase (2 values)
        obs[idx + self._game_state.phase] = 1.0
        idx += 2

        # Level rank (13 values, one-hot)
        obs[idx + self._game_state.level_rank] = 1.0
        idx += 13

        # Player's team (2 values)
        obs[idx + player.team] = 1.0
        idx += 2

        # Hand counts for all players (4 values, normalized)
        for i in range(GameConsts.NUM_PLAYERS):
            p = self._game_state.get_player(i)
            obs[idx + i] = p.hand_count / 27.0
        idx += 4

        # Finished status for all players (4 values)
        for i in range(GameConsts.NUM_PLAYERS):
            p = self._game_state.get_player(i)
            obs[idx + i] = 1.0 if p.finished else 0.0
        idx += 4

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
                    "team": p.team.name,
                    "finished": p.finished,
                    "finish_order": p.finish_order,
                }
            )

        return {
            "current_player": self._current_player_idx,
            "num_players": self._num_players,
            "players": players_info,
            "phase": self._game_state.phase.name,
            "level_rank": self._game_state.level_rank.name,
            "last_play": [str(c) for c in self._game_state.last_play],
        }

    def _get_action_mask(self, player_idx: int) -> list[int]:
        mask = [0] * self.MAX_ACTIONS

        if self._game_state is None:
            return mask

        player = self._game_state.get_player(player_idx)
        if player.finished:
            return mask

        # Rebuild action mapping for current player
        if player_idx == self._current_player_idx:
            self._build_action_mapping()

        phase = self._game_state.phase
        if phase != GamePhase.PLAYING:
            return mask

        last_info = self._game_state.last_play_info

        # Can pass unless you played last
        if self._game_state.last_player_idx != player_idx:
            mask[0] = 1

        # Check each action
        level_rank = self._game_state.level_rank
        for action_idx in range(1, min(len(self._action_mapping), self.MAX_ACTIONS)):
            card_ids = self._action_mapping[action_idx]
            if not card_ids:
                continue

            cards = [Card.from_id(cid) for cid in card_ids]
            hand_info = HandDetector.detect(cards, level_rank)

            if hand_info.hand_type == HandType.INVALID:
                continue

            # If no last play, any valid hand is OK; otherwise must beat
            if last_info is None or hand_info.can_beat(last_info):
                mask[action_idx] = 1

        return mask

    def _apply_action(self, action: Any) -> tuple[float, bool]:
        if self._game_state is None:
            return 0.0, True

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
            winner_team = self._game_state.get_winner_team()
            player_team = self._game_state.get_player(self._current_player_idx).team
            if winner_team == player_team:
                reward = float(self._game_state.get_team_score(player_team))
            elif winner_team is not None:
                reward = -float(self._game_state.get_team_score(winner_team))

        return reward, terminated

    def _render_text(self) -> str:
        if self._game_state is None:
            return "Game not initialized"

        lines = []
        lines.append("=== 掼蛋 (Guandan) ===")
        lines.append(f"Phase: {self._game_state.phase.name}")
        lines.append(f"Level Rank: {self._game_state.level_rank.name}")
        lines.append(f"Current Player: {self._current_player_idx}")
        lines.append("")

        lines.append("Players:")
        for i in range(GameConsts.NUM_PLAYERS):
            p = self._game_state.get_player(i)
            team = "A" if p.team == Team.TEAM_A else "B"
            status = f"(#{p.finish_order})" if p.finished else ""
            marker = " <--" if i == self._current_player_idx else ""
            hand_str = " ".join(str(c) for c in p.hand[:10])
            if p.hand_count > 10:
                hand_str += f" ... (+{p.hand_count - 10})"
            lines.append(
                f"  P{i} [Team {team}] {status}: [{p.hand_count}] {hand_str}{marker}"
            )

        if self._game_state.last_play:
            last_str = " ".join(str(c) for c in self._game_state.last_play)
            lines.append(
                f"\nLast Play (P{self._game_state.last_player_idx}): {last_str}"
            )

        return "\n".join(lines)
