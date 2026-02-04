from typing import Any

import numpy as np
from gymnasium import spaces

from games.board_game import BoardGameEnv
from games.registry import register_game
from games.scout.card.dealer import Dealer
from games.scout.card.playable_checker import PlayableChecker
from games.scout.constants import BoardConsts, PlayerConsts
from games.scout.enums import ScoutFlip, ScoutPosition
from games.scout.game_status.game_state import GameState
from games.scout.player.action import PlayAction, ScoutAction, ScoutPlayAction


@register_game("scout")
class ScoutGameEnv(BoardGameEnv):
    """Scout card game environment implementing the OinkGameEnv interface.

    Action space encoding:
    - Play actions: cards can be played as consecutive subsets from hand
    - Scout actions: take card from board (left/right), insert at position, flip or not
    - ScoutPlay actions: scout then play (requires token)
    """

    MAX_HAND_SIZE = 12
    MAX_CARD_VALUE = 10
    OBSERVATION_SIZE = 200

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
        self._dealer: Dealer = Dealer(random_generator=self._rng)
        self._game_state: GameState | None = None

        self._action_mapping: list[dict[str, Any]] = []
        self._build_action_mapping()

        self.action_space: spaces.Discrete = spaces.Discrete(len(self._action_mapping))
        self.observation_space: spaces.Box = spaces.Box(
            low=0, high=1, shape=(self.OBSERVATION_SIZE,), dtype=np.float32
        )

    def _build_action_mapping(self) -> None:
        """Build mapping from discrete action index to game actions."""
        self._action_mapping = []

        hand_size = PlayerConsts.PLAYER_CARD_NUM[self._num_players]

        for start in range(hand_size):
            for end in range(start, hand_size):
                self._action_mapping.append(
                    {"type": "play", "start_idx": start, "end_idx": end}
                )

        for scout_pos in [ScoutPosition.LEFT, ScoutPosition.RIGHT]:
            for insert_pos in range(hand_size + 1):
                for flip in [ScoutFlip.NO, ScoutFlip.YES]:
                    self._action_mapping.append(
                        {
                            "type": "scout",
                            "scout_position": scout_pos,
                            "insert_position": insert_pos,
                            "scout_flip": flip,
                        }
                    )

        for scout_pos in [ScoutPosition.LEFT, ScoutPosition.RIGHT]:
            for insert_pos in range(hand_size + 1):
                for flip in [ScoutFlip.NO, ScoutFlip.YES]:
                    for play_start in range(hand_size + 1):
                        for play_end in range(play_start, hand_size + 1):
                            self._action_mapping.append(
                                {
                                    "type": "scout_play",
                                    "scout_position": scout_pos,
                                    "insert_position": insert_pos,
                                    "scout_flip": flip,
                                    "play_start_idx": play_start,
                                    "play_end_idx": play_end,
                                }
                            )

    def _reset_logic(
        self, seed: int | None, options: dict[str, Any] | None = None
    ) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        else:
            self._rng = np.random.default_rng()

        self._dealer.reset(random_generator=self._rng)
        player_cards = self._dealer.dispatch_cards(player_num=self._num_players)

        self._game_state = GameState(
            player_num=self._num_players, player_cards=player_cards
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
                obs[idx] = card.top / self.MAX_CARD_VALUE
                obs[idx + 1] = card.bottom / self.MAX_CARD_VALUE
                idx += 2

        idx = 50

        board_cards = self._game_state.board.cards
        for card in board_cards:
            if idx + 2 < self.OBSERVATION_SIZE:
                obs[idx] = card.top / self.MAX_CARD_VALUE
                obs[idx + 1] = card.bottom / self.MAX_CARD_VALUE
                idx += 2

        idx = 100
        owner_idx = self._game_state.board.owner_idx
        if owner_idx == BoardConsts.EMPTY_OWNER_ID:
            obs[idx] = 0.0
        else:
            obs[idx] = (owner_idx + 1) / (self._num_players + 1)
        idx += 1

        for i in range(self._num_players):
            if idx < self.OBSERVATION_SIZE:
                obs[idx] = self._game_state.score.score_dict[i] / 50.0
                idx += 1

        idx = 110
        obs[idx] = 1.0 if player.scout_and_show_token else 0.0
        idx += 1

        for i in range(self._num_players):
            p = self._game_state.get_player(i)
            if idx < self.OBSERVATION_SIZE:
                obs[idx] = p.hand_count / self.MAX_HAND_SIZE
                idx += 1

        return obs

    def _get_global_state(self) -> dict[str, Any]:
        if self._game_state is None:
            return {}

        players_hands = []
        for i in range(self._num_players):
            player = self._game_state.get_player(i)
            hand_repr = [
                {"top": c.top, "bottom": c.bottom, "idx": c.idx} for c in player.hand
            ]
            players_hands.append(hand_repr)

        board_repr = [
            {"top": c.top, "bottom": c.bottom, "idx": c.idx}
            for c in self._game_state.board.cards
        ]

        return {
            "current_player": self._current_player_idx,
            "num_players": self._num_players,
            "hands": players_hands,
            "board": board_repr,
            "board_owner": self._game_state.board.owner_idx,
            "scores": dict(self._game_state.score.score_dict),
            "tokens": [
                self._game_state.get_player(i).scout_and_show_token
                for i in range(self._num_players)
            ],
        }

    def _get_action_mask(self, player_idx: int) -> list[int]:
        """Get valid action mask with optimized filtering.

        Optimization strategy:
        - Pre-check global conditions (board state, token) before iterating
        - Skip entire action type categories when not applicable
        - Only check play validity for hand indices that exist
        """
        mask = [0] * len(self._action_mapping)

        if self._game_state is None:
            return mask

        player = self._game_state.get_player(player_idx)
        board_cards = list(self._game_state.board.cards)
        hand = list(player.hand)
        hand_count = player.hand_count

        # Pre-compute conditions
        board_empty = self._game_state.board.owner_idx == BoardConsts.EMPTY_OWNER_ID
        has_token = player.scout_and_show_token

        # Import once outside loop
        from games.scout.card.card_pattern_checker import CardPatternChecker
        from games.scout.enums import CardPattern

        # Build index ranges for each action type for faster lookup
        play_end_idx = 0
        scout_start_idx = 0
        scout_play_start_idx = 0

        for i, action_def in enumerate(self._action_mapping):
            if action_def["type"] == "play":
                play_end_idx = i + 1
            elif action_def["type"] == "scout" and scout_start_idx == 0:
                scout_start_idx = i
            elif action_def["type"] == "scout_play" and scout_play_start_idx == 0:
                scout_play_start_idx = i
                break

        # Process play actions (only check valid hand indices)
        for action_idx in range(play_end_idx):
            action_def = self._action_mapping[action_idx]
            start_idx = action_def["start_idx"]
            end_idx = action_def["end_idx"]

            if start_idx >= hand_count or end_idx >= hand_count:
                continue

            target_cards = hand[start_idx : end_idx + 1]

            if board_empty:
                pattern = CardPatternChecker.get_pattern(cards=target_cards)
                if pattern != CardPattern.INVALID_PATTERN:
                    mask[action_idx] = 1
            else:
                if PlayableChecker.is_playable(
                    board_cards=board_cards, target_cards=target_cards
                ):
                    mask[action_idx] = 1

        # Process scout actions (skip if board is empty)
        if not board_empty:
            for action_idx in range(scout_start_idx, scout_play_start_idx):
                action_def = self._action_mapping[action_idx]
                if action_def["type"] != "scout":
                    continue
                insert_pos = action_def["insert_position"]
                if insert_pos <= hand_count:
                    mask[action_idx] = 1

        # Process scout_play actions (skip if no token or board empty)
        if has_token and not board_empty:
            new_hand_count = hand_count + 1
            for action_idx in range(scout_play_start_idx, len(self._action_mapping)):
                action_def = self._action_mapping[action_idx]
                insert_pos = action_def["insert_position"]
                play_start = action_def["play_start_idx"]
                play_end = action_def["play_end_idx"]

                if insert_pos > hand_count:
                    continue

                adj_start = play_start + 1 if insert_pos <= play_start else play_start
                adj_end = play_end + 1 if insert_pos <= play_start else play_end

                if adj_start < new_hand_count and adj_end < new_hand_count:
                    mask[action_idx] = 1

        return mask

    def _apply_action(self, action: Any) -> tuple[float, bool]:
        if self._game_state is None:
            return 0.0, True

        action_def = self._action_mapping[action]
        action_type = action_def["type"]

        if action_type == "play":
            play_action = PlayAction(
                start_idx=action_def["start_idx"], end_idx=action_def["end_idx"]
            )
            play_action.execute(game_state=self._game_state)
        elif action_type == "scout":
            scout_action = ScoutAction(
                scout_position=action_def["scout_position"],
                insert_position=action_def["insert_position"],
                scout_flip=action_def["scout_flip"],
            )
            scout_action.execute(game_state=self._game_state)
        else:
            scout_play_action = ScoutPlayAction(
                scout_position=action_def["scout_position"],
                insert_position=action_def["insert_position"],
                scout_flip=action_def["scout_flip"],
                play_start_idx=action_def["play_start_idx"],
                play_end_idx=action_def["play_end_idx"],
            )
            scout_play_action.execute(game_state=self._game_state)

        terminated = self._game_state.is_terminated

        reward = 0.0
        if terminated:
            current_score = self._game_state.score.score_dict[self._current_player_idx]
            current_hand = self._game_state.get_player(
                self._current_player_idx
            ).hand_count
            reward = float(current_score - current_hand)

        if not terminated:
            self._game_state.next_player()
            self._current_player_idx = self._game_state.current_player_idx

        return reward, terminated

    def _render_text(self) -> str:
        if self._game_state is None:
            return "Game not initialized"

        lines = []
        lines.append(f"=== Scout Game (Players: {self._num_players}) ===")
        lines.append(f"Current Player: {self._current_player_idx}")
        lines.append("")

        lines.append("Board:")
        if self._game_state.board.cards:
            cards_str = " ".join(str(c) for c in self._game_state.board.cards)
            lines.append(f"  Cards: {cards_str}")
            lines.append(f"  Owner: Player {self._game_state.board.owner_idx}")
        else:
            lines.append("  (empty)")
        lines.append("")

        lines.append("Scores:")
        for i in range(self._num_players):
            score = self._game_state.score.score_dict[i]
            token = (
                "[T]" if self._game_state.get_player(i).scout_and_show_token else "[ ]"
            )
            lines.append(f"  Player {i}: {score} points {token}")
        lines.append("")

        lines.append("Hands:")
        for i in range(self._num_players):
            player = self._game_state.get_player(i)
            if i == self._current_player_idx:
                cards_str = " ".join(str(c) for c in player.hand)
                lines.append(f"  Player {i} (current): {cards_str}")
            else:
                lines.append(f"  Player {i}: [{player.hand_count} cards]")

        return "\n".join(lines)
