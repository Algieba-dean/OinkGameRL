from __future__ import annotations

from games.scout.card.playable_checker import PlayableChecker
from games.scout.constants import BoardConsts
from games.scout.enums import ScoutFlip, ScoutPosition
from games.scout.game_status.game_state import GameState, GameStates


class Action:
    """Base class for all game actions."""

    def is_valid(self, game_state: GameStates | GameState) -> bool:
        raise NotImplementedError("is_valid is not supported on base action.")

    def execute(self, game_state: GameStates | GameState) -> None:
        raise NotImplementedError("execute is not supported on base action.")


class PlayAction(Action):
    """Action to play cards from hand to board."""

    def __init__(self, start_idx: int, end_idx: int) -> None:
        self.__start_idx = start_idx
        self.__end_idx = end_idx

    @property
    def start_idx(self) -> int:
        return self.__start_idx

    @property
    def end_idx(self) -> int:
        return self.__end_idx

    def is_valid(self, game_state: GameStates | GameState) -> bool:
        if not isinstance(game_state, GameState):
            raise NotImplementedError("PlayAction requires GameState")
        player = game_state.get_current_player()

        if self.__start_idx < 0 or self.__end_idx >= player.hand_count:
            return False
        if self.__start_idx > self.__end_idx:
            return False

        target_cards = list(player.hand[self.__start_idx : self.__end_idx + 1])
        board_cards = list(game_state.board.cards)

        if not board_cards:
            from games.scout.card.card_pattern_checker import CardPatternChecker
            from games.scout.enums import CardPattern

            pattern = CardPatternChecker.get_pattern(cards=target_cards)
            return pattern != CardPattern.INVALID_PATTERN

        return PlayableChecker.is_playable(
            board_cards=board_cards, target_cards=target_cards
        )

    def execute(self, game_state: GameStates | GameState) -> None:
        if not isinstance(game_state, GameState):
            raise NotImplementedError("PlayAction requires GameState")
        player = game_state.get_current_player()
        played_cards = player.play_cards(
            start_idx=self.__start_idx, end_idx=self.__end_idx
        )
        game_state.board.play_to_board(
            player_idx=game_state.current_player_idx, played_cards=played_cards
        )


class ScoutAction(Action):
    """Action to scout a card from the board."""

    def __init__(
        self, scout_position: ScoutPosition, insert_position: int, scout_flip: ScoutFlip
    ) -> None:
        self.__scout_position = scout_position
        self.__insert_position = insert_position
        self.__scout_flip = scout_flip

    @property
    def scout_position(self) -> ScoutPosition:
        return self.__scout_position

    @property
    def insert_position(self) -> int:
        return self.__insert_position

    @property
    def scout_flip(self) -> ScoutFlip:
        return self.__scout_flip

    def is_valid(self, game_state: GameStates | GameState) -> bool:
        if not isinstance(game_state, GameState):
            raise NotImplementedError("ScoutAction requires GameState")
        if game_state.board.owner_idx == BoardConsts.EMPTY_OWNER_ID:
            return False

        player = game_state.get_current_player()
        return not (
            self.__insert_position < 0 or self.__insert_position > player.hand_count
        )

    def execute(self, game_state: GameStates | GameState) -> None:
        if not isinstance(game_state, GameState):
            raise NotImplementedError("ScoutAction requires GameState")
        board_owner_idx = game_state.board.owner_idx

        scouted_card = game_state.board.scout_from_board(
            scout_position=self.__scout_position
        )

        if self.__scout_flip == ScoutFlip.YES:
            scouted_card.flip()

        player = game_state.get_current_player()
        player.insert_card(card=scouted_card, position=self.__insert_position)

        if board_owner_idx != BoardConsts.EMPTY_OWNER_ID:
            game_state.score.increase_score(player_idx=board_owner_idx, value=1)


class ScoutPlayAction(Action):
    """Action to scout a card and then play cards (requires token)."""

    def __init__(
        self,
        scout_position: ScoutPosition,
        insert_position: int,
        scout_flip: ScoutFlip,
        play_start_idx: int,
        play_end_idx: int,
    ) -> None:
        self.__scout_position = scout_position
        self.__insert_position = insert_position
        self.__scout_flip = scout_flip
        self.__play_start_idx = play_start_idx
        self.__play_end_idx = play_end_idx

    @property
    def scout_position(self) -> ScoutPosition:
        return self.__scout_position

    @property
    def insert_position(self) -> int:
        return self.__insert_position

    @property
    def scout_flip(self) -> ScoutFlip:
        return self.__scout_flip

    @property
    def play_start_idx(self) -> int:
        return self.__play_start_idx

    @property
    def play_end_idx(self) -> int:
        return self.__play_end_idx

    def is_valid(self, game_state: GameStates | GameState) -> bool:
        if not isinstance(game_state, GameState):
            raise NotImplementedError("ScoutPlayAction requires GameState")
        player = game_state.get_current_player()

        if not player.scout_and_show_token:
            return False

        scout_action = ScoutAction(
            scout_position=self.__scout_position,
            insert_position=self.__insert_position,
            scout_flip=self.__scout_flip,
        )
        return scout_action.is_valid(game_state=game_state)

    def execute(self, game_state: GameStates | GameState) -> None:
        if not isinstance(game_state, GameState):
            raise NotImplementedError("ScoutPlayAction requires GameState")
        player = game_state.get_current_player()
        player.use_scout_and_show_token()

        scout_action = ScoutAction(
            scout_position=self.__scout_position,
            insert_position=self.__insert_position,
            scout_flip=self.__scout_flip,
        )
        scout_action.execute(game_state=game_state)

        adjusted_start = self.__play_start_idx
        adjusted_end = self.__play_end_idx
        if self.__insert_position <= self.__play_start_idx:
            adjusted_start += 1
            adjusted_end += 1

        play_action = PlayAction(start_idx=adjusted_start, end_idx=adjusted_end)
        play_action.execute(game_state=game_state)
