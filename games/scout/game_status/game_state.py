from games.scout.card.cards import Card
from games.scout.constants import PlayerConsts
from games.scout.game_status.board import Board
from games.scout.game_status.score import Score
from games.scout.player.player import Player


class GameStates:
    """Legacy class for backward compatibility."""

    def __init__(self) -> None:
        pass


class GameState:
    """Manages the complete state of a Scout game.

    Includes players, board, scores, and turn management.
    """

    def __init__(self, player_num: int, player_cards: list[list[Card]]) -> None:
        if player_num not in PlayerConsts.ALLOWED_PLAYER_NUM:
            raise ValueError(
                f"Invalid player num {player_num}, "
                f"allowed: {PlayerConsts.ALLOWED_PLAYER_NUM}"
            )
        if len(player_cards) != player_num:
            raise ValueError(
                f"Expected {player_num} sets of cards, got {len(player_cards)}"
            )

        self.__player_num: int = player_num
        self.__players: list[Player] = [
            Player(player_idx=i, cards=player_cards[i]) for i in range(player_num)
        ]
        self.__current_player_idx: int = 0
        self.__board: Board = Board()
        self.__score: Score = Score(player_num=player_num)

    @property
    def player_num(self) -> int:
        return self.__player_num

    @property
    def players(self) -> tuple[Player, ...]:
        return tuple(self.__players)

    @property
    def current_player_idx(self) -> int:
        return self.__current_player_idx

    @property
    def board(self) -> Board:
        return self.__board

    @property
    def score(self) -> Score:
        return self.__score

    @property
    def is_terminated(self) -> bool:
        """Check if game is over (any player has empty hand)."""
        return any(player.hand_count == 0 for player in self.__players)

    def get_player(self, player_idx: int) -> Player:
        """Get player by index."""
        return self.__players[player_idx]

    def get_current_player(self) -> Player:
        """Get the current active player."""
        return self.__players[self.__current_player_idx]

    def next_player(self) -> None:
        """Advance to the next player."""
        self.__current_player_idx = (self.__current_player_idx + 1) % self.__player_num

    def reset(self, player_cards: list[list[Card]]) -> None:
        """Reset game state with new cards.

        Args:
            player_cards: New cards for each player
        """
        for i, player in enumerate(self.__players):
            player.reset(cards=player_cards[i])
        self.__current_player_idx = 0
        self.__board = Board()
        self.__score.clean_all_score(player_num=self.__player_num)
