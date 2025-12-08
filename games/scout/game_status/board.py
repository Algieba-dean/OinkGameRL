from games.scout.card.cards import Card
from games.scout.constants import BoardConsts
from games.scout.enums import ScoutPosition


class Board:
    def __init__(self) -> None:
        self.__owner_idx: int = BoardConsts.EMPTY_OWNER_ID
        self.__cards: list[Card] = []

    @property
    def owner_idx(self) -> int:
        return self.__owner_idx

    @property
    def cards(self) -> tuple[Card, ...]:
        return tuple(self.__cards)

    def play_to_board(self, player_idx: int, played_cards: list[Card]) -> None:
        """play cards to board
        PS:
        1. board won't check if it's playable, please make sure playable is checked outside.
        2. board won't validate player_idx

        Args:
            player_idx (int):
            played_cards (list[Card]):
        """
        self.__owner_idx = player_idx
        self.__cards = played_cards

    def scout_from_board(self, scout_position: ScoutPosition) -> Card:
        """scout one card from the board
        PS:
        1. board will only return scoutted card, please make sure getting owner_idx before calling, and don't forget to increase score and apply flip action outsdie


        Args:
            scout_position (ScoutPosition): LEFT | RIGHT, only the most left|right one can be scouted

        """
        if not self.__cards:
            raise ValueError("can't scout from empty board")
        target_card: Card
        if scout_position == ScoutPosition.LEFT:
            target_card = self.__cards[0]
            self.__cards = self.__cards[1:]
        else:
            target_card = self.__cards[-1]
            self.__cards = self.__cards[:-1]
        if not self.__cards:
            self.__owner_idx = BoardConsts.EMPTY_OWNER_ID
        return target_card
