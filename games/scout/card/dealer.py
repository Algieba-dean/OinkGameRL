import random
from copy import deepcopy
from queue import Queue

from games.scout.card.card_data import CardData
from games.scout.card.cards import Card
from games.scout.constants import PlayerConsts


class Dealer:
    def __init__(self) -> None:
        self.__card_data: CardData = CardData()
        self.__current_queue_dict: dict[int, Queue[Card]] = {}
        for player_num in PlayerConsts.PLAYER_CARD_NUM:
            self.__current_queue_dict[player_num] = Queue()
        self.__initailize_card_queue_dict()

    def dispatch_cards(self, player_num: int) -> list[list[Card]]:
        if player_num not in PlayerConsts.ALLOWED_PLAYER_NUM:
            raise ValueError(f"unexpected player number:{player_num} for card dispatch")
        result = []
        for _ in range(player_num):
            cards = self.__get_cards_from_current_cards(player_num=player_num)
            result.append(cards)
        return result

    def __get_cards_from_current_cards(self, player_num: int) -> list[Card]:
        expected_card_num = PlayerConsts.PLAYER_CARD_NUM[player_num]
        cards = []
        for _ in range(expected_card_num):
            if self.__current_queue_dict[player_num].empty():
                # reload the queue again
                self.__initailize_card_queue_dict(target_players=[player_num])
            cards.append(self.__current_queue_dict[player_num].get())
        return cards

    def __initailize_card_queue_dict(
        self, target_players: list[int] = PlayerConsts.ALLOWED_PLAYER_NUM
    ) -> None:
        for player_num in target_players:
            # initialize queue dict
            self.__current_queue_dict[player_num] = Queue()

            # get corresponding cards
            cards = deepcopy(
                self.__card_data.get_cards_for_player(player_num=player_num)
            )

            # random flip cards
            self.__random_flip_all_cards(current_cards=cards)

            # load into queue
            for card in cards:
                self.__current_queue_dict[player_num].put(card)

    def __random_flip_all_cards(self, current_cards: list[Card]):
        for card in current_cards:
            if random.random() > 0.5:
                card.flip()
