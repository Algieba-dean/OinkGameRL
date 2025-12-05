import random
from copy import deepcopy
from queue import Queue

from games.scout.card.CardData import CardData
from games.scout.card.Cards import Card
from games.scout.Constants import PlayerConsts


class Dealer:
    def __init__(self):
        self.__card_data: CardData = CardData()
        self.__current_cards: Queue[Card] = self.__initailize_and_get_cards()

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
            cards.append(self.__current_cards.get())
        return cards

    def __initailize_and_get_cards(self) -> Queue[Card]:
        cards = deepcopy(self.__card_data.cards)
        current_queue = Queue()
        self.__random_flip_all_cards(current_cards=cards)
        for card in cards:
            current_queue.put(card)

        return current_queue

    def __random_flip_all_cards(self, current_cards: list[Card]):
        for card in current_cards:
            if random.random() > 0.5:
                card.flip()
