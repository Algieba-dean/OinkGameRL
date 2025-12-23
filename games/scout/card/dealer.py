from collections import deque
from copy import deepcopy

import numpy as np

from games.scout.card.card_data import CardData
from games.scout.card.cards import Card
from games.scout.constants import PlayerConsts


class Dealer:
    """
    The Dealer will do the card dispatching. We don't need to take care about how to shuffle or random flip for initialized cards.
    Both shuffle and random will be done with according to random generation in silence.
    """

    def __init__(self, random_generator: np.random.Generator) -> None:
        self.__random_generator: np.random.Generator = random_generator
        self.__card_data: CardData = CardData()
        self.__current_queue_dict: dict[int, deque[Card]] = {}
        for player_num in PlayerConsts.PLAYER_CARD_NUM:
            self.__current_queue_dict[player_num] = deque()
        self.__initailize_card_queue_dict()

    def dispatch_cards(self, player_num: int) -> list[list[Card]]:
        if player_num not in PlayerConsts.ALLOWED_PLAYER_NUM:
            raise ValueError(f"unexpected player number:{player_num} for card dispatch")
        result = []
        for _ in range(player_num):
            cards = self.__get_cards_from_current_cards(player_num=player_num)
            result.append(cards)
        return result

    def reset(self, random_generator: np.random.Generator) -> None:
        """when reset in env is called, the random generator might be udpated

        Args:
            random_generator (np.random.Generator): the random generator
        """
        self.__random_generator = random_generator
        self.__initailize_card_queue_dict()

    def __get_cards_from_current_cards(self, player_num: int) -> list[Card]:
        expected_card_num = PlayerConsts.PLAYER_CARD_NUM[player_num]
        cards = []
        for _ in range(expected_card_num):
            if not self.__current_queue_dict[player_num]:
                # reload the queue again
                self.__initailize_card_queue_dict(target_players=[player_num])
            cards.append(self.__current_queue_dict[player_num].popleft())
        return cards

    def __initailize_card_queue_dict(
        self, target_players: list[int] = PlayerConsts.ALLOWED_PLAYER_NUM
    ) -> None:
        """initialize the card , will be called on initializer and also when card is empty in dispatching. It will initialize the card dict, andd shuffle/flip all the cards, then load them into queue

        Args:
            target_players (list[int], optional): we can limit the target_players here, in default it will initial for all supported players so that if in playing the player num updated it can also work.
        """
        for player_num in target_players:
            # initialize queue dict
            self.__current_queue_dict[player_num] = deque()

            # get corresponding cards
            cards = deepcopy(
                self.__card_data.get_cards_for_player(player_num=player_num)
            )

            # random shuffle cards
            self.__random_shuffle_all_cards(current_cards=cards)
            # random flip cards
            self.__random_flip_all_cards(current_cards=cards)

            # load into queue
            for card in cards:
                self.__current_queue_dict[player_num].append(card)

    def __random_shuffle_all_cards(self, current_cards: list[Card]) -> None:
        self.__random_generator.shuffle(current_cards)

    def __random_flip_all_cards(self, current_cards: list[Card]) -> None:
        for card in current_cards:
            if self.__random_generator.random() > 0.5:
                card.flip()
