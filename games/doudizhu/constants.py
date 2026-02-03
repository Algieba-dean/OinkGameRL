"""Constants for Doudizhu (斗地主) game."""


class GameConsts:
    """Game constants."""

    NUM_PLAYERS = 3
    TOTAL_CARDS = 54  # 52 + 2 jokers
    LANDLORD_CARDS = 20  # 地主手牌数
    PEASANT_CARDS = 17  # 农民手牌数
    BOTTOM_CARDS = 3  # 底牌数


class CardConsts:
    """Card constants."""

    NUM_SUITS = 4  # 不含王
    NUM_RANKS = 13  # 3-2 (不含王)
    NUM_JOKERS = 2
    MIN_STRAIGHT_LENGTH = 5  # 顺子最少5张
    MIN_STRAIGHT_PAIR_LENGTH = 3  # 连对最少3对
    MIN_AIRPLANE_LENGTH = 2  # 飞机最少2个三张
