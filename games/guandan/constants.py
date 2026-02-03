"""Constants for Guandan (掼蛋) game."""


class GameConsts:
    """Game constants."""

    NUM_PLAYERS = 4
    NUM_DECKS = 2
    TOTAL_CARDS = 108  # 54 * 2
    CARDS_PER_PLAYER = 27  # 108 / 4


class CardConsts:
    """Card constants."""

    NUM_SUITS = 4  # 不含王
    NUM_RANKS = 13  # 2-A (不含王)
    NUM_JOKERS_PER_DECK = 2
    MIN_STRAIGHT_LENGTH = 5  # 顺子最少5张
    MIN_TUBE_LENGTH = 3  # 钢板最少3对
    MIN_PLATE_LENGTH = 2  # 板子最少2个三张


class RankConsts:
    """Rank progression constants."""

    STARTING_RANK = 2  # 从2开始打
    WINNING_RANK = 14  # 打到A结束 (ACE = 12, but we use 14 for "beyond A")
