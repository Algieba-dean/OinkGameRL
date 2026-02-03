"""Enums for Doudizhu (斗地主) game."""

from enum import IntEnum


class CardSuit(IntEnum):
    """Card suits (花色)."""

    SPADE = 0  # 黑桃
    HEART = 1  # 红桃
    CLUB = 2  # 梅花
    DIAMOND = 3  # 方块
    JOKER = 4  # 王


class CardRank(IntEnum):
    """Card ranks (点数), ordered by Doudizhu rules."""

    THREE = 0
    FOUR = 1
    FIVE = 2
    SIX = 3
    SEVEN = 4
    EIGHT = 5
    NINE = 6
    TEN = 7
    JACK = 8
    QUEEN = 9
    KING = 10
    ACE = 11
    TWO = 12
    BLACK_JOKER = 13  # 小王
    RED_JOKER = 14  # 大王


class PlayerRole(IntEnum):
    """Player roles in Doudizhu."""

    PEASANT = 0  # 农民
    LANDLORD = 1  # 地主


class GamePhase(IntEnum):
    """Game phases."""

    BIDDING = 0  # 叫地主阶段
    PLAYING = 1  # 出牌阶段
    FINISHED = 2  # 游戏结束


class HandType(IntEnum):
    """Types of card combinations (牌型)."""

    PASS = 0  # 不出/过
    SINGLE = 1  # 单张
    PAIR = 2  # 对子
    TRIPLE = 3  # 三张
    TRIPLE_WITH_SINGLE = 4  # 三带一
    TRIPLE_WITH_PAIR = 5  # 三带二
    STRAIGHT = 6  # 顺子 (至少5张连续单牌)
    STRAIGHT_PAIR = 7  # 连对 (至少3对连续对子)
    AIRPLANE = 8  # 飞机 (至少2个连续三张)
    AIRPLANE_WITH_SINGLES = 9  # 飞机带单
    AIRPLANE_WITH_PAIRS = 10  # 飞机带对
    FOUR_WITH_TWO_SINGLES = 11  # 四带二单
    FOUR_WITH_TWO_PAIRS = 12  # 四带二对
    BOMB = 13  # 炸弹 (四张相同)
    ROCKET = 14  # 火箭 (双王)
    INVALID = 15  # 无效牌型
