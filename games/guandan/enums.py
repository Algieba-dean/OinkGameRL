"""Enums for Guandan (掼蛋) game."""

from enum import IntEnum


class CardSuit(IntEnum):
    """Card suits (花色)."""

    SPADE = 0  # 黑桃
    HEART = 1  # 红桃
    CLUB = 2  # 梅花
    DIAMOND = 3  # 方块
    JOKER = 4  # 王


class CardRank(IntEnum):
    """Card ranks (点数), ordered by Guandan rules (2 is lowest regular)."""

    TWO = 0
    THREE = 1
    FOUR = 2
    FIVE = 3
    SIX = 4
    SEVEN = 5
    EIGHT = 6
    NINE = 7
    TEN = 8
    JACK = 9
    QUEEN = 10
    KING = 11
    ACE = 12
    BLACK_JOKER = 13  # 小王
    RED_JOKER = 14  # 大王


class GamePhase(IntEnum):
    """Game phases."""

    PLAYING = 0  # 出牌阶段
    FINISHED = 1  # 游戏结束


class HandType(IntEnum):
    """Types of card combinations (牌型)."""

    PASS = 0  # 不出/过
    SINGLE = 1  # 单张
    PAIR = 2  # 对子
    TRIPLE = 3  # 三张 (三不带)
    TRIPLE_WITH_TWO = 4  # 三带二
    STRAIGHT = 5  # 顺子 (至少5张连续单牌)
    STRAIGHT_FLUSH = 6  # 同花顺
    TUBE = 7  # 钢板/连对 (至少3对连续对子)
    PLATE = 8  # 板子/三连 (至少2个连续三张)
    BOMB_4 = 9  # 炸弹 (4张相同)
    BOMB_5 = 10  # 5张炸弹
    BOMB_6 = 11  # 6张炸弹
    BOMB_7 = 12  # 7张炸弹
    BOMB_8 = 13  # 8张炸弹
    ROCKET = 14  # 天王炸 (四个王)
    INVALID = 15  # 无效牌型


class Team(IntEnum):
    """Team assignment."""

    TEAM_A = 0  # 玩家0和2
    TEAM_B = 1  # 玩家1和3
