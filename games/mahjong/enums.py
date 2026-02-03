"""Enums for Mahjong (麻将) game."""

from enum import IntEnum


class TileSuit(IntEnum):
    """Tile suits (花色)."""

    WAN = 0  # 万
    TIAO = 1  # 条
    TONG = 2  # 筒
    FENG = 3  # 风 (东南西北)
    JIAN = 4  # 箭 (中发白)


class TileRank(IntEnum):
    """Tile ranks within a suit."""

    # For numbered suits (万条筒): 1-9
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    # For 风: 东=1, 南=2, 西=3, 北=4
    EAST = 1
    SOUTH = 2
    WEST = 3
    NORTH = 4
    # For 箭: 中=1, 发=2, 白=3
    ZHONG = 1
    FA = 2
    BAI = 3


class GamePhase(IntEnum):
    """Game phases."""

    DRAWING = 0  # 摸牌阶段
    DISCARDING = 1  # 出牌阶段
    WAITING_RESPONSE = 2  # 等待其他玩家响应 (吃碰杠胡)
    FINISHED = 3  # 游戏结束


class ActionType(IntEnum):
    """Types of actions."""

    DRAW = 0  # 摸牌
    DISCARD = 1  # 出牌
    CHI = 2  # 吃
    PONG = 3  # 碰
    GANG = 4  # 杠
    HU = 5  # 胡
    PASS = 6  # 过


class MeldType(IntEnum):
    """Types of melds (面子)."""

    CHI = 0  # 吃 (顺子)
    PONG = 1  # 碰 (刻子)
    MING_GANG = 2  # 明杠
    AN_GANG = 3  # 暗杠
