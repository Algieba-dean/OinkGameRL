"""Constants for Mahjong (麻将) game."""


class GameConsts:
    """Game constants."""

    NUM_PLAYERS = 4
    TOTAL_TILES = 136  # 34 types * 4 copies
    HAND_SIZE = 13  # Initial hand size
    WINNING_HAND_SIZE = 14  # 4 melds + 1 pair


class TileConsts:
    """Tile constants."""

    NUM_NUMBERED_SUITS = 3  # 万条筒
    NUM_HONOR_SUITS = 2  # 风箭
    TILES_PER_RANK = 4  # 每种牌4张
    NUM_NUMBERED_RANKS = 9  # 1-9
    NUM_FENG_RANKS = 4  # 东南西北
    NUM_JIAN_RANKS = 3  # 中发白
    TOTAL_TILE_TYPES = 34  # 9*3 + 4 + 3 = 34
