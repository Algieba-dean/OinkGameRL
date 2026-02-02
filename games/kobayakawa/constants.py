"""Constants for Kobayakawa game."""


class PlayerConsts:
    """Player-related constants."""

    ALLOWED_PLAYER_NUM: tuple[int, ...] = (3, 4, 5, 6)
    MIN_PLAYERS: int = 3
    MAX_PLAYERS: int = 6
    INITIAL_COINS: int = 4


class CardConsts:
    """Card-related constants."""

    TOTAL_CARDS: int = 15
    MIN_VALUE: int = 1
    MAX_VALUE: int = 15


class GameConsts:
    """Game-related constants."""

    ROUNDS_PER_GAME: int = 7
    BET_AMOUNT: int = 1
