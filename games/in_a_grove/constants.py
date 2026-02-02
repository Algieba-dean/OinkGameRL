"""Constants for In a Grove game."""


class PlayerConsts:
    """Player-related constants."""

    ALLOWED_PLAYER_NUM: tuple[int, ...] = (2, 3, 4)
    MIN_PLAYERS: int = 2
    MAX_PLAYERS: int = 4


class CardConsts:
    """Card-related constants."""

    TOTAL_SUSPECT_CARDS: int = 8
    SUSPECT_VALUES: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8)
    ACCOMPLICE_VALUE: int = 1


class GameConsts:
    """Game-related constants."""

    TILES_PER_PLAYER: int = 3
    ROUNDS_PER_GAME: int = 3
