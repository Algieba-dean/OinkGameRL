"""Constants for Maskmen game."""


class PlayerConsts:
    """Player-related constants."""

    ALLOWED_PLAYER_NUM: tuple[int, ...] = (2, 3, 4, 5, 6)
    MIN_PLAYERS: int = 2
    MAX_PLAYERS: int = 6


class CardConsts:
    """Card-related constants."""

    TOTAL_CARDS: int = 18
    CARD_COLORS: tuple[str, ...] = (
        "red",
        "blue",
        "yellow",
        "green",
        "purple",
        "orange",
    )
    CARDS_PER_COLOR: int = 3
    CARD_VALUES: tuple[int, ...] = (1, 2, 3)


class GameConsts:
    """Game-related constants."""

    INITIAL_HAND_SIZE: dict[int, int] = {
        2: 6,
        3: 5,
        4: 4,
        5: 3,
        6: 3,
    }
    WINNING_SETS: int = 3
