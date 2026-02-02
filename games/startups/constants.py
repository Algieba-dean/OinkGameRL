"""Constants for Startups game."""


class PlayerConsts:
    """Player-related constants."""

    ALLOWED_PLAYER_NUM: tuple[int, ...] = (3, 4, 5, 6, 7)
    MIN_PLAYERS: int = 3
    MAX_PLAYERS: int = 7
    INITIAL_COINS: int = 10


class CardConsts:
    """Card-related constants."""

    NUM_COMPANIES: int = 6
    CARDS_PER_COMPANY: int = 6
    TOTAL_CARDS: int = 36


class CompanyConsts:
    """Company-related constants."""

    COMPANY_NAMES: tuple[str, ...] = (
        "Appy Fizz",
        "Beeswax",
        "Crabwalk",
        "Driftwood",
        "Elephun",
        "Fishtank",
    )
    COMPANY_VALUES: tuple[int, ...] = (5, 4, 3, 3, 2, 1)
