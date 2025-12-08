class CardConsts:
    # card csv required columns
    IDX: str = "idx"
    BIGGER_NUMBER: str = "bigger_number"
    SMALLER_NUMBER: str = "smaller_number"
    SUPPORTED_PLAYERS: str = "supported_players"

    # card number
    TOTAL_CARD_NUMBER: int = 45


class PlayerConsts:
    ALLOWED_PLAYER_NUM: list[int] = [
        2,
        3,
        4,
        5,
    ]
    PLAYER_CARD_NUM: dict[int, int] = {
        2: 11,
        3: 12,
        4: 11,
        5: 9,
    }


class BoardConsts:
    EMPTY_OWNER_ID = -1
