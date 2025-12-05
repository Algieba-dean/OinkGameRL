class CardConsts:
    # card csv required columns
    BIGGER_NUMBER: str = "bigger_number"
    SMALLER_NUMBER: str = "smaller_number"
    SUPPORTED_PLAYERS: str = "supported_players"

    # card number
    TOTAL_CARD_NUMBER: int = 45


class PlayerConsts:
    ALLOWED_PLAYER_NUM: list[int] = [2, 3, 4, 5]
