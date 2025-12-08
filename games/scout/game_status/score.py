from games.scout.constants import PlayerConsts


class Score:
    def __init__(self, player_num: int) -> None:
        self.__validate_player_num(player_num=player_num)
        self.__player_num: int = player_num
        self.__score_dict: dict[int, int] = self.__get_empty_score_dict(
            player_num=self.__player_num
        )

    @property
    def player_num(self) -> int:
        return self.__player_num

    @property
    def score_dict(self) -> dict[int, int]:
        return self.__score_dict

    def clean_all_score(self, player_num: int | None = None) -> None:
        if player_num is None:
            player_num = self.__player_num
        self.__validate_player_num(player_num=player_num)
        self.__score_dict = self.__get_empty_score_dict(player_num=player_num)
        self.__player_num = player_num

    def increase_score(self, player_idx: int, value: int) -> None:
        self.__validate_player_idx(player_num=self.__player_num, player_idx=player_idx)
        self.__validate_value(value=value)
        self.__score_dict[player_idx] += value

    def decrease_score(self, player_idx: int, value: int) -> None:
        self.__validate_player_idx(player_num=self.__player_num, player_idx=player_idx)
        self.__validate_value(value=value)
        self.__score_dict[player_idx] -= value

    @staticmethod
    def __validate_value(value: int) -> None:
        if value <= 0:
            raise ValueError(f"invalid value {value}, only positive values are allowed")

    @staticmethod
    def __validate_player_idx(player_num: int, player_idx: int) -> None:
        if player_idx not in range(player_num):
            raise ValueError(
                f"invalid player idx, for player num:{player_num}, valid idxs are {list(range(player_num))}"
            )

    @staticmethod
    def __validate_player_num(player_num: int) -> None:
        if player_num not in PlayerConsts.ALLOWED_PLAYER_NUM:
            raise ValueError(
                f"invalid player num {player_num}, supporteds are {PlayerConsts.ALLOWED_PLAYER_NUM}"
            )

    @staticmethod
    def __get_empty_score_dict(player_num: int) -> dict[int, int]:
        return dict.fromkeys(range(player_num), 0)
