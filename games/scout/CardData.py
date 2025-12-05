import importlib.resources
import json
from pathlib import Path

import pandas as pd

from games.scout import data
from games.scout.Cards import Card
from games.scout.constants import CardConsts


class CardData:
    SUPPORTED_SUFFIXS: list[str] = [
        ".csv",
    ]
    REQUIRED_DATA_COLUMNS: list[str] = [
        CardConsts.BIGGER_NUMBER,
        CardConsts.SMALLER_NUMBER,
        CardConsts.SUPPORTED_PLAYERS,
    ]
    ALLOWED_PLAYER_NUM: list[int] = [2, 3, 4, 5]

    def __init__(self, data_path: Path | None = None):
        self.__data_path: Path = (
            data_path if data_path is not None else self.__get_default_data_path()
        )
        self.__validate_data_path()
        self.__validate_and_load_data_csv()
        self.__data = self.__validate_and_load_data_csv()
        self.__cards: list[Card] = self.__convert_data2cards()

    @property
    def cards(self) -> list[Card]:
        return self.__cards

    def get_cards_for_player(self, player_num: int) -> list[Card]:
        if player_num not in self.ALLOWED_PLAYER_NUM:
            raise ValueError(f"unexpected player num {player_num}")
        return [card for card in self.cards if player_num in card.supported_players]

    def __validate_data_path(self) -> None:
        if self.__data_path.suffix not in self.SUPPORTED_SUFFIXS:
            raise ValueError(
                f"format {self.__data_path.suffix} is not supported, supported suffix(s) is/are {self.SUPPORTED_SUFFIXS}"
            )
        if not self.__data_path.exists():
            raise FileNotFoundError(f"data path {self.__data_path} does not exist.")

    def __validate_and_load_data_csv(self) -> pd.DataFrame:
        with open(file=self.__data_path, encoding="utf-8") as f:
            data = pd.read_csv(filepath_or_buffer=f)
        missing_column = []
        # check missing required data columns
        for required_column in self.REQUIRED_DATA_COLUMNS:
            if required_column not in data.columns:
                missing_column.append(required_column)
        if len(missing_column) > 0:
            raise ValueError(f"missing {missing_column} in {self.__data_path}")
        return data

    def __convert_data2cards(self) -> list[Card]:
        # validate data format
        self.__data[CardConsts.SUPPORTED_PLAYERS] = self.__data[
            CardConsts.SUPPORTED_PLAYERS
        ].apply(json.loads)
        data_dict = self.__data.to_dict(orient="records")
        return [
            Card(
                top=card[CardConsts.BIGGER_NUMBER],
                bottom=card[CardConsts.SMALLER_NUMBER],
                supported_players=card[CardConsts.SUPPORTED_PLAYERS],
            )
            for card in data_dict
        ]

    @staticmethod
    def __get_default_data_path() -> Path:
        resource = importlib.resources.files(data) / "card.csv"
        with importlib.resources.as_file(resource) as path:
            return path
