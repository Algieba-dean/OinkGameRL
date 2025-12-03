import json
from pathlib import Path

import pandas as pd

from games.scout.Cards import Card


class CardData:
    SUPPORTED_SUFFIXS: list[str] = [
        ".csv",
    ]
    REQUIRED_DATA_COLUMNS: list[str] = [
        "bigger_number",
        "smaller_number",
        "supported_players",
    ]

    def __init__(self, data_path: Path):
        self.__data_path = data_path
        self.__validate_data_path()
        self.__validate_and_load_data_csv()
        self.__data = self.__validate_and_load_data_csv()
        self.cards: list[Card] = self.__convert_data2cards()

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
        self.__data["supported_players"] = self.__data["supported_players"].apply(
            json.loads
        )
        data_dict = self.__data.to_dict(orient="records")
        return [
            Card(
                top=card["bigger_number"],
                bottom=card["smaller_number"],
                supported_players=card["supported_players"],
            )
            for card in data_dict
        ]
