import csv
import importlib.resources
import json
from functools import lru_cache
from pathlib import Path

from games.scout import data
from games.scout.card.cards import Card
from games.scout.constants import CardConsts, PlayerConsts


class CardData:
    SUPPORTED_SUFFIXS: list[str] = [
        ".csv",
    ]
    REQUIRED_DATA_COLUMNS: list[str] = [
        CardConsts.IDX,
        CardConsts.BIGGER_NUMBER,
        CardConsts.SMALLER_NUMBER,
        CardConsts.SUPPORTED_PLAYERS,
    ]

    def __init__(self, data_path: Path | None = None) -> None:
        self.__data_path: Path = (
            data_path if data_path is not None else self.__get_default_data_path()
        )
        self.__validate_data_path()
        self.__cards: list[Card] = self.__load_cards_cached(self.__data_path)

    @property
    def cards(self) -> list[Card]:
        return self.__cards

    def get_cards_for_player(self, player_num: int) -> list[Card]:
        if player_num not in PlayerConsts.ALLOWED_PLAYER_NUM:
            raise ValueError(f"unexpected player num {player_num}")
        return [card for card in self.cards if player_num in card.supported_players]

    def __validate_data_path(self) -> None:
        if self.__data_path.suffix not in self.SUPPORTED_SUFFIXS:
            raise ValueError(
                f"format {self.__data_path.suffix} is not supported, supported suffix(s) is/are {self.SUPPORTED_SUFFIXS}"
            )
        if not self.__data_path.exists():
            raise FileNotFoundError(f"data path {self.__data_path} does not exist.")

    @staticmethod
    def __get_default_data_path() -> Path:
        resource = importlib.resources.files(data) / "card.csv"
        with importlib.resources.as_file(resource) as path:
            return path

    @staticmethod
    @lru_cache(maxsize=1)
    def __load_cards_cached(path: Path) -> list[Card]:
        """take response of I/O and parsing the cards

        Args:
            path (Path): card path

        Returns:
            list[Card]: cards
        """
        cards = []

        # load data
        with open(file=path, encoding="utf-8") as f:
            reader = csv.DictReader(f)

            # validate columns
            if reader.fieldnames is None:
                raise ValueError(f"empty file {path}")
            missing_colums = [
                column
                for column in CardData.REQUIRED_DATA_COLUMNS
                if column not in reader.fieldnames
            ]
            if missing_colums:
                raise ValueError(f"missing {missing_colums} in {path}")

            # parsing data
            for row in reader:
                idx = int(row[CardConsts.IDX])
                top = int(row[CardConsts.BIGGER_NUMBER])
                bottom = int(row[CardConsts.SMALLER_NUMBER])

                # parse supported_players
                supported_players: list[int] = json.loads(
                    row[CardConsts.SUPPORTED_PLAYERS]
                )
                cards.append(
                    Card(
                        idx=idx,
                        top=top,
                        bottom=bottom,
                        supported_players=supported_players,
                    )
                )
        return cards
