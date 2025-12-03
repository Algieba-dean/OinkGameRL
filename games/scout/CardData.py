from pathlib import Path


class CardData:
    SUPPORTED_SUFFIXS: list[str] = [
        ".csv",
    ]

    def __init__(self, data_path: Path):
        self.__data_path = data_path
        self.__valid_data_path()

    def __valid_data_path(self):
        if self.__data_path.suffix not in self.SUPPORTED_SUFFIXS:
            raise ValueError(
                f"format {self.__data_path.suffix} is not supported, supported suffix(s) is/are {self.SUPPORTED_SUFFIXS}"
            )
        if not self.__data_path.exists():
            raise FileNotFoundError(f"data path {self.__data_path} does not exist.")
