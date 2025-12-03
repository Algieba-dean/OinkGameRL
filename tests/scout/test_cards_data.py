import re
from pathlib import Path

import pytest

from games.scout.CardData import CardData


class TestCardDataContract: ...


class TestLoadCards:
    def test_load_unexisted_path(self):
        unexisted_file = Path("~/foobar.csv")
        with pytest.raises(FileNotFoundError, match="data path .* does not exist."):
            CardData(data_path=unexisted_file)

    @pytest.mark.parametrize(argnames="suffix", argvalues=[(".json"), (".xlsx")])
    def test_load_wrong_format(self, suffix):
        unexpected_format_file = Path(f"./tests/test_data/wrong_format_data{suffix}")
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"format {suffix} is not supported, supported suffix(s) is/are {CardData.SUPPORTED_SUFFIXS}"
            ),
        ):
            CardData(data_path=unexpected_format_file)

    # def test_load_invalid_data(self, scout_test_data_dir):
    #     invalid_data_file = scout_test_data_dir / "invalid_data.csv"
    #     with pytest.raises(ValueError, match="invalid data file"):
    #         CardData(data_path=invalid_data_file)
    #     ...

    # def test_load_success(self, card_data_path):
    #     CardData(data_path=card_data_path)


# class TestCardValidation:
#     def test_validate_card_numbers(self): ...
#     def test_validate_card_unique(self): ...
#     def test_validate_card_supported_players(self): ...
#
#     ...
