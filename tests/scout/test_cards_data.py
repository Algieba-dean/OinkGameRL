import re
from pathlib import Path

import pytest

from games.scout.CardData import CardData
from games.scout.constants import CardConsts


@pytest.fixture
def card_data() -> CardData:
    return CardData()


class TestCardDataContract:
    def test_cards_propetry(self, card_data):
        with pytest.raises(
            AttributeError, match="property 'cards' of '.*' object has no setter"
        ):
            card_data.cards = []


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

    @pytest.mark.parametrize(
        argnames=("required_column"),
        argvalues=CardData.REQUIRED_DATA_COLUMNS,
        ids=lambda column: f"missing {column}",
    )
    def test_load_invalid_column_name(self, scout_test_data_dir, required_column):
        invalid_data_file = scout_test_data_dir / "invalid_column_data.csv"
        with pytest.raises(
            ValueError,
            match=f"missing .*{required_column}.* in *",
        ):
            CardData(data_path=invalid_data_file)

    def test_load_invalid_data(self, scout_test_data_dir):
        invalid_data_file = scout_test_data_dir / "invalid_supported_players.csv"
        with pytest.raises(ValueError):
            CardData(data_path=invalid_data_file)

    def test_load_success(self):
        card_data = CardData()
        assert len(card_data.cards) == CardConsts.TOTAL_CARD_NUMBER


class TestCardValidation:
    @pytest.mark.parametrize(
        argnames="player_num, expected_card_num",
        argvalues=[
            (2, 11 * 2 * 2),  # 2 players, 11 for each one, and cards for two rounds
            (3, 12 * 3),  # 3 players, 12 for each
            (4, 11 * 4),  # 4 players, 11 for each
            (5, 9 * 5),  # 5 players, 9 for each
        ],
    )
    def test_card_numbers(self, card_data, player_num, expected_card_num):
        supported_cards = [
            card for card in card_data.cards if player_num in card.supported_players
        ]
        assert len(supported_cards) == expected_card_num

    @pytest.mark.parametrize(
        argnames="player_num",
        argvalues=[2, 4],
    )
    def test_forbidden_card_in_two_or_four_players(self, card_data, player_num):
        # card with 9 and 10 together should not shown up when players is 2 or 4
        supported_cards = [
            card for card in card_data.cards if player_num in card.supported_players
        ]
        is_any_forbidden_card = False
        for card in supported_cards:
            if (card.top == 9 and card.bottom == 10) or (
                card.top == 10 and card.top == 9
            ):
                is_any_forbidden_card = True
                break

        assert is_any_forbidden_card is False

    def test_forbidden_card_in_three_players(self, card_data):
        # card with 10 together should not shown up when players is 2 or 4
        supported_cards = [
            card for card in card_data.cards if 3 in card.supported_players
        ]
        is_any_forbidden_card = False
        for card in supported_cards:
            if card.top == 10 or card.bottom == 10:
                is_any_forbidden_card = True
                break
        assert is_any_forbidden_card is False

    def test_card_unique_top_bottom(self, card_data):
        # top and bottom number should be difference on each card
        is_any_identical_card = False
        for card in card_data.cards:
            if card.top == card.bottom:
                is_any_identical_card = True
                break
        assert is_any_identical_card is False

    def test_card_unique(self, card_data):
        # should have no any identical card, where flip or not
        keys = [tuple(sorted((card.top, card.bottom))) for card in card_data.cards]
        assert len(set(keys)) == len(keys)

    ...
