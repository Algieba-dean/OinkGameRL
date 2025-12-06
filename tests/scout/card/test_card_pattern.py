import pytest

from games.scout.card.card_pattern_checker import CardPatternChecker
from games.scout.card.cards import Card
from games.scout.enums import CardPattern


class TestGetCardPattern:
    @pytest.mark.parametrize(
        argnames="top_values, expected_pattern",
        argvalues=[
            # sequence
            ([1, 2, 3, 4], CardPattern.SEQUENCE),
            ([1, 2], CardPattern.SEQUENCE),
            ([4, 3, 2, 1], CardPattern.SEQUENCE),
            # same rank
            ([1], CardPattern.SAME_RANK),
            ([1, 1], CardPattern.SAME_RANK),
            ([1, 1, 1], CardPattern.SAME_RANK),
            ([9, 9, 9, 9], CardPattern.SAME_RANK),
            # invalid pattern
            ([1, 3], CardPattern.INVALID_PATTERN),
            ([1, 2, 4], CardPattern.INVALID_PATTERN),
            ([1, 3, 5, 7, 9], CardPattern.INVALID_PATTERN),
            ([10, 8, 6, 4, 2], CardPattern.INVALID_PATTERN),
            ([1, 3, 2], CardPattern.INVALID_PATTERN),
            ([3, 1, 2], CardPattern.INVALID_PATTERN),
        ],
    )
    def test_get_card_pattern(self, top_values, expected_pattern):
        cards = [
            Card(idx=1, top=top, bottom=1, supported_players=[2]) for top in top_values
        ]
        assert CardPatternChecker.get_pattern(cards=cards) == expected_pattern
