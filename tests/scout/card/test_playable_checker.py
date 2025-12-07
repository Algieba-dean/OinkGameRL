import pytest

from games.scout.card.cards import Card
from games.scout.card.playable_checker import PlayableChecker


class TestIsPlayable:
    @pytest.mark.parametrize(
        argnames="board_tops,target_tops,is_playable",
        argvalues=[
            # target tops are invalid
            ([], [1, 3, 2], False),
            ([], [1, 2, 2], False),
            ([1], [1, 3, 2], False),
            ([1], [1, 1, 2], False),
            # target tops are shorter
            ([1, 2], [10], False),
            ([1, 2, 3, 4, 5], [10, 10, 10, 10, False], False),
            # target tops are longer
            ([10], [1, 2], True),
            ([10, 10, 10, 10], [1, 2, 3, 4, 5], True),
            # target tops pattern is smaller
            ([1, 1], [9, 10], False),
            ([1, 1, 1, 1], [7, 8, 9, 10], False),
            # target tops pattern is bigger
            ([9, 10], [1, 1], True),
            ([7, 8, 9, 10], [1, 1, 1, 1], True),
            # target tops value is smaller
            ([2, 3], [1, 2], False),
            ([7, 8, 9, 10], [6, 7, 8, 9], False),
            ([2, 2], [1, 1], False),
            # target tops value is same
            ([1, 2], [1, 2], False),
            ([7, 8, 9, 10], [7, 8, 9, 10], False),
            ([1, 1], [1, 1], False),
            # target tops value is bigger
            ([1, 2], [2, 3], True),
            ([6, 7, 8, 9], [7, 8, 9, 10], True),
            ([1, 1], [2, 2], True),
        ],
    )
    def test_target_is_not_playable(self, board_tops, target_tops, is_playable):
        board_cards = [
            Card(idx=1, top=top, bottom=1, supported_players=[2]) for top in board_tops
        ]
        target_cards = [
            Card(idx=1, top=top, bottom=1, supported_players=[2]) for top in target_tops
        ]
        assert (
            PlayableChecker.is_playable(
                board_cards=board_cards, target_cards=target_cards
            )
            is is_playable
        )
