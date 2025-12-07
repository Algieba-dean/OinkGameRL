import pytest

from games.scout.card.cards import Card
from games.scout.card.playable_checker import PlayableChecker


def to_card(top: int) -> Card:
    return Card(idx=1, top=top, bottom=1, supported_players=[2])


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
        ([], [1], True),
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
def test_target_playable(board_tops, target_tops, is_playable):
    board_cards = [to_card(top) for top in board_tops]
    target_cards = [to_card(top) for top in target_tops]
    assert (
        PlayableChecker.is_playable(board_cards=board_cards, target_cards=target_cards)
        is is_playable
    )


class TestGetPlayableSubsets:
    @pytest.mark.parametrize(
        argnames="target_tops,expected_subsets",
        argvalues=[
            # only singles possible
            ([1], {(0, 0)}),
            ([1, 3, 5], {(0, 0), (1, 1), (2, 2)}),
            # sequence possible
            ([1, 2], {(0, 0), (1, 1), (0, 1)}),
            ([2, 1], {(0, 0), (1, 1), (0, 1)}),
            (
                [1, 2, 3],
                {
                    # single
                    (0, 0),
                    (1, 1),
                    (2, 2),
                    # two
                    (0, 1),
                    (1, 2),
                    # three
                    (0, 2),
                },
            ),
            (
                [3, 2, 1],
                {
                    # single
                    (0, 0),
                    (1, 1),
                    (2, 2),
                    # two
                    (0, 1),
                    (1, 2),
                    # three
                    (0, 2),
                },
            ),
            # same rank
            ([1, 1], {(0, 0), (1, 1), (0, 1)}),
            # mixed
            (
                [1, 2, 8, 8, 5],
                {
                    # single
                    (0, 0),
                    (1, 1),
                    (2, 2),
                    (3, 3),
                    (4, 4),
                    # sequence
                    (0, 1),
                    # same rank
                    (2, 3),
                },
            ),
        ],
    )
    def test_empty_board(self, target_tops, expected_subsets):
        target_cards = [to_card(top=top) for top in target_tops]
        subsets = PlayableChecker.get_all_playable_subsets(
            board_cards=[], target_cards=target_cards
        )
        assert set(subsets) == expected_subsets
