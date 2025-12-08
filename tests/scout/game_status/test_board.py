from copy import copy

import pytest

from games.scout.constants import BoardConsts
from games.scout.enums import ScoutPosition
from games.scout.game_status.board import Board


@pytest.fixture
def board() -> Board:
    return Board()


FOO_PLAYER_IDX = 0


class TestBoardContract:
    def test_immutable_cards_property(self, board):
        with pytest.raises(
            AttributeError, match="property 'cards' of 'Board' object has no setter"
        ):
            board.cards = []

    def test_immutable_cards_value(self, board, mocker, card_factory):
        mock_cards = [card_factory(top=top) for top in range(5)]
        mocker.patch.object(board, "_Board__cards", mock_cards)
        with pytest.raises(
            TypeError, match="'tuple' object does not support item assignment"
        ):
            board.cards[0] = card_factory(top=5)

    def test_immutable_owner_id_property(self, board):
        with pytest.raises(
            AttributeError, match="property 'owner_idx' of 'Board' object has no setter"
        ):
            board.owner_idx = 1

    def test_initialized_properties(self):
        initialized_board = Board()
        assert initialized_board.cards == ()
        assert initialized_board.owner_idx == BoardConsts.EMPTY_OWNER_ID


class TestPlayToBoard:
    @pytest.mark.parametrize(argnames="card_tops", argvalues=[[1], [1, 2], [2, 2]])
    def test_play_to_board(self, card_tops, board, card_factory):
        played_cards = [card_factory(top=top) for top in card_tops]
        board.play_to_board(player_idx=FOO_PLAYER_IDX, played_cards=played_cards)
        assert board.owner_idx == FOO_PLAYER_IDX
        assert list(board.cards) == played_cards


class TestScoutFromBoard:
    MOCK_OWNER_IDX = 1

    @pytest.fixture
    def mocked_normal_board(self, mocker, card_factory) -> Board:
        mocked_board = Board()
        cards = [card_factory(top=top) for top in range(1, 5)]
        mocker.patch.object(mocked_board, "_Board__cards", cards)
        mocker.patch.object(mocked_board, "_Board__owner_idx", self.MOCK_OWNER_IDX)
        return mocked_board

    @pytest.fixture
    def mocked_one_card_board(self, mocker, card_factory) -> Board:
        mocked_board = Board()
        cards = [card_factory(top=1)]
        mocker.patch.object(mocked_board, "_Board__owner_idx", self.MOCK_OWNER_IDX)
        mocker.patch.object(mocked_board, "_Board__cards", cards)
        return mocked_board

    @pytest.mark.parametrize(
        argnames="scout_position", argvalues=[ScoutPosition.LEFT, ScoutPosition.RIGHT]
    )
    def test_scout_empty_board(self, board, scout_position):
        with pytest.raises(ValueError, match="can't scout from empty board"):
            board.scout_from_board(scout_position=scout_position)

    def test_normal_scout_left(self, mocked_normal_board):
        old_board_cards = copy(mocked_normal_board.cards)
        scoutted_card = mocked_normal_board.scout_from_board(ScoutPosition.LEFT)

        assert scoutted_card.top == old_board_cards[0].top
        assert scoutted_card.bottom == old_board_cards[0].bottom
        assert mocked_normal_board.cards == old_board_cards[1:]
        assert mocked_normal_board.owner_idx == self.MOCK_OWNER_IDX

    def test_normal_scout_right(self, mocked_normal_board):
        old_board_cards = copy(mocked_normal_board.cards)
        scoutted_card = mocked_normal_board.scout_from_board(ScoutPosition.RIGHT)

        assert scoutted_card.top == old_board_cards[-1].top
        assert scoutted_card.bottom == old_board_cards[-1].bottom
        assert mocked_normal_board.cards == old_board_cards[:-1]
        assert mocked_normal_board.owner_idx == self.MOCK_OWNER_IDX

    @pytest.mark.parametrize(
        argnames="scout_position", argvalues=[ScoutPosition.LEFT, ScoutPosition.RIGHT]
    )
    def test_one_card_scout(self, mocked_one_card_board, scout_position):
        old_board_cards = copy(mocked_one_card_board.cards)
        scoutted_card = mocked_one_card_board.scout_from_board(scout_position)

        assert scoutted_card.top == old_board_cards[-1].top
        assert scoutted_card.bottom == old_board_cards[-1].bottom
        assert mocked_one_card_board.cards == old_board_cards[:-1]

        assert mocked_one_card_board.cards == ()
        assert mocked_one_card_board.owner_idx == BoardConsts.EMPTY_OWNER_ID
