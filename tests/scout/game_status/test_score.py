import re

import pytest

from games.scout.constants import PlayerConsts
from games.scout.game_status.score import Score

FOO_PLAYER_NUM = 4


@pytest.fixture
def score() -> Score:
    return Score(player_num=FOO_PLAYER_NUM)


def get_empty_score_dict(player_num: int = FOO_PLAYER_NUM) -> dict[int, int]:
    return dict.fromkeys(range(player_num), 0)


class TestScoreContract:
    def test_imutable_player_num_property(self, score):
        with pytest.raises(
            AttributeError,
            match="property 'player_num' of 'Score' object has no setter",
        ):
            score.player_num = 2

    def test_imutable_score_dict_property(self, score):
        with pytest.raises(
            AttributeError,
            match="property 'score_dict' of 'Score' object has no setter",
        ):
            score.score_dict = {}

    def test_imutable_score_dict_value(self, score):
        with pytest.raises(
            TypeError, match="'mappingproxy' object does not support item assignment"
        ):
            score.score_dict[0] = 10

    @pytest.mark.parametrize(argnames="invalid_num", argvalues=[-1, 0, 1, 6])
    def test_invalid_player_num(self, invalid_num):
        with pytest.raises(ValueError, match=f"invalid player num {invalid_num} *"):
            Score(player_num=invalid_num)

    @pytest.mark.parametrize(argnames="invalid_num", argvalues=[-1, 0, 1, 6])
    def test_invalid_player_num_for_clean(self, invalid_num, score):
        with pytest.raises(ValueError, match=f"invalid player num {invalid_num} *"):
            score.clean_all_score(player_num=invalid_num)

    @pytest.mark.parametrize(
        argnames="invalid_idx", argvalues=[-1, FOO_PLAYER_NUM, FOO_PLAYER_NUM + 1]
    )
    def test_invalid_player_idx_for_increasement(self, score, invalid_idx):
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"invalid player idx, for player num:{score.player_num}, valid idxs are {list(range(FOO_PLAYER_NUM))}"
            ),
        ):
            score.increase_score(player_idx=invalid_idx, value=1)

    @pytest.mark.parametrize(
        argnames="invalid_idx", argvalues=[-1, FOO_PLAYER_NUM, FOO_PLAYER_NUM + 1]
    )
    def test_invalid_player_idx_for_decreasement(self, score, invalid_idx):
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"invalid player idx, for player num:{score.player_num}, valid idxs are {list(range(FOO_PLAYER_NUM))}"
            ),
        ):
            score.decrease_score(player_idx=invalid_idx, value=1)

    @pytest.mark.parametrize(argnames="invalid_value", argvalues=[-1, 0])
    def test_invalid_value_for_increasement(self, score, invalid_value):
        player_idx = 0
        with pytest.raises(
            ValueError,
            match=f"invalid value {invalid_value}, only positive values are allowed",
        ):
            score.increase_score(player_idx=player_idx, value=invalid_value)

    @pytest.mark.parametrize(argnames="invalid_value", argvalues=[-1, 0])
    def test_invalid_value_for_decreasement(self, score, invalid_value):
        player_idx = 0
        with pytest.raises(
            ValueError,
            match=f"invalid value {invalid_value}, only positive values are allowed",
        ):
            score.decrease_score(player_idx=player_idx, value=invalid_value)


class TestCleanScore:
    @staticmethod
    def do_some_score_change(score_manager: Score):
        score_manager.increase_score(player_idx=0, value=10)
        score_manager.decrease_score(player_idx=1, value=5)
        # the final score should be 0:10,1:-5,2:0,3:0

    @pytest.mark.parametrize(
        argnames="player_num", argvalues=PlayerConsts.ALLOWED_PLAYER_NUM
    )
    def test_clean_all_score(self, score, player_num):
        self.do_some_score_change(score_manager=score)
        score.clean_all_score(player_num=player_num)
        assert score.player_num == player_num
        assert score.score_dict == get_empty_score_dict(player_num=player_num)

    def test_clean_default(self, score):
        self.do_some_score_change(score_manager=score)
        score.clean_all_score()
        assert score.player_num == FOO_PLAYER_NUM
        assert score.score_dict == get_empty_score_dict(player_num=FOO_PLAYER_NUM)


class TestOperateScore:
    @pytest.mark.parametrize(argnames="player_idx", argvalues=range(FOO_PLAYER_NUM))
    @pytest.mark.parametrize(argnames="value", argvalues=[1, 2, 3])
    def test_increase_score(self, score, player_idx, value):
        score.increase_score(player_idx=player_idx, value=value)
        assert score.score_dict[player_idx] == value
        score.increase_score(player_idx=player_idx, value=value)
        assert score.score_dict[player_idx] == value * 2

    @pytest.mark.parametrize(argnames="player_idx", argvalues=range(FOO_PLAYER_NUM))
    @pytest.mark.parametrize(argnames="value", argvalues=[1, 2, 3])
    def test_decrease_score(self, score, player_idx, value):
        score.decrease_score(player_idx=player_idx, value=value)
        assert score.score_dict[player_idx] == 0 - value
        score.decrease_score(player_idx=player_idx, value=value)
        assert score.score_dict[player_idx] == 0 - value * 2
