import pytest

from games.scout.card.Dealer import Dealer
from games.scout.Constants import PlayerConsts


@pytest.fixture
def dealer() -> Dealer:
    return Dealer()


class TestDealerContract: ...


class TestDealerInteraction:
    def test_init_calls_internal_methods(self, mocker):
        flip_cards_mock = mocker.spy(Dealer, "_Dealer__random_flip_all_cards")
        initial_cards_mock = mocker.spy(Dealer, "_Dealer__initailize_and_get_cards")

        Dealer()
        flip_cards_mock.assert_called_once()
        initial_cards_mock.assert_called_once()


class TestDealerBasic:
    @pytest.mark.parametrize(argnames="player_num", argvalues=[-1, 0, 1, 6, 999])
    def test_invalid_player_dispatch(self, dealer, player_num):
        with pytest.raises(
            ValueError, match=f"unexpected player number:{player_num} for card dispatch"
        ):
            dealer.dispatch_cards(player_num=player_num)

    @pytest.mark.parametrize(
        argnames="player_num", argvalues=PlayerConsts.ALLOWED_PLAYER_NUM
    )
    def test_dispatched_card_follow_player_num(self, dealer, player_num):
        assert len(dealer.dispatch_cards(player_num=player_num)) == player_num

    @pytest.mark.parametrize(
        argnames="player_num", argvalues=PlayerConsts.ALLOWED_PLAYER_NUM
    )
    def test_dispached_card_follow_card_num_on_player(self, dealer, player_num):
        players_cards = dealer.dispatch_cards(player_num=player_num)
        for player_card in players_cards:
            assert len(player_card) == PlayerConsts.PLAYER_CARD_NUM[player_num]


class TestDealerDispatch:
    def test_dispatch_multi_rounds(self, dealer): ...
    def test_two_players_dispatch(self, dealer): ...
    def test_two_players_dispatch_multi_rounds(self, dealer): ...
