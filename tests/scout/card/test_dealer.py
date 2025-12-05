import math

import pytest

from games.scout.card.dealer import Dealer
from games.scout.constants import PlayerConsts


@pytest.fixture
def dealer() -> Dealer:
    return Dealer()


class TestDealerInteraction:
    def test_init_calls_internal_methods(self, mocker):
        flip_cards_mock = mocker.spy(Dealer, "_Dealer__random_flip_all_cards")
        initial_cards_mock = mocker.spy(Dealer, "_Dealer__initailize_card_queue_dict")

        Dealer()
        assert (
            flip_cards_mock.call_count == 4
        )  # player 2,3,4,5, totally will be called in 4 times
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

    @pytest.mark.parametrize(
        argnames="player_num",
        argvalues=[2, 4],
    )
    def test_forbidden_card_in_two_or_four_players(self, dealer, player_num):
        # card with 9 and 10 together should not shown up when players is 2 or 4
        dispatched_cards = dealer.dispatch_cards(player_num=player_num)
        is_any_forbidden_card = False

        # validate card for each player
        for player_cards in dispatched_cards:
            for card in player_cards:
                # check where fliped or not, (9,10) card should not be here
                if (card.top == 9 and card.bottom == 10) or (
                    card.top == 10 and card.top == 9
                ):
                    is_any_forbidden_card = True
                    break

        assert is_any_forbidden_card is False

    def test_forbidden_card_in_three_players(self, dealer):
        # card with 10 together should not shown up when players is 2 or 4
        dispatched_cards = dealer.dispatch_cards(player_num=3)
        is_any_forbidden_card = False

        # validate card for each player
        for player_cards in dispatched_cards:
            for card in player_cards:
                # check where fliped or not, (*,10) card should not be here
                if card.top == 10 or card.bottom == 10:
                    is_any_forbidden_card = True
                    break
        assert is_any_forbidden_card is False


class TestDealerDispatch:
    @pytest.mark.parametrize(
        argnames="round_num",
        argvalues=[2, 3, 4],
    )
    @pytest.mark.parametrize(
        argnames="player_num",
        argvalues=[3, 4, 5],
    )
    def test_dispatch_multi_rounds(self, mocker, dealer, player_num, round_num):
        reload_queue_spy = mocker.spy(dealer, "_Dealer__initailize_card_queue_dict")
        for _ in range(round_num):
            dealer.dispatch_cards(player_num=player_num)
        assert reload_queue_spy.call_count == round_num - 1

    @pytest.mark.parametrize(
        argnames="round_num",
        argvalues=[1, 2, 3, 4, 5, 6],
    )
    def test_two_players_dispatch_multi_rounds(self, mocker, dealer, round_num):
        reload_queue_spy = mocker.spy(dealer, "_Dealer__initailize_card_queue_dict")
        for _ in range(round_num):
            dealer.dispatch_cards(player_num=2)

        if round_num % 2 == 0:
            # as for 2 players, 2 rounds one reload
            assert reload_queue_spy.call_count == math.floor(round_num / 2) - 1
        else:
            assert reload_queue_spy.call_count == math.floor(round_num / 2)
