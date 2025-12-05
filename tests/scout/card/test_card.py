import pytest

from games.scout.card.Cards import Card


class TestCardContract:
    def card(self):
        return Card(top=1, bottom=2, supported_players=[2, 3, 4, 5])

    def test_card_attribution(self):
        card = self.card()
        assert card.top == 1
        assert card.bottom == 2

    def test_immutable_properties(self):
        card = self.card()
        with pytest.raises(
            AttributeError,
            match="property 'top' of 'Card' object has no setter",
        ):
            card.top = 10
        with pytest.raises(
            AttributeError, match="property 'bottom' of 'Card' object has no setter"
        ):
            card.bottom = 6
        with pytest.raises(
            AttributeError,
            match="property 'supported_players' of 'Card' object has no setter",
        ):
            card.supported_players = [2, 3, 4, 5]


class TestCardFunctionality:
    def card(self):
        return Card(top=1, bottom=2, supported_players=[2, 3, 4, 5])

    def test_card_flip(self):
        card = self.card()
        fliped_card = card.flip()
        assert card.top == 2
        assert card.bottom == 1
        assert fliped_card.top == 2
        assert fliped_card.bottom == 1

    def test_card_display(self):
        card = self.card()
        assert str(card) == f"[{card.top}]/{card.bottom}"
        assert repr(card) == f"[{card.top}]/{card.bottom}"

    @pytest.mark.parametrize(
        argnames=(
            "current_top, current_bottom, another_top, another_bottom, expected_compared_result"  # compare result 0 is =, -1 is <, 1 is >
        ),
        argvalues=[
            (1, 2, 3, 4, -1),
            (1, 3, 2, 3, -1),
            (4, 3, 2, 1, 1),
            (2, 1, 1, 9, 1),
            (9, 10, 9, 8, 0),
            (3, 1, 3, 2, 0),
        ],
        ids=[
            "both top bottom smaller",
            "top smaller bottom bigger",
            "both top bottom bigger",
            "top bigger, bottom smaller",
            "top equal, bottom bigger",
            "top equal, bottom smaller",
        ],
    )
    def test_card_comparison(
        self,
        current_top,
        current_bottom,
        another_top,
        another_bottom,
        expected_compared_result,
    ):
        current_card = Card(
            top=current_top, bottom=current_bottom, supported_players=[2]
        )
        another_card = Card(
            top=another_top, bottom=another_bottom, supported_players=[2]
        )
        compared_result = (current_card > another_card) - (current_card < another_card)
        assert compared_result == expected_compared_result

    @pytest.mark.parametrize(
        argnames="supported_players",
        argvalues=[[2, 3, 4, 5], [2, 4, 5], [5]],
        ids=["2,3,4,5", "2,4,5", "5"],
    )
    def test_card_support_player(self, supported_players):
        card = Card(top=1, bottom=2, supported_players=supported_players)
        assert 1 not in card.supported_players
        for supported_num in supported_players:
            assert supported_num in card.supported_players
