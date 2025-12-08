import pytest

from games.scout.card.cards import Card


class TestCardContract:
    def card(self):
        return Card(idx=1, top=1, bottom=2, supported_players=[2, 3, 4, 5])

    def test_card_attribution(self):
        card = self.card()
        assert card.top == 1
        assert card.bottom == 2
        assert card.idx == 1
        assert card.supported_players == [2, 3, 4, 5]

    def test_immutable_idx_properties(self):
        with pytest.raises(
            AttributeError,
            match="property 'idx' of 'Card' object has no setter",
        ):
            self.card().idx = 8

    def test_immutable_top_properties(self):
        with pytest.raises(
            AttributeError,
            match="property 'top' of 'Card' object has no setter",
        ):
            self.card().top = 10

    def test_immutable_bottom_properties(self):
        with pytest.raises(
            AttributeError, match="property 'bottom' of 'Card' object has no setter"
        ):
            self.card().bottom = 6

    def test_immutable_supported_players_properties(self):
        with pytest.raises(
            AttributeError,
            match="property 'supported_players' of 'Card' object has no setter",
        ):
            self.card().supported_players = [2, 3, 4, 5]


class TestCardFunctionality:
    def card(self):
        return Card(idx=1, top=1, bottom=2, supported_players=[2, 3, 4, 5])

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
        argnames="current_idx, current_top, current_bottom,another_idx,another_top,another_bottom,expected_result",
        argvalues=[
            (1, 1, 2, 2, 1, 2, False),
            (1, 1, 2, 1, 1, 3, False),
            (1, 1, 2, 1, 3, 2, False),
            (1, 1, 2, 1, 1, 2, True),
            (1, 1, 2, 1, 2, 1, True),
        ],
    )
    def test_card_equality(
        self,
        current_idx,
        current_top,
        current_bottom,
        another_idx,
        another_top,
        another_bottom,
        expected_result,
    ):
        current_card = Card(
            idx=current_idx,
            top=current_top,
            bottom=current_bottom,
            supported_players=[2],
        )
        another_card = Card(
            idx=another_idx,
            top=another_top,
            bottom=another_bottom,
            supported_players=[2],
        )
        result = current_card == another_card
        assert result is expected_result

    @pytest.mark.parametrize(
        argnames="supported_players",
        argvalues=[[2, 3, 4, 5], [2, 4, 5], [5]],
        ids=["2,3,4,5", "2,4,5", "5"],
    )
    def test_card_support_player(self, supported_players):
        card = Card(idx=1, top=1, bottom=2, supported_players=supported_players)
        assert 1 not in card.supported_players
        for supported_num in supported_players:
            assert supported_num in card.supported_players
