"""Tests for Maskmen Card module."""

import pytest

from games.maskmen.card import Card
from games.maskmen.enums import CardColor


class TestCardInit:
    """Test Card initialization."""

    def test_create_card(self):
        card = Card(color=CardColor.RED, value=1)
        assert card.color == CardColor.RED
        assert card.value == 1

    @pytest.mark.parametrize("value", [1, 2, 3])
    def test_valid_values(self, value):
        card = Card(color=CardColor.BLUE, value=value)
        assert card.value == value

    @pytest.mark.parametrize("invalid_value", [0, 4, -1, 10])
    def test_invalid_values_raise_error(self, invalid_value):
        with pytest.raises(ValueError, match="Invalid card value"):
            Card(color=CardColor.RED, value=invalid_value)


class TestCardProperties:
    """Test Card properties."""

    def test_color_property(self):
        card = Card(color=CardColor.GREEN, value=2)
        assert card.color == CardColor.GREEN

    def test_value_property(self):
        card = Card(color=CardColor.YELLOW, value=3)
        assert card.value == 3


class TestCardComparison:
    """Test Card comparison methods."""

    def test_equal_cards(self):
        card1 = Card(color=CardColor.RED, value=1)
        card2 = Card(color=CardColor.RED, value=1)
        assert card1 == card2

    def test_different_color(self):
        card1 = Card(color=CardColor.RED, value=1)
        card2 = Card(color=CardColor.BLUE, value=1)
        assert card1 != card2

    def test_different_value(self):
        card1 = Card(color=CardColor.RED, value=1)
        card2 = Card(color=CardColor.RED, value=2)
        assert card1 != card2

    def test_not_equal_to_non_card(self):
        card = Card(color=CardColor.RED, value=1)
        assert card != "not a card"
        assert card != 1


class TestCardHash:
    """Test Card hash."""

    def test_equal_cards_same_hash(self):
        card1 = Card(color=CardColor.RED, value=1)
        card2 = Card(color=CardColor.RED, value=1)
        assert hash(card1) == hash(card2)

    def test_cards_usable_in_set(self):
        card1 = Card(color=CardColor.RED, value=1)
        card2 = Card(color=CardColor.RED, value=1)
        card3 = Card(color=CardColor.BLUE, value=1)
        card_set = {card1, card2, card3}
        assert len(card_set) == 2


class TestCardRepresentation:
    """Test Card string representations."""

    def test_repr(self):
        card = Card(color=CardColor.RED, value=2)
        assert repr(card) == "Card(RED, 2)"

    def test_str(self):
        card = Card(color=CardColor.RED, value=2)
        assert str(card) == "R2"

    @pytest.mark.parametrize(
        "color,expected_prefix",
        [
            (CardColor.RED, "R"),
            (CardColor.BLUE, "B"),
            (CardColor.YELLOW, "Y"),
            (CardColor.GREEN, "G"),
            (CardColor.PURPLE, "P"),
            (CardColor.ORANGE, "O"),
        ],
    )
    def test_str_color_prefix(self, color, expected_prefix):
        card = Card(color=color, value=1)
        assert str(card).startswith(expected_prefix)
