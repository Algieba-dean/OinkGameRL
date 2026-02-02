"""Tests for Kobayakawa Card module."""

import pytest

from games.kobayakawa.card import Card


class TestCardInit:
    """Test Card initialization."""

    def test_create_card(self):
        card = Card(value=5)
        assert card.value == 5

    @pytest.mark.parametrize("value", range(1, 16))
    def test_valid_values(self, value):
        card = Card(value=value)
        assert card.value == value

    @pytest.mark.parametrize("invalid_value", [0, 16, -1, 100])
    def test_invalid_values_raise_error(self, invalid_value):
        with pytest.raises(ValueError, match="Invalid card value"):
            Card(value=invalid_value)


class TestCardComparison:
    """Test Card comparison methods."""

    def test_equal_cards(self):
        card1 = Card(value=5)
        card2 = Card(value=5)
        assert card1 == card2

    def test_different_cards(self):
        card1 = Card(value=5)
        card2 = Card(value=6)
        assert card1 != card2

    def test_not_equal_to_non_card(self):
        card = Card(value=5)
        assert card != "not a card"
        assert card != 5

    def test_less_than(self):
        card1 = Card(value=3)
        card2 = Card(value=7)
        assert card1 < card2

    def test_hash_equal_cards(self):
        card1 = Card(value=5)
        card2 = Card(value=5)
        assert hash(card1) == hash(card2)

    def test_cards_usable_in_set(self):
        card1 = Card(value=5)
        card2 = Card(value=5)
        card3 = Card(value=6)
        card_set = {card1, card2, card3}
        assert len(card_set) == 2


class TestCardRepresentation:
    """Test Card string representations."""

    def test_repr(self):
        card = Card(value=10)
        assert repr(card) == "Card(10)"

    def test_str(self):
        card = Card(value=10)
        assert str(card) == "10"
