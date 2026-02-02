"""Tests for Startups Card module."""

import pytest

from games.startups.card import Card
from games.startups.enums import Company


class TestCardInit:
    """Test Card initialization."""

    def test_create_card(self):
        card = Card(Company.APPY_FIZZ, 3)
        assert card.company == Company.APPY_FIZZ
        assert card.value == 3

    @pytest.mark.parametrize("invalid_value", [0, -1, 7, 10])
    def test_invalid_value_raises(self, invalid_value):
        with pytest.raises(ValueError):
            Card(Company.APPY_FIZZ, invalid_value)


class TestCardComparison:
    """Test Card comparison methods."""

    def test_equality(self):
        c1 = Card(Company.BEESWAX, 2)
        c2 = Card(Company.BEESWAX, 2)
        assert c1 == c2

    def test_inequality_different_company(self):
        c1 = Card(Company.BEESWAX, 2)
        c2 = Card(Company.CRABWALK, 2)
        assert c1 != c2

    def test_inequality_different_value(self):
        c1 = Card(Company.BEESWAX, 2)
        c2 = Card(Company.BEESWAX, 3)
        assert c1 != c2

    def test_equality_with_non_card(self):
        c1 = Card(Company.BEESWAX, 2)
        assert c1 != "not a card"

    def test_hash(self):
        c1 = Card(Company.BEESWAX, 2)
        c2 = Card(Company.BEESWAX, 2)
        assert hash(c1) == hash(c2)


class TestCardRepresentation:
    """Test Card string representations."""

    def test_repr(self):
        card = Card(Company.APPY_FIZZ, 3)
        assert "APPY_FIZZ" in repr(card)
        assert "3" in repr(card)

    def test_str(self):
        card = Card(Company.APPY_FIZZ, 3)
        assert str(card) == "A3"
