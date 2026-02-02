"""Tests for In a Grove Card module."""

import pytest

from games.in_a_grove.card import SuspectCard


class TestSuspectCardInit:
    """Test SuspectCard initialization."""

    @pytest.mark.parametrize("value", range(1, 9))
    def test_create_valid_card(self, value):
        card = SuspectCard(value)
        assert card.value == value

    @pytest.mark.parametrize("invalid_value", [0, -1, 9, 10])
    def test_invalid_value_raises(self, invalid_value):
        with pytest.raises(ValueError):
            SuspectCard(invalid_value)


class TestSuspectCardProperties:
    """Test SuspectCard properties."""

    def test_is_accomplice(self):
        card = SuspectCard(1)
        assert card.is_accomplice is True

    def test_not_accomplice(self):
        card = SuspectCard(5)
        assert card.is_accomplice is False


class TestSuspectCardComparison:
    """Test SuspectCard comparison methods."""

    def test_equality(self):
        c1 = SuspectCard(3)
        c2 = SuspectCard(3)
        assert c1 == c2

    def test_inequality(self):
        c1 = SuspectCard(3)
        c2 = SuspectCard(5)
        assert c1 != c2

    def test_equality_with_non_card(self):
        c1 = SuspectCard(3)
        assert c1 != "not a card"

    def test_less_than(self):
        c1 = SuspectCard(3)
        c2 = SuspectCard(5)
        assert c1 < c2

    def test_hash(self):
        c1 = SuspectCard(3)
        c2 = SuspectCard(3)
        assert hash(c1) == hash(c2)


class TestSuspectCardRepresentation:
    """Test SuspectCard string representations."""

    def test_repr(self):
        card = SuspectCard(5)
        assert "5" in repr(card)

    def test_str_accomplice(self):
        card = SuspectCard(1)
        assert str(card) == "A"

    def test_str_suspect(self):
        card = SuspectCard(5)
        assert str(card) == "5"
