"""Tests for Guandan card module."""

import pytest

from games.guandan.card import Card, create_double_deck
from games.guandan.enums import CardRank, CardSuit


class TestCard:
    """Test Card class."""

    def test_card_creation(self):
        card = Card(CardRank.TWO, CardSuit.SPADE, 0)
        assert card.rank == CardRank.TWO
        assert card.suit == CardSuit.SPADE
        assert card.deck == 0

    def test_card_str_regular(self):
        card = Card(CardRank.ACE, CardSuit.HEART, 0)
        assert str(card) == "♥A"

    def test_card_str_ten(self):
        card = Card(CardRank.TEN, CardSuit.CLUB, 1)
        assert str(card) == "♣10"

    def test_card_str_black_joker(self):
        card = Card(CardRank.BLACK_JOKER, CardSuit.JOKER, 0)
        assert str(card) == "BJ0"

    def test_card_str_red_joker(self):
        card = Card(CardRank.RED_JOKER, CardSuit.JOKER, 1)
        assert str(card) == "RJ1"

    def test_card_id_deck0(self):
        card = Card(CardRank.TWO, CardSuit.SPADE, 0)
        assert card.card_id == 0

        card = Card(CardRank.ACE, CardSuit.DIAMOND, 0)
        assert card.card_id == 51

    def test_card_id_deck1(self):
        card = Card(CardRank.TWO, CardSuit.SPADE, 1)
        assert card.card_id == 54

        card = Card(CardRank.ACE, CardSuit.DIAMOND, 1)
        assert card.card_id == 105

    def test_card_id_jokers(self):
        bj0 = Card(CardRank.BLACK_JOKER, CardSuit.JOKER, 0)
        assert bj0.card_id == 52

        rj0 = Card(CardRank.RED_JOKER, CardSuit.JOKER, 0)
        assert rj0.card_id == 53

        bj1 = Card(CardRank.BLACK_JOKER, CardSuit.JOKER, 1)
        assert bj1.card_id == 106

        rj1 = Card(CardRank.RED_JOKER, CardSuit.JOKER, 1)
        assert rj1.card_id == 107

    def test_card_from_id(self):
        for card_id in range(108):
            card = Card.from_id(card_id)
            assert card.card_id == card_id

    def test_card_ordering(self):
        card1 = Card(CardRank.TWO, CardSuit.SPADE, 0)
        card2 = Card(CardRank.THREE, CardSuit.SPADE, 0)
        assert card1 < card2

    def test_card_frozen(self):
        card = Card(CardRank.TWO, CardSuit.SPADE, 0)
        with pytest.raises(AttributeError):
            card.rank = CardRank.THREE

    def test_effective_rank_regular(self):
        card = Card(CardRank.FIVE, CardSuit.SPADE, 0)
        assert card.get_effective_rank(CardRank.TWO) == CardRank.FIVE

    def test_effective_rank_level_card(self):
        card = Card(CardRank.TWO, CardSuit.SPADE, 0)
        # When level is 2, 2s become highest regular (98)
        assert card.get_effective_rank(CardRank.TWO) == 98

    def test_effective_rank_jokers(self):
        bj = Card(CardRank.BLACK_JOKER, CardSuit.JOKER, 0)
        rj = Card(CardRank.RED_JOKER, CardSuit.JOKER, 0)
        assert bj.get_effective_rank(CardRank.TWO) == 99
        assert rj.get_effective_rank(CardRank.TWO) == 100


class TestCreateDoubleDeck:
    """Test create_double_deck function."""

    def test_deck_size(self):
        deck = create_double_deck()
        assert len(deck) == 108

    def test_deck_contains_two_of_each(self):
        deck = create_double_deck()
        # Count cards by rank and suit (ignoring deck)
        from collections import Counter

        card_types = Counter((c.rank, c.suit) for c in deck)
        for count in card_types.values():
            assert count == 2

    def test_deck_contains_jokers(self):
        deck = create_double_deck()
        jokers = [c for c in deck if c.suit == CardSuit.JOKER]
        assert len(jokers) == 4  # 2 black + 2 red

    def test_deck_unique_card_ids(self):
        deck = create_double_deck()
        card_ids = [c.card_id for c in deck]
        assert len(set(card_ids)) == 108
