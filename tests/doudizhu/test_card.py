"""Tests for Doudizhu card module."""

import pytest

from games.doudizhu.card import Card, create_full_deck
from games.doudizhu.enums import CardRank, CardSuit


class TestCard:
    """Test Card class."""

    def test_card_creation(self):
        card = Card(CardRank.THREE, CardSuit.SPADE)
        assert card.rank == CardRank.THREE
        assert card.suit == CardSuit.SPADE

    def test_card_str_regular(self):
        card = Card(CardRank.ACE, CardSuit.HEART)
        assert str(card) == "♥A"

    def test_card_str_ten(self):
        card = Card(CardRank.TEN, CardSuit.CLUB)
        assert str(card) == "♣10"

    def test_card_str_black_joker(self):
        card = Card(CardRank.BLACK_JOKER, CardSuit.JOKER)
        assert str(card) == "BJ"

    def test_card_str_red_joker(self):
        card = Card(CardRank.RED_JOKER, CardSuit.JOKER)
        assert str(card) == "RJ"

    def test_card_id_regular(self):
        card = Card(CardRank.THREE, CardSuit.SPADE)
        assert card.card_id == 0

        card = Card(CardRank.THREE, CardSuit.DIAMOND)
        assert card.card_id == 3

        card = Card(CardRank.TWO, CardSuit.DIAMOND)
        assert card.card_id == 51

    def test_card_id_jokers(self):
        black_joker = Card(CardRank.BLACK_JOKER, CardSuit.JOKER)
        assert black_joker.card_id == 52

        red_joker = Card(CardRank.RED_JOKER, CardSuit.JOKER)
        assert red_joker.card_id == 53

    def test_card_from_id(self):
        for card_id in range(54):
            card = Card.from_id(card_id)
            assert card.card_id == card_id

    def test_card_from_id_jokers(self):
        black_joker = Card.from_id(52)
        assert black_joker.rank == CardRank.BLACK_JOKER

        red_joker = Card.from_id(53)
        assert red_joker.rank == CardRank.RED_JOKER

    def test_card_ordering(self):
        card1 = Card(CardRank.THREE, CardSuit.SPADE)
        card2 = Card(CardRank.FOUR, CardSuit.SPADE)
        assert card1 < card2

    def test_card_equality(self):
        card1 = Card(CardRank.ACE, CardSuit.HEART)
        card2 = Card(CardRank.ACE, CardSuit.HEART)
        assert card1 == card2

    def test_card_frozen(self):
        card = Card(CardRank.THREE, CardSuit.SPADE)
        with pytest.raises(AttributeError):
            card.rank = CardRank.FOUR


class TestCreateFullDeck:
    """Test create_full_deck function."""

    def test_deck_size(self):
        deck = create_full_deck()
        assert len(deck) == 54

    def test_deck_contains_all_regular_cards(self):
        deck = create_full_deck()
        regular_cards = [c for c in deck if c.suit != CardSuit.JOKER]
        assert len(regular_cards) == 52

    def test_deck_contains_jokers(self):
        deck = create_full_deck()
        jokers = [c for c in deck if c.suit == CardSuit.JOKER]
        assert len(jokers) == 2

    def test_deck_unique_cards(self):
        deck = create_full_deck()
        card_ids = [c.card_id for c in deck]
        assert len(set(card_ids)) == 54

    def test_deck_all_suits(self):
        deck = create_full_deck()
        suits = set(c.suit for c in deck if c.suit != CardSuit.JOKER)
        assert len(suits) == 4

    def test_deck_all_ranks(self):
        deck = create_full_deck()
        ranks = set(c.rank for c in deck)
        assert CardRank.THREE in ranks
        assert CardRank.TWO in ranks
        assert CardRank.BLACK_JOKER in ranks
        assert CardRank.RED_JOKER in ranks
