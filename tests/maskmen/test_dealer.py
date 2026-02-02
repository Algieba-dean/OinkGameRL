"""Tests for Maskmen Dealer module."""

import numpy as np
import pytest

from games.maskmen.constants import GameConsts
from games.maskmen.dealer import Dealer
from games.maskmen.enums import CardColor


class TestDealerInit:
    """Test Dealer initialization."""

    def test_create_dealer_without_rng(self):
        dealer = Dealer()
        assert dealer is not None

    def test_create_dealer_with_rng(self):
        rng = np.random.default_rng(42)
        dealer = Dealer(random_generator=rng)
        assert dealer is not None


class TestCreateDeck:
    """Test deck creation."""

    def test_deck_size(self):
        dealer = Dealer()
        deck = dealer.create_deck()
        assert len(deck) == 18

    def test_deck_contains_all_colors(self):
        dealer = Dealer()
        deck = dealer.create_deck()
        colors = {card.color for card in deck}
        assert colors == set(CardColor)

    def test_deck_contains_all_values_per_color(self):
        dealer = Dealer()
        deck = dealer.create_deck()
        for color in CardColor:
            color_cards = [c for c in deck if c.color == color]
            values = {c.value for c in color_cards}
            assert values == {1, 2, 3}


class TestShuffleDeck:
    """Test deck shuffling."""

    def test_shuffle_maintains_size(self):
        dealer = Dealer(random_generator=np.random.default_rng(42))
        deck = dealer.create_deck()
        shuffled = dealer.shuffle_deck(deck)
        assert len(shuffled) == len(deck)

    def test_shuffle_maintains_cards(self):
        dealer = Dealer(random_generator=np.random.default_rng(42))
        deck = dealer.create_deck()
        shuffled = dealer.shuffle_deck(deck)
        assert set(deck) == set(shuffled)

    def test_shuffle_changes_order(self):
        dealer = Dealer(random_generator=np.random.default_rng(42))
        deck = dealer.create_deck()
        shuffled = dealer.shuffle_deck(deck)
        assert deck != shuffled

    def test_shuffle_deterministic_with_seed(self):
        dealer1 = Dealer(random_generator=np.random.default_rng(42))
        dealer2 = Dealer(random_generator=np.random.default_rng(42))
        deck = dealer1.create_deck()
        shuffled1 = dealer1.shuffle_deck(deck)
        shuffled2 = dealer2.shuffle_deck(deck)
        assert shuffled1 == shuffled2


class TestDealCards:
    """Test card dealing."""

    @pytest.mark.parametrize("player_num", [2, 3, 4, 5, 6])
    def test_deal_correct_hand_size(self, player_num):
        dealer = Dealer(random_generator=np.random.default_rng(42))
        hands, remaining = dealer.deal_cards(player_num)

        expected_hand_size = GameConsts.INITIAL_HAND_SIZE[player_num]
        for hand in hands:
            assert len(hand) == expected_hand_size

    @pytest.mark.parametrize("player_num", [2, 3, 4, 5, 6])
    def test_deal_correct_player_count(self, player_num):
        dealer = Dealer(random_generator=np.random.default_rng(42))
        hands, _ = dealer.deal_cards(player_num)
        assert len(hands) == player_num

    def test_deal_no_duplicate_cards(self):
        dealer = Dealer(random_generator=np.random.default_rng(42))
        hands, remaining = dealer.deal_cards(4)

        all_cards = []
        for hand in hands:
            all_cards.extend(hand)
        all_cards.extend(remaining)

        assert len(all_cards) == 18
        assert len(set(id(c) for c in all_cards)) == 18

    def test_deal_deterministic_with_seed(self):
        dealer1 = Dealer(random_generator=np.random.default_rng(42))
        dealer2 = Dealer(random_generator=np.random.default_rng(42))

        hands1, _ = dealer1.deal_cards(4)
        hands2, _ = dealer2.deal_cards(4)

        for h1, h2 in zip(hands1, hands2, strict=True):
            assert h1 == h2


class TestReset:
    """Test dealer reset."""

    def test_reset_changes_rng(self):
        dealer = Dealer(random_generator=np.random.default_rng(42))
        hands1, _ = dealer.deal_cards(4)

        dealer.reset(random_generator=np.random.default_rng(123))
        hands2, _ = dealer.deal_cards(4)

        assert hands1 != hands2
