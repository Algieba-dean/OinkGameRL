"""Tests for Kobayakawa Dealer module."""

import numpy as np

from games.kobayakawa.dealer import Dealer


class TestDealerInit:
    """Test Dealer initialization."""

    def test_create_dealer_without_rng(self):
        dealer = Dealer()
        assert dealer is not None

    def test_create_dealer_with_rng(self):
        rng = np.random.default_rng(42)
        dealer = Dealer(random_generator=rng)
        assert dealer is not None


class TestCreateAndShuffleDeck:
    """Test deck creation and shuffling."""

    def test_deck_size(self):
        dealer = Dealer(random_generator=np.random.default_rng(42))
        dealer.create_and_shuffle_deck()
        assert dealer.deck_count == 15

    def test_deck_contains_all_values(self):
        dealer = Dealer(random_generator=np.random.default_rng(42))
        dealer.create_and_shuffle_deck()
        cards = [dealer.deal_one() for _ in range(15)]
        values = {c.value for c in cards if c}
        assert values == set(range(1, 16))


class TestDealOne:
    """Test deal_one method."""

    def test_deal_one_returns_card(self):
        dealer = Dealer(random_generator=np.random.default_rng(42))
        dealer.create_and_shuffle_deck()
        card = dealer.deal_one()
        assert card is not None
        assert dealer.deck_count == 14

    def test_deal_from_empty_deck(self):
        dealer = Dealer(random_generator=np.random.default_rng(42))
        card = dealer.deal_one()
        assert card is None


class TestDealToPlayers:
    """Test deal_to_players method."""

    def test_deal_to_players(self):
        dealer = Dealer(random_generator=np.random.default_rng(42))
        dealer.create_and_shuffle_deck()
        cards = dealer.deal_to_players(4)
        assert len(cards) == 4
        assert dealer.deck_count == 11


class TestReset:
    """Test reset method."""

    def test_reset_clears_deck(self):
        dealer = Dealer(random_generator=np.random.default_rng(42))
        dealer.create_and_shuffle_deck()
        dealer.reset(random_generator=np.random.default_rng(123))
        assert dealer.deck_count == 0
