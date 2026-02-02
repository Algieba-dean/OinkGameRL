"""Tests for Maskmen Player module."""

import pytest

from games.maskmen.card import Card
from games.maskmen.enums import CardColor
from games.maskmen.player import Player


@pytest.fixture
def sample_hand() -> list[Card]:
    return [
        Card(CardColor.RED, 1),
        Card(CardColor.RED, 2),
        Card(CardColor.BLUE, 1),
        Card(CardColor.GREEN, 3),
    ]


class TestPlayerInit:
    """Test Player initialization."""

    def test_create_player(self, sample_hand):
        player = Player(player_idx=0, cards=sample_hand)
        assert player.player_idx == 0
        assert player.hand_count == 4

    def test_hand_is_copy(self, sample_hand):
        player = Player(player_idx=0, cards=sample_hand)
        sample_hand.append(Card(CardColor.YELLOW, 1))
        assert player.hand_count == 4


class TestPlayerProperties:
    """Test Player properties."""

    def test_player_idx_immutable(self, sample_hand):
        player = Player(player_idx=0, cards=sample_hand)
        assert player.player_idx == 0

    def test_hand_returns_tuple(self, sample_hand):
        player = Player(player_idx=0, cards=sample_hand)
        assert isinstance(player.hand, tuple)

    def test_collected_sets_initially_empty(self, sample_hand):
        player = Player(player_idx=0, cards=sample_hand)
        assert player.collected_sets == ()
        assert player.set_count == 0


class TestPlayCard:
    """Test play_card method."""

    def test_play_card_removes_from_hand(self, sample_hand):
        player = Player(player_idx=0, cards=sample_hand)
        card = player.play_card(0)
        assert player.hand_count == 3
        assert card not in player.hand

    def test_play_card_returns_correct_card(self, sample_hand):
        player = Player(player_idx=0, cards=sample_hand)
        expected = sample_hand[1]
        card = player.play_card(1)
        assert card == expected

    def test_play_card_invalid_index_negative(self, sample_hand):
        player = Player(player_idx=0, cards=sample_hand)
        with pytest.raises(ValueError, match="Invalid card index"):
            player.play_card(-1)

    def test_play_card_invalid_index_too_large(self, sample_hand):
        player = Player(player_idx=0, cards=sample_hand)
        with pytest.raises(ValueError, match="Invalid card index"):
            player.play_card(10)


class TestAddCard:
    """Test add_card method."""

    def test_add_card_increases_hand(self, sample_hand):
        player = Player(player_idx=0, cards=sample_hand)
        new_card = Card(CardColor.PURPLE, 2)
        player.add_card(new_card)
        assert player.hand_count == 5
        assert new_card in player.hand


class TestCollectSet:
    """Test collect_set method."""

    def test_collect_set_adds_color(self, sample_hand):
        player = Player(player_idx=0, cards=sample_hand)
        player.collect_set(CardColor.RED)
        assert CardColor.RED in player.collected_sets
        assert player.set_count == 1

    def test_collect_multiple_sets(self, sample_hand):
        player = Player(player_idx=0, cards=sample_hand)
        player.collect_set(CardColor.RED)
        player.collect_set(CardColor.BLUE)
        assert player.set_count == 2


class TestHasColor:
    """Test has_color method."""

    def test_has_color_true(self, sample_hand):
        player = Player(player_idx=0, cards=sample_hand)
        assert player.has_color(CardColor.RED) is True

    def test_has_color_false(self, sample_hand):
        player = Player(player_idx=0, cards=sample_hand)
        assert player.has_color(CardColor.PURPLE) is False


class TestGetCardsOfColor:
    """Test get_cards_of_color method."""

    def test_get_cards_of_color(self, sample_hand):
        player = Player(player_idx=0, cards=sample_hand)
        red_cards = player.get_cards_of_color(CardColor.RED)
        assert len(red_cards) == 2
        assert all(c.color == CardColor.RED for c in red_cards)

    def test_get_cards_of_missing_color(self, sample_hand):
        player = Player(player_idx=0, cards=sample_hand)
        purple_cards = player.get_cards_of_color(CardColor.PURPLE)
        assert len(purple_cards) == 0


class TestReset:
    """Test reset method."""

    def test_reset_replaces_hand(self, sample_hand):
        player = Player(player_idx=0, cards=sample_hand)
        new_cards = [Card(CardColor.YELLOW, 1), Card(CardColor.YELLOW, 2)]
        player.reset(new_cards)
        assert player.hand_count == 2

    def test_reset_clears_collected_sets(self, sample_hand):
        player = Player(player_idx=0, cards=sample_hand)
        player.collect_set(CardColor.RED)
        player.reset([Card(CardColor.YELLOW, 1)])
        assert player.set_count == 0
