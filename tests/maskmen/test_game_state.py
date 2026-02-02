"""Tests for Maskmen GameState module."""

import pytest

from games.maskmen.card import Card
from games.maskmen.constants import PlayerConsts
from games.maskmen.enums import CardColor
from games.maskmen.game_state import GameState


@pytest.fixture
def sample_hands_4p() -> list[list[Card]]:
    """4 players with 4 cards each."""
    return [
        [Card(CardColor.RED, v) for v in [1, 2, 3, 1]],
        [Card(CardColor.BLUE, v) for v in [1, 2, 3, 1]],
        [Card(CardColor.GREEN, v) for v in [1, 2, 3, 1]],
        [Card(CardColor.YELLOW, v) for v in [1, 2, 3, 1]],
    ]


@pytest.fixture
def sample_deck() -> list[Card]:
    return [Card(CardColor.PURPLE, v) for v in [1, 2, 3]]


class TestGameStateInit:
    """Test GameState initialization."""

    def test_create_game_state(self, sample_hands_4p, sample_deck):
        gs = GameState(player_num=4, player_hands=sample_hands_4p, deck=sample_deck)
        assert gs.player_num == 4
        assert len(gs.players) == 4

    @pytest.mark.parametrize("player_num", PlayerConsts.ALLOWED_PLAYER_NUM)
    def test_valid_player_nums(self, player_num, sample_deck):
        hands = [[Card(CardColor.RED, 1)] for _ in range(player_num)]
        gs = GameState(player_num=player_num, player_hands=hands, deck=sample_deck)
        assert gs.player_num == player_num

    @pytest.mark.parametrize("invalid_num", [1, 7, 0])
    def test_invalid_player_nums(self, invalid_num, sample_deck):
        hands = [[Card(CardColor.RED, 1)] for _ in range(invalid_num)]
        with pytest.raises(ValueError, match="Invalid player num"):
            GameState(player_num=invalid_num, player_hands=hands, deck=sample_deck)

    def test_mismatched_hands_count(self, sample_deck):
        hands = [[Card(CardColor.RED, 1)] for _ in range(3)]
        with pytest.raises(ValueError, match="Expected 4 hands"):
            GameState(player_num=4, player_hands=hands, deck=sample_deck)


class TestGameStateProperties:
    """Test GameState properties."""

    def test_current_player_starts_at_zero(self, sample_hands_4p, sample_deck):
        gs = GameState(player_num=4, player_hands=sample_hands_4p, deck=sample_deck)
        assert gs.current_player_idx == 0

    def test_deck_count(self, sample_hands_4p, sample_deck):
        gs = GameState(player_num=4, player_hands=sample_hands_4p, deck=sample_deck)
        assert gs.deck_count == 3

    def test_table_initially_empty(self, sample_hands_4p, sample_deck):
        gs = GameState(player_num=4, player_hands=sample_hands_4p, deck=sample_deck)
        for color in CardColor:
            assert gs.table[color] == ()


class TestNextPlayer:
    """Test next_player method."""

    def test_next_player_advances(self, sample_hands_4p, sample_deck):
        gs = GameState(player_num=4, player_hands=sample_hands_4p, deck=sample_deck)
        gs.next_player()
        assert gs.current_player_idx == 1

    def test_next_player_wraps_around(self, sample_hands_4p, sample_deck):
        gs = GameState(player_num=4, player_hands=sample_hands_4p, deck=sample_deck)
        for _ in range(4):
            gs.next_player()
        assert gs.current_player_idx == 0


class TestPlayCardToTable:
    """Test play_card_to_table method."""

    def test_play_card_adds_to_table(self, sample_hands_4p, sample_deck):
        gs = GameState(player_num=4, player_hands=sample_hands_4p, deck=sample_deck)
        card = Card(CardColor.RED, 1)
        gs.play_card_to_table(card)
        assert card in gs.table[CardColor.RED]

    def test_complete_set_collected(self, sample_hands_4p, sample_deck):
        gs = GameState(player_num=4, player_hands=sample_hands_4p, deck=sample_deck)
        gs.play_card_to_table(Card(CardColor.ORANGE, 1))
        gs.play_card_to_table(Card(CardColor.ORANGE, 2))
        gs.play_card_to_table(Card(CardColor.ORANGE, 3))

        assert gs.get_current_player().set_count == 1
        assert CardColor.ORANGE in gs.get_current_player().collected_sets
        assert gs.table[CardColor.ORANGE] == ()


class TestDrawCard:
    """Test draw_card method."""

    def test_draw_card_returns_card(self, sample_hands_4p, sample_deck):
        gs = GameState(player_num=4, player_hands=sample_hands_4p, deck=sample_deck)
        card = gs.draw_card()
        assert card is not None
        assert gs.deck_count == 2

    def test_draw_from_empty_deck(self, sample_hands_4p):
        gs = GameState(player_num=4, player_hands=sample_hands_4p, deck=[])
        card = gs.draw_card()
        assert card is None


class TestIsTerminated:
    """Test is_terminated property."""

    def test_not_terminated_initially(self, sample_hands_4p, sample_deck):
        gs = GameState(player_num=4, player_hands=sample_hands_4p, deck=sample_deck)
        assert gs.is_terminated is False

    def test_terminated_when_player_wins(self, sample_hands_4p, sample_deck):
        gs = GameState(player_num=4, player_hands=sample_hands_4p, deck=sample_deck)
        player = gs.get_current_player()
        player.collect_set(CardColor.RED)
        player.collect_set(CardColor.BLUE)
        player.collect_set(CardColor.GREEN)
        assert gs.is_terminated is True


class TestGetWinner:
    """Test get_winner method."""

    def test_no_winner_initially(self, sample_hands_4p, sample_deck):
        gs = GameState(player_num=4, player_hands=sample_hands_4p, deck=sample_deck)
        assert gs.get_winner() is None

    def test_winner_with_three_sets(self, sample_hands_4p, sample_deck):
        gs = GameState(player_num=4, player_hands=sample_hands_4p, deck=sample_deck)
        player = gs.get_current_player()
        player.collect_set(CardColor.RED)
        player.collect_set(CardColor.BLUE)
        player.collect_set(CardColor.GREEN)
        assert gs.get_winner() == 0


class TestReset:
    """Test reset method."""

    def test_reset_restores_state(self, sample_hands_4p, sample_deck):
        gs = GameState(player_num=4, player_hands=sample_hands_4p, deck=sample_deck)
        gs.next_player()
        gs.play_card_to_table(Card(CardColor.RED, 1))

        new_hands = [[Card(CardColor.YELLOW, 1)] for _ in range(4)]
        new_deck = [Card(CardColor.ORANGE, 1)]
        gs.reset(player_hands=new_hands, deck=new_deck)

        assert gs.current_player_idx == 0
        assert gs.deck_count == 1
        assert gs.table[CardColor.RED] == ()
