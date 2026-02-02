"""Tests for Kobayakawa GameState module."""

import numpy as np
import pytest

from games.kobayakawa.card import Card
from games.kobayakawa.constants import PlayerConsts
from games.kobayakawa.enums import GamePhase
from games.kobayakawa.game_state import GameState


class TestGameStateInit:
    """Test GameState initialization."""

    def test_create_game_state(self):
        gs = GameState(player_num=4)
        assert gs.player_num == 4
        assert len(gs.players) == 4

    @pytest.mark.parametrize("player_num", PlayerConsts.ALLOWED_PLAYER_NUM)
    def test_valid_player_nums(self, player_num):
        gs = GameState(player_num=player_num)
        assert gs.player_num == player_num

    @pytest.mark.parametrize("invalid_num", [1, 2, 7, 0])
    def test_invalid_player_nums(self, invalid_num):
        with pytest.raises(ValueError, match="Invalid player num"):
            GameState(player_num=invalid_num)


class TestGameStateProperties:
    """Test GameState properties."""

    def test_initial_properties(self):
        gs = GameState(player_num=4)
        assert gs.current_player_idx == 0
        assert gs.kobayakawa_card is None
        assert gs.phase == GamePhase.DRAW_OR_SWAP
        assert gs.round == 1
        assert gs.pot == 0

    def test_get_player(self):
        gs = GameState(player_num=4)
        player = gs.get_player(2)
        assert player.player_idx == 2

    def test_get_current_player(self):
        gs = GameState(player_num=4)
        player = gs.get_current_player()
        assert player.player_idx == 0


class TestGameStateActions:
    """Test GameState action methods."""

    def test_set_kobayakawa(self):
        gs = GameState(player_num=4)
        card = Card(value=7)
        gs.set_kobayakawa(card)
        assert gs.kobayakawa_card == card

    def test_add_to_pot(self):
        gs = GameState(player_num=4)
        gs.add_to_pot(3)
        assert gs.pot == 3

    def test_next_player(self):
        gs = GameState(player_num=4)
        gs.next_player()
        assert gs.current_player_idx == 1

    def test_next_player_wraps(self):
        gs = GameState(player_num=4)
        for _ in range(4):
            gs.next_player()
        assert gs.current_player_idx == 0

    def test_next_player_skips_eliminated(self):
        gs = GameState(player_num=4)
        rng = np.random.default_rng(42)
        gs.reset(rng)
        # Eliminate player 1 by removing all coins
        p1 = gs.get_player(1)
        for _ in range(PlayerConsts.INITIAL_COINS):
            p1.place_bet()
        # Now p1 has 0 coins and is eliminated
        gs.next_player()
        assert gs.current_player_idx == 2

    def test_set_phase(self):
        gs = GameState(player_num=4)
        gs.set_phase(GamePhase.BETTING)
        assert gs.phase == GamePhase.BETTING


class TestStartNewRound:
    """Test start_new_round method."""

    def test_start_new_round(self):
        gs = GameState(player_num=4)
        rng = np.random.default_rng(42)
        gs.start_new_round(rng)
        assert gs.round == 2
        assert gs.pot == 0
        assert gs.phase == GamePhase.DRAW_OR_SWAP

    def test_start_new_round_deals_cards(self):
        gs = GameState(player_num=4)
        rng = np.random.default_rng(42)
        gs.start_new_round(rng)
        for player in gs.players:
            assert player.card is not None

    def test_start_new_round_sets_kobayakawa(self):
        gs = GameState(player_num=4)
        rng = np.random.default_rng(42)
        gs.start_new_round(rng)
        assert gs.kobayakawa_card is not None


class TestResolveShowdown:
    """Test resolve_showdown method."""

    def test_no_betting_players(self):
        gs = GameState(player_num=4)
        rng = np.random.default_rng(42)
        gs.reset(rng)
        winner = gs.resolve_showdown()
        assert winner is None

    def test_single_betting_player_wins(self):
        gs = GameState(player_num=4)
        rng = np.random.default_rng(42)
        gs.reset(rng)
        gs.get_player(0).place_bet()
        gs.add_to_pot(1)
        initial_coins = gs.get_player(0).coins
        winner = gs.resolve_showdown()
        assert winner == 0
        assert gs.get_player(0).coins == initial_coins + 1

    def test_lowest_card_gets_kobayakawa(self):
        gs = GameState(player_num=4)
        rng = np.random.default_rng(42)
        gs.reset(rng)
        # Player 0 has card 1, player 1 has card 10
        # Kobayakawa is 8, so player 0 gets 1+8=9, player 1 gets 10
        # Player 1 wins with higher value
        gs.get_player(0).set_card(Card(value=1))
        gs.get_player(1).set_card(Card(value=10))
        gs.set_kobayakawa(Card(value=8))
        gs.get_player(0).place_bet()
        gs.get_player(1).place_bet()
        gs.add_to_pot(2)
        winner = gs.resolve_showdown()
        assert winner == 1  # 10 > 1+8=9


class TestIsTerminated:
    """Test is_terminated property."""

    def test_not_terminated_initially(self):
        gs = GameState(player_num=4)
        assert gs.is_terminated is False

    def test_terminated_after_7_rounds(self):
        gs = GameState(player_num=4)
        rng = np.random.default_rng(42)
        for _ in range(8):
            gs.start_new_round(rng)
        assert gs.is_terminated is True

    def test_terminated_one_player_left(self):
        gs = GameState(player_num=3)
        rng = np.random.default_rng(42)
        gs.reset(rng)
        # Eliminate players 1 and 2 by using all their coins
        for p in gs.players[1:]:
            for _ in range(PlayerConsts.INITIAL_COINS):
                p.place_bet()
        assert gs.is_terminated is True


class TestGetWinner:
    """Test get_winner method."""

    def test_no_winner_when_not_terminated(self):
        gs = GameState(player_num=4)
        assert gs.get_winner() is None

    def test_winner_last_player_standing(self):
        gs = GameState(player_num=3)
        rng = np.random.default_rng(42)
        gs.reset(rng)
        # Eliminate players 1 and 2 by using all their coins
        for p in gs.players[1:]:
            for _ in range(PlayerConsts.INITIAL_COINS):
                p.place_bet()
        assert gs.get_winner() == 0

    def test_winner_most_coins(self):
        gs = GameState(player_num=4)
        rng = np.random.default_rng(42)
        for _ in range(8):
            gs.start_new_round(rng)
        gs.get_player(0).win_pot(10)
        winner = gs.get_winner()
        assert winner == 0


class TestReset:
    """Test reset method."""

    def test_reset_restores_initial_state(self):
        gs = GameState(player_num=4)
        rng = np.random.default_rng(42)
        gs.start_new_round(rng)
        gs.add_to_pot(5)
        gs.reset(rng)
        # After reset, start_new_round is called which sets round to 2
        # So we check the state is properly initialized
        assert gs.pot == 0
        assert gs.current_player_idx == 0
        assert gs.phase == GamePhase.DRAW_OR_SWAP


class TestStartNewRoundWithEliminated:
    """Test start_new_round with eliminated players."""

    def test_start_new_round_skips_eliminated_first_player(self):
        gs = GameState(player_num=3)
        rng = np.random.default_rng(42)
        gs.reset(rng)
        # Eliminate player 0
        p0 = gs.get_player(0)
        for _ in range(PlayerConsts.INITIAL_COINS):
            p0.place_bet()
        gs.start_new_round(rng)
        # First non-eliminated player should be current
        assert gs.current_player_idx == 1


class TestResolveShowdownEdgeCases:
    """Test edge cases in resolve_showdown."""

    def test_betting_player_with_no_card(self):
        """Test showdown when betting player has no card (line 144)."""
        gs = GameState(player_num=3)
        rng = np.random.default_rng(42)
        gs.reset(rng)
        p0 = gs.get_player(0)
        p0.place_bet()
        gs.add_to_pot(1)
        # Remove player's card
        p0._Player__card = None
        result = gs.resolve_showdown()
        assert result is None

    def test_all_betting_players_no_cards(self):
        """Test showdown when all betting players have no cards (line 151)."""
        gs = GameState(player_num=3)
        rng = np.random.default_rng(42)
        gs.reset(rng)
        for p in gs.players:
            p.place_bet()
            gs.add_to_pot(1)
            p._Player__card = None
        result = gs.resolve_showdown()
        assert result is None

    def test_betting_player_with_card_another_without(self):
        """Test showdown with mixed card states (covers line 144)."""
        gs = GameState(player_num=3)
        rng = np.random.default_rng(42)
        gs.reset(rng)
        # Player 0 bets with card
        gs.get_player(0).place_bet()
        gs.add_to_pot(1)
        # Player 1 bets but has no card
        gs.get_player(1).place_bet()
        gs.add_to_pot(1)
        gs.get_player(1)._Player__card = None
        result = gs.resolve_showdown()
        # Player 0 should win since player 1 has no card
        assert result == 0


class TestGetWinnerTie:
    """Test get_winner with tied scores."""

    def test_tie_returns_none(self):
        """Test that tied scores return None (line 169)."""
        gs = GameState(player_num=3)
        rng = np.random.default_rng(42)
        # Play 8 rounds to terminate
        for _ in range(8):
            gs.start_new_round(rng)
        # Ensure all players have same coins
        for p in gs.players:
            p._Player__coins = 10
        winner = gs.get_winner()
        assert winner is None
