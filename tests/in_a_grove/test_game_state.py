"""Tests for In a Grove GameState module."""

import numpy as np
import pytest

from games.in_a_grove.enums import GamePhase, TileType
from games.in_a_grove.game_state import GameState


class TestGameStateInit:
    """Test GameState initialization."""

    def test_create_game_state(self):
        gs = GameState(player_num=4)
        assert gs.player_num == 4
        assert len(gs.players) == 4

    @pytest.mark.parametrize("invalid_num", [1, 5, 8])
    def test_invalid_player_num(self, invalid_num):
        with pytest.raises(ValueError):
            GameState(player_num=invalid_num)


class TestGameStateProperties:
    """Test GameState properties."""

    def test_current_player_idx(self):
        gs = GameState(player_num=4)
        assert gs.current_player_idx == 0

    def test_phase(self):
        gs = GameState(player_num=4)
        assert gs.phase == GamePhase.DEALING

    def test_round(self):
        gs = GameState(player_num=4)
        assert gs.round == 1


class TestGameStateReset:
    """Test reset functionality."""

    def test_reset(self):
        gs = GameState(player_num=4)
        rng = np.random.default_rng(42)
        gs.reset(rng)
        assert gs.round == 1
        assert gs.phase == GamePhase.VOTING


class TestGameStateRound:
    """Test round management."""

    def test_start_new_round(self):
        gs = GameState(player_num=4)
        rng = np.random.default_rng(42)
        gs.reset(rng)
        initial_round = gs.round
        gs.start_new_round(rng)
        assert gs.round == initial_round + 1


class TestGameStateTermination:
    """Test termination conditions."""

    def test_not_terminated_initially(self):
        gs = GameState(player_num=4)
        rng = np.random.default_rng(42)
        gs.reset(rng)
        assert gs.is_terminated is False

    def test_terminated_after_3_rounds(self):
        gs = GameState(player_num=4)
        rng = np.random.default_rng(42)
        gs.reset(rng)
        for _ in range(3):
            gs.start_new_round(rng)
        assert gs.is_terminated is True


class TestGameStateWinner:
    """Test winner determination."""

    def test_no_winner_when_not_terminated(self):
        gs = GameState(player_num=4)
        rng = np.random.default_rng(42)
        gs.reset(rng)
        assert gs.get_winner() is None

    def test_winner_highest_score(self):
        gs = GameState(player_num=3)
        rng = np.random.default_rng(42)
        gs.reset(rng)
        for _ in range(3):
            gs.start_new_round(rng)
        gs.get_player(0).add_score(100)
        assert gs.get_winner() == 0


class TestGameStateResolveRound:
    """Test round resolution."""

    def test_resolve_round_culprit_vote(self):
        gs = GameState(player_num=3)
        rng = np.random.default_rng(42)
        gs.reset(rng)
        p0 = gs.get_player(0)
        p0.vote(TileType.CULPRIT)
        gs.resolve_round()

    def test_resolve_round_no_center_card(self):
        gs = GameState(player_num=3)
        gs.resolve_round()


class TestGameStateNextPlayer:
    """Test next player functionality."""

    def test_next_player(self):
        gs = GameState(player_num=4)
        gs.next_player()
        assert gs.current_player_idx == 1

    def test_next_player_wraps(self):
        gs = GameState(player_num=3)
        for _ in range(3):
            gs.next_player()
        assert gs.current_player_idx == 0


class TestGameStateSetters:
    """Test setter methods."""

    def test_set_phase(self):
        gs = GameState(player_num=4)
        gs.set_phase(GamePhase.VOTING)
        assert gs.phase == GamePhase.VOTING

    def test_set_center_card(self):
        from games.in_a_grove.card import SuspectCard

        gs = GameState(player_num=4)
        card = SuspectCard(5)
        gs.set_center_card(card)
        assert gs.center_card == card


class TestResolveRoundScoring:
    """Test resolve_round scoring logic."""

    def test_culprit_vote_scores(self):
        """Test scoring when player votes culprit and has culprit card."""
        from games.in_a_grove.card import SuspectCard

        gs = GameState(player_num=3)
        rng = np.random.default_rng(42)
        gs.reset(rng)
        # Set center card as culprit
        culprit_card = SuspectCard(7)
        gs.set_center_card(culprit_card)
        # Give player 0 the culprit card and make them vote culprit
        p0 = gs.get_player(0)
        p0.set_hand([SuspectCard(7)])
        p0.vote(TileType.CULPRIT)
        initial_score = p0.score
        gs.resolve_round()
        assert p0.score == initial_score + 7

    def test_accomplice_vote_scores(self):
        """Test scoring when player votes accomplice and has accomplice card."""
        from games.in_a_grove.card import SuspectCard

        gs = GameState(player_num=3)
        rng = np.random.default_rng(42)
        gs.reset(rng)
        # Set center card
        gs.set_center_card(SuspectCard(5))
        # Give player 0 the accomplice card and make them vote accomplice
        p0 = gs.get_player(0)
        p0.set_hand([SuspectCard(1)])
        p0.vote(TileType.ACCOMPLICE)
        initial_score = p0.score
        gs.resolve_round()
        assert p0.score == initial_score + 1

    def test_witness_vote_scores(self):
        """Test scoring when player votes witness and has neither card."""
        from games.in_a_grove.card import SuspectCard

        gs = GameState(player_num=3)
        rng = np.random.default_rng(42)
        gs.reset(rng)
        # Set center card as 7
        gs.set_center_card(SuspectCard(7))
        # Give player 0 cards that are neither culprit (7) nor accomplice (1)
        p0 = gs.get_player(0)
        p0.set_hand([SuspectCard(3), SuspectCard(4)])
        p0.vote(TileType.WITNESS)
        initial_score = p0.score
        gs.resolve_round()
        assert p0.score == initial_score + 1


class TestGetWinnerTie:
    """Test winner with tied scores."""

    def test_tie_returns_none(self):
        gs = GameState(player_num=3)
        rng = np.random.default_rng(42)
        gs.reset(rng)
        for _ in range(3):
            gs.start_new_round(rng)
        # All players have same score (0)
        assert gs.get_winner() is None
