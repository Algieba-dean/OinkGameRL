import pytest

from games.scout.card.cards import Card
from games.scout.constants import BoardConsts


class TestGameStateContract:
    """Test GameState initialization and properties."""

    @pytest.fixture
    def player_cards(self, card_factory) -> list[list[Card]]:
        """Create cards for 4 players."""
        return [
            [card_factory(top=j, idx=i * 11 + j) for j in range(1, 12)]
            for i in range(4)
        ]

    @pytest.fixture
    def game_state(self, player_cards):
        from games.scout.game_status.game_state import GameState

        return GameState(player_num=4, player_cards=player_cards)

    def test_initialization(self, game_state):
        assert game_state.player_num == 4
        assert game_state.current_player_idx == 0
        assert len(game_state.players) == 4

    def test_immutable_player_num(self, game_state):
        with pytest.raises(AttributeError):
            game_state.player_num = 3

    def test_immutable_players(self, game_state):
        with pytest.raises(AttributeError):
            game_state.players = []

    @pytest.mark.parametrize(
        argnames="invalid_player_num",
        argvalues=[1, 6, 0, -1],
        ids=["one_player", "six_players", "zero_players", "negative_players"],
    )
    def test_invalid_player_num(self, invalid_player_num, card_factory):
        from games.scout.game_status.game_state import GameState

        player_cards = [
            [card_factory(top=j) for j in range(1, 12)]
            for _ in range(invalid_player_num)
        ]
        with pytest.raises(ValueError):
            GameState(player_num=invalid_player_num, player_cards=player_cards)


class TestGameStatePlayerManagement:
    """Test player turn management."""

    @pytest.fixture
    def player_cards(self, card_factory) -> list[list[Card]]:
        return [
            [card_factory(top=j, idx=i * 11 + j) for j in range(1, 12)]
            for i in range(4)
        ]

    @pytest.fixture
    def game_state(self, player_cards):
        from games.scout.game_status.game_state import GameState

        return GameState(player_num=4, player_cards=player_cards)

    def test_initial_current_player(self, game_state):
        assert game_state.current_player_idx == 0

    def test_next_player(self, game_state):
        game_state.next_player()
        assert game_state.current_player_idx == 1

    def test_next_player_wraps_around(self, game_state):
        for _ in range(4):
            game_state.next_player()
        assert game_state.current_player_idx == 0

    def test_get_player(self, game_state):
        player = game_state.get_player(0)
        assert player.player_idx == 0

    def test_get_current_player(self, game_state):
        current = game_state.get_current_player()
        assert current.player_idx == game_state.current_player_idx


class TestGameStateBoardIntegration:
    """Test board integration in game state."""

    @pytest.fixture
    def player_cards(self, card_factory) -> list[list[Card]]:
        return [
            [card_factory(top=j, idx=i * 11 + j) for j in range(1, 12)]
            for i in range(4)
        ]

    @pytest.fixture
    def game_state(self, player_cards):
        from games.scout.game_status.game_state import GameState

        return GameState(player_num=4, player_cards=player_cards)

    def test_initial_board_empty(self, game_state):
        assert game_state.board.cards == ()
        assert game_state.board.owner_idx == BoardConsts.EMPTY_OWNER_ID

    def test_board_accessible(self, game_state, card_factory):
        cards = [card_factory(top=5)]
        game_state.board.play_to_board(player_idx=0, played_cards=cards)
        assert game_state.board.owner_idx == 0


class TestGameStateScoreIntegration:
    """Test score integration in game state."""

    @pytest.fixture
    def player_cards(self, card_factory) -> list[list[Card]]:
        return [
            [card_factory(top=j, idx=i * 11 + j) for j in range(1, 12)]
            for i in range(4)
        ]

    @pytest.fixture
    def game_state(self, player_cards):
        from games.scout.game_status.game_state import GameState

        return GameState(player_num=4, player_cards=player_cards)

    def test_initial_scores_zero(self, game_state):
        for player_idx in range(4):
            assert game_state.score.score_dict[player_idx] == 0

    def test_score_increase(self, game_state):
        game_state.score.increase_score(player_idx=0, value=5)
        assert game_state.score.score_dict[0] == 5


class TestGameStateTermination:
    """Test game termination conditions."""

    @pytest.fixture
    def player_cards(self, card_factory) -> list[list[Card]]:
        return [
            [card_factory(top=j, idx=i * 11 + j) for j in range(1, 12)]
            for i in range(4)
        ]

    @pytest.fixture
    def game_state(self, player_cards):
        from games.scout.game_status.game_state import GameState

        return GameState(player_num=4, player_cards=player_cards)

    def test_initial_not_terminated(self, game_state):
        assert game_state.is_terminated is False

    def test_terminated_when_player_empty_hand(self, game_state, mocker):
        from games.scout.player.player import Player

        empty_player = mocker.MagicMock(spec=Player)
        empty_player.hand_count = 0
        mocker.patch.object(game_state, "_GameState__players", [empty_player] * 4)
        assert game_state.is_terminated is True


class TestGameStateReset:
    """Test game state reset functionality."""

    @pytest.fixture
    def player_cards(self, card_factory) -> list[list[Card]]:
        return [
            [card_factory(top=j, idx=i * 11 + j) for j in range(1, 12)]
            for i in range(4)
        ]

    @pytest.fixture
    def game_state(self, player_cards):
        from games.scout.game_status.game_state import GameState

        return GameState(player_num=4, player_cards=player_cards)

    def test_reset_state(self, game_state, card_factory):
        game_state.next_player()
        game_state.score.increase_score(player_idx=0, value=10)

        new_cards = [
            [card_factory(top=j, idx=i * 11 + j) for j in range(1, 12)]
            for i in range(4)
        ]
        game_state.reset(player_cards=new_cards)

        assert game_state.current_player_idx == 0
        assert game_state.score.score_dict[0] == 0
        assert game_state.board.cards == ()
