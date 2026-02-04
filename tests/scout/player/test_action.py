import pytest

from games.scout.enums import ScoutFlip, ScoutPosition
from games.scout.game_status.game_state import GameState, GameStates
from games.scout.player.action import Action, PlayAction, ScoutAction, ScoutPlayAction


@pytest.fixture
def game_state_fixture(card_factory) -> GameState:
    """Create a game state with 4 players, each having 11 cards."""
    player_cards = [
        [card_factory(top=j, idx=i * 11 + j) for j in range(1, 12)] for i in range(4)
    ]
    return GameState(player_num=4, player_cards=player_cards)


class TestActionContract:
    def empty_game_state(self):
        return GameStates()

    def dummy_play_action(self):
        return PlayAction(start_idx=0, end_idx=0)

    def dummy_scout_action(self):
        return ScoutAction(ScoutPosition.LEFT, 0, scout_flip=ScoutFlip.NO)

    def dummy_scout_play_action(self):
        return ScoutPlayAction(
            scout_position=ScoutPosition.LEFT,
            insert_position=0,
            scout_flip=ScoutFlip.NO,
            play_start_idx=0,
            play_end_idx=0,
        )

    def test_base_action_execute(self):
        with pytest.raises(
            NotImplementedError, match="execute is not supported on base action"
        ):
            Action().execute(game_state=self.empty_game_state())

    def test_base_action_is_valid(self):
        with pytest.raises(
            NotImplementedError, match="is_valid is not supported on base action"
        ):
            Action().is_valid(game_state=self.empty_game_state())

    def test_is_play_action_an_action(self):
        assert isinstance(self.dummy_play_action(), Action)

    def test_is_scout_action_an_action(self):
        assert isinstance(self.dummy_scout_action(), Action)

    def test_is_scout_play_action_an_action(self):
        assert isinstance(self.dummy_scout_play_action(), Action)


class TestPlayAction:
    """Test PlayAction functionality."""

    def test_play_action_properties(self):
        action = PlayAction(start_idx=1, end_idx=3)
        assert action.start_idx == 1
        assert action.end_idx == 3

    def test_play_action_is_valid_empty_board(self, game_state_fixture, card_factory):
        """Playing any valid pattern to empty board should be valid."""
        action = PlayAction(start_idx=0, end_idx=0)
        assert action.is_valid(game_state=game_state_fixture) is True

    def test_play_action_execute(self, game_state_fixture):
        """Execute should move cards from hand to board."""
        player = game_state_fixture.get_current_player()
        original_hand_count = player.hand_count
        action = PlayAction(start_idx=0, end_idx=0)

        action.execute(game_state=game_state_fixture)

        assert player.hand_count == original_hand_count - 1
        assert len(game_state_fixture.board.cards) == 1
        assert game_state_fixture.board.owner_idx == 0

    def test_play_action_invalid_indices(self, game_state_fixture):
        """Invalid card indices should make action invalid."""
        action = PlayAction(start_idx=100, end_idx=100)
        assert action.is_valid(game_state=game_state_fixture) is False


class TestScoutAction:
    """Test ScoutAction functionality."""

    def test_scout_action_properties(self):
        action = ScoutAction(
            scout_position=ScoutPosition.LEFT,
            insert_position=2,
            scout_flip=ScoutFlip.YES,
        )
        assert action.scout_position == ScoutPosition.LEFT
        assert action.insert_position == 2
        assert action.scout_flip == ScoutFlip.YES

    def test_scout_action_is_valid_with_cards_on_board(
        self, game_state_fixture, card_factory
    ):
        """Scout is valid when there are cards on board."""
        game_state_fixture.board.play_to_board(
            player_idx=1, played_cards=[card_factory(top=5)]
        )
        action = ScoutAction(
            scout_position=ScoutPosition.LEFT,
            insert_position=0,
            scout_flip=ScoutFlip.NO,
        )
        assert action.is_valid(game_state=game_state_fixture) is True

    def test_scout_action_invalid_empty_board(self, game_state_fixture):
        """Scout is invalid when board is empty."""
        action = ScoutAction(
            scout_position=ScoutPosition.LEFT,
            insert_position=0,
            scout_flip=ScoutFlip.NO,
        )
        assert action.is_valid(game_state=game_state_fixture) is False

    def test_scout_action_execute(self, game_state_fixture, card_factory):
        """Execute should take card from board and add to hand."""
        board_card = card_factory(top=8, idx=100)
        game_state_fixture.board.play_to_board(player_idx=1, played_cards=[board_card])
        player = game_state_fixture.get_current_player()
        original_hand_count = player.hand_count

        action = ScoutAction(
            scout_position=ScoutPosition.LEFT,
            insert_position=0,
            scout_flip=ScoutFlip.NO,
        )
        action.execute(game_state=game_state_fixture)

        assert player.hand_count == original_hand_count + 1
        assert player.hand[0].idx == board_card.idx

    def test_scout_action_execute_with_flip(self, game_state_fixture, card_factory):
        """Execute with flip should flip the card."""
        board_card = card_factory(top=8, bottom=3, idx=100)
        game_state_fixture.board.play_to_board(player_idx=1, played_cards=[board_card])

        action = ScoutAction(
            scout_position=ScoutPosition.LEFT,
            insert_position=0,
            scout_flip=ScoutFlip.YES,
        )
        action.execute(game_state=game_state_fixture)

        player = game_state_fixture.get_current_player()
        assert player.hand[0].top == 3
        assert player.hand[0].bottom == 8

    def test_scout_action_increases_board_owner_score(
        self, game_state_fixture, card_factory
    ):
        """Scout should increase the board owner's score."""
        game_state_fixture.board.play_to_board(
            player_idx=1, played_cards=[card_factory(top=5), card_factory(top=6)]
        )
        action = ScoutAction(
            scout_position=ScoutPosition.LEFT,
            insert_position=0,
            scout_flip=ScoutFlip.NO,
        )
        action.execute(game_state=game_state_fixture)

        assert game_state_fixture.score.score_dict[1] == 1

    @pytest.mark.parametrize(
        argnames="scout_position", argvalues=[ScoutPosition.LEFT, ScoutPosition.RIGHT]
    )
    @pytest.mark.parametrize(
        argnames="scout_flip", argvalues=[ScoutFlip.NO, ScoutFlip.YES]
    )
    @pytest.mark.parametrize(argnames="insert_position", argvalues=[0, 5])
    def test_scout_combinations(
        self,
        game_state_fixture,
        card_factory,
        scout_position: ScoutPosition,
        insert_position: int,
        scout_flip: ScoutFlip,
    ) -> None:
        """Test various scout parameter combinations."""
        game_state_fixture.board.play_to_board(
            player_idx=1, played_cards=[card_factory(top=5), card_factory(top=6)]
        )
        action = ScoutAction(
            scout_position=scout_position,
            insert_position=insert_position,
            scout_flip=scout_flip,
        )
        assert action.is_valid(game_state=game_state_fixture) is True


class TestScoutPlayAction:
    """Test ScoutPlayAction functionality."""

    def test_scout_play_action_properties(self):
        action = ScoutPlayAction(
            scout_position=ScoutPosition.RIGHT,
            insert_position=1,
            scout_flip=ScoutFlip.YES,
            play_start_idx=0,
            play_end_idx=2,
        )
        assert action.scout_position == ScoutPosition.RIGHT
        assert action.insert_position == 1
        assert action.scout_flip == ScoutFlip.YES
        assert action.play_start_idx == 0
        assert action.play_end_idx == 2

    def test_scout_play_requires_token(self, game_state_fixture, card_factory):
        """ScoutPlay requires the Scout-and-Show token."""
        game_state_fixture.board.play_to_board(
            player_idx=1, played_cards=[card_factory(top=2)]
        )
        action = ScoutPlayAction(
            scout_position=ScoutPosition.LEFT,
            insert_position=0,
            scout_flip=ScoutFlip.NO,
            play_start_idx=0,
            play_end_idx=0,
        )
        assert action.is_valid(game_state=game_state_fixture) is True

        game_state_fixture.get_current_player().use_scout_and_show_token()
        assert action.is_valid(game_state=game_state_fixture) is False

    def test_scout_play_execute(self, game_state_fixture, card_factory):
        """Execute should scout then play cards."""
        board_card = card_factory(top=2, idx=100)
        game_state_fixture.board.play_to_board(player_idx=1, played_cards=[board_card])
        player = game_state_fixture.get_current_player()
        original_hand_count = player.hand_count

        action = ScoutPlayAction(
            scout_position=ScoutPosition.LEFT,
            insert_position=0,
            scout_flip=ScoutFlip.NO,
            play_start_idx=1,
            play_end_idx=1,
        )
        action.execute(game_state=game_state_fixture)

        assert player.hand_count == original_hand_count
        assert player.scout_and_show_token is False
        assert game_state_fixture.board.owner_idx == 0


class TestPlayActionEdgeCases:
    """Test edge cases for PlayAction."""

    def test_play_action_start_greater_than_end(self, game_state_fixture):
        """Test that start_idx > end_idx is invalid."""
        action = PlayAction(start_idx=5, end_idx=2)
        assert action.is_valid(game_state=game_state_fixture) is False

    def test_play_action_execute_not_game_state(self):
        """Test that execute raises error for non-GameState."""
        action = PlayAction(start_idx=0, end_idx=0)
        with pytest.raises(NotImplementedError):
            action.execute(game_state=GameStates())


class TestScoutActionEdgeCases:
    """Test edge cases for ScoutAction."""

    def test_scout_action_execute_not_game_state(self):
        """Test that execute raises error for non-GameState."""
        action = ScoutAction(
            scout_position=ScoutPosition.LEFT,
            insert_position=0,
            scout_flip=ScoutFlip.NO,
        )
        with pytest.raises(NotImplementedError):
            action.execute(game_state=GameStates())


class TestScoutPlayActionEdgeCases:
    """Test edge cases for ScoutPlayAction."""

    def test_scout_play_action_execute_not_game_state(self):
        """Test that execute raises error for non-GameState."""
        action = ScoutPlayAction(
            scout_position=ScoutPosition.LEFT,
            insert_position=0,
            scout_flip=ScoutFlip.NO,
            play_start_idx=0,
            play_end_idx=0,
        )
        with pytest.raises(NotImplementedError):
            action.execute(game_state=GameStates())
