import pytest

from games.scout.card.cards import Card


class TestPlayerContract:
    """Test Player class contract and immutability."""

    @pytest.fixture
    def sample_cards(self, card_factory) -> list[Card]:
        return [card_factory(top=i) for i in range(1, 6)]

    @pytest.fixture
    def player(self, sample_cards):
        from games.scout.player.player import Player

        return Player(player_idx=0, cards=sample_cards)

    def test_player_initialization(self, player, sample_cards):
        assert player.player_idx == 0
        assert list(player.hand) == sample_cards
        assert player.scout_and_show_token is True

    def test_immutable_player_idx_property(self, player):
        with pytest.raises(
            AttributeError,
            match="property 'player_idx' of 'Player' object has no setter",
        ):
            player.player_idx = 1

    def test_immutable_hand_property(self, player):
        with pytest.raises(
            AttributeError,
            match="property 'hand' of 'Player' object has no setter",
        ):
            player.hand = []

    def test_immutable_hand_value(self, player, card_factory):
        with pytest.raises(
            TypeError, match="'tuple' object does not support item assignment"
        ):
            player.hand[0] = card_factory(top=10)

    def test_hand_count(self, player, sample_cards):
        assert player.hand_count == len(sample_cards)


class TestPlayerHandManagement:
    """Test Player hand operations."""

    @pytest.fixture
    def sample_cards(self, card_factory) -> list[Card]:
        return [card_factory(top=i, idx=i) for i in range(1, 6)]

    @pytest.fixture
    def player(self, sample_cards):
        from games.scout.player.player import Player

        return Player(player_idx=0, cards=sample_cards)

    def test_play_cards_single(self, player, card_factory):
        original_count = player.hand_count
        played = player.play_cards(start_idx=0, end_idx=0)
        assert len(played) == 1
        assert player.hand_count == original_count - 1

    def test_play_cards_multiple(self, player):
        original_count = player.hand_count
        played = player.play_cards(start_idx=1, end_idx=3)
        assert len(played) == 3
        assert player.hand_count == original_count - 3

    @pytest.mark.parametrize(
        argnames="start_idx,end_idx",
        argvalues=[(-1, 0), (0, 10), (3, 2)],
        ids=["negative_start", "end_out_of_range", "start_greater_than_end"],
    )
    def test_play_cards_invalid_indices(self, player, start_idx, end_idx):
        with pytest.raises(ValueError):
            player.play_cards(start_idx=start_idx, end_idx=end_idx)

    def test_insert_card_at_position(self, player, card_factory):
        new_card = card_factory(top=10, idx=100)
        original_count = player.hand_count
        player.insert_card(card=new_card, position=2)
        assert player.hand_count == original_count + 1
        assert player.hand[2] == new_card

    def test_insert_card_at_start(self, player, card_factory):
        new_card = card_factory(top=10, idx=100)
        player.insert_card(card=new_card, position=0)
        assert player.hand[0] == new_card

    def test_insert_card_at_end(self, player, card_factory):
        new_card = card_factory(top=10, idx=100)
        end_position = player.hand_count
        player.insert_card(card=new_card, position=end_position)
        assert player.hand[-1] == new_card

    @pytest.mark.parametrize(
        argnames="position",
        argvalues=[-1, 100],
        ids=["negative_position", "out_of_range"],
    )
    def test_insert_card_invalid_position(self, player, card_factory, position):
        new_card = card_factory(top=10, idx=100)
        with pytest.raises(ValueError):
            player.insert_card(card=new_card, position=position)


class TestPlayerScoutAndShowToken:
    """Test Scout and Show token management."""

    @pytest.fixture
    def sample_cards(self, card_factory) -> list[Card]:
        return [card_factory(top=i) for i in range(1, 6)]

    @pytest.fixture
    def player(self, sample_cards):
        from games.scout.player.player import Player

        return Player(player_idx=0, cards=sample_cards)

    def test_initial_token_state(self, player):
        assert player.scout_and_show_token is True

    def test_use_scout_and_show_token(self, player):
        player.use_scout_and_show_token()
        assert player.scout_and_show_token is False

    def test_use_token_twice_raises_error(self, player):
        player.use_scout_and_show_token()
        with pytest.raises(ValueError, match="Scout and Show token already used"):
            player.use_scout_and_show_token()

    def test_reset_token(self, player):
        player.use_scout_and_show_token()
        player.reset_token()
        assert player.scout_and_show_token is True


class TestPlayerReset:
    """Test Player reset functionality."""

    @pytest.fixture
    def sample_cards(self, card_factory) -> list[Card]:
        return [card_factory(top=i) for i in range(1, 6)]

    @pytest.fixture
    def player(self, sample_cards):
        from games.scout.player.player import Player

        return Player(player_idx=0, cards=sample_cards)

    def test_reset_with_new_cards(self, player, card_factory):
        new_cards = [card_factory(top=i, idx=i + 100) for i in range(1, 4)]
        player.use_scout_and_show_token()
        player.reset(cards=new_cards)

        assert player.hand_count == len(new_cards)
        assert list(player.hand) == new_cards
        assert player.scout_and_show_token is True
