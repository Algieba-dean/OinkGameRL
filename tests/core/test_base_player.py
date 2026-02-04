"""Tests for BasePlayer class."""

import pytest

from games.core.base_player import BasePlayer


class ConcretePlayer(BasePlayer[int]):
    """Concrete implementation for testing with default ascending sort."""

    def _sort_hand(self, pieces: list[int]) -> list[int]:
        return sorted(pieces)

    def reset(self) -> None:
        self._hand = []


class TestBasePlayer:
    """Test BasePlayer class."""

    @pytest.fixture
    def player(self) -> ConcretePlayer:
        return ConcretePlayer(0)

    def test_player_creation(self, player):
        assert player.player_idx == 0
        assert player.hand_count == 0
        assert player.hand == []

    def test_set_hand(self, player):
        player.set_hand([3, 1, 2])
        assert player.hand == [1, 2, 3]
        assert player.hand_count == 3

    def test_add_piece(self, player):
        player.set_hand([1, 3])
        player.add_piece(2)
        assert player.hand == [1, 2, 3]

    def test_add_pieces(self, player):
        player.set_hand([1])
        player.add_pieces([3, 2])
        assert player.hand == [1, 2, 3]

    def test_remove_piece_success(self, player):
        player.set_hand([1, 2, 3])
        assert player.remove_piece(2)
        assert player.hand == [1, 3]

    def test_remove_piece_not_found(self, player):
        player.set_hand([1, 2])
        assert not player.remove_piece(5)
        assert player.hand == [1, 2]

    def test_remove_pieces_success(self, player):
        player.set_hand([1, 2, 3, 4])
        assert player.remove_pieces([2, 4])
        assert player.hand == [1, 3]

    def test_remove_pieces_partial_fail(self, player):
        player.set_hand([1, 2, 3])
        assert not player.remove_pieces([2, 5])
        assert player.hand == [1, 2, 3]  # Unchanged

    def test_has_piece(self, player):
        player.set_hand([1, 2, 3])
        assert player.has_piece(2)
        assert not player.has_piece(5)

    def test_has_pieces(self, player):
        player.set_hand([1, 2, 3, 4])
        assert player.has_pieces([1, 3])
        assert not player.has_pieces([1, 5])

    def test_play_pieces(self, player):
        player.set_hand([1, 2, 3, 4])
        played = player.play_pieces([2, 4])
        assert played == [2, 4]
        assert player.hand == [1, 3]

    def test_play_pieces_partial(self, player):
        player.set_hand([1, 2, 3])
        played = player.play_pieces([2, 5])
        assert played == [2]
        assert player.hand == [1, 3]

    def test_reset(self, player):
        player.set_hand([1, 2, 3])
        player.reset()
        assert player.hand == []


class CustomSortPlayer(BasePlayer[int]):
    """Player with custom sorting (descending)."""

    def _sort_hand(self, pieces: list[int]) -> list[int]:
        return sorted(pieces, reverse=True)

    def reset(self) -> None:
        self._hand = []


class TestCustomSorting:
    """Test custom sorting functionality."""

    def test_custom_sort_descending(self):
        player = CustomSortPlayer(0)
        player.set_hand([1, 3, 2])
        assert player.hand == [3, 2, 1]

    def test_custom_sort_on_add(self):
        player = CustomSortPlayer(0)
        player.set_hand([3, 1])
        player.add_piece(2)
        assert player.hand == [3, 2, 1]

    def test_custom_sort_on_add_pieces(self):
        player = CustomSortPlayer(0)
        player.set_hand([5])
        player.add_pieces([1, 3])
        assert player.hand == [5, 3, 1]


class NoSortPlayer(BasePlayer[int]):
    """Player that preserves insertion order (no sorting)."""

    def _sort_hand(self, pieces: list[int]) -> list[int]:
        return pieces  # No sorting

    def reset(self) -> None:
        self._hand = []


class TestNoSorting:
    """Test player that doesn't sort."""

    def test_no_sort_preserves_order(self):
        player = NoSortPlayer(0)
        player.set_hand([3, 1, 2])
        assert player.hand == [3, 1, 2]

    def test_no_sort_on_add(self):
        player = NoSortPlayer(0)
        player.set_hand([3, 1])
        player.add_piece(2)
        assert player.hand == [3, 1, 2]
