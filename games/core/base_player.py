"""Base player class for all games."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BasePlayer[T](ABC):
    """Abstract base class for all game players.

    This class provides common functionality for managing a player's hand
    and basic operations like adding, removing, and checking cards/tiles.

    Type Parameters:
        T: The type of game piece (Card, Tile, etc.)
    """

    def __init__(self, player_idx: int) -> None:
        """Initialize a player.

        Args:
            player_idx: The player's index (0-based).
        """
        self._player_idx = player_idx
        self._hand: list[T] = []

    @property
    def player_idx(self) -> int:
        """Get the player's index."""
        return self._player_idx

    @property
    def hand(self) -> list[T]:
        """Get the player's hand."""
        return self._hand

    @property
    def hand_count(self) -> int:
        """Get the number of pieces in hand."""
        return len(self._hand)

    def set_hand(self, pieces: list[T]) -> None:
        """Set the player's hand.

        Args:
            pieces: List of game pieces to set as hand.
        """
        self._hand = self._sort_hand(pieces)

    def add_piece(self, piece: T) -> None:
        """Add a piece to hand.

        Args:
            piece: The piece to add.
        """
        self._hand.append(piece)
        self._hand = self._sort_hand(self._hand)

    def add_pieces(self, pieces: list[T]) -> None:
        """Add multiple pieces to hand.

        Args:
            pieces: List of pieces to add.
        """
        self._hand.extend(pieces)
        self._hand = self._sort_hand(self._hand)

    def remove_piece(self, piece: T) -> bool:
        """Remove a piece from hand.

        Args:
            piece: The piece to remove.

        Returns:
            True if piece was found and removed, False otherwise.
        """
        if piece in self._hand:
            self._hand.remove(piece)
            return True
        return False

    def remove_pieces(self, pieces: list[T]) -> bool:
        """Remove multiple pieces from hand.

        Args:
            pieces: List of pieces to remove.

        Returns:
            True if all pieces were found and removed, False otherwise.
        """
        hand_copy = self._hand.copy()
        for piece in pieces:
            if piece in hand_copy:
                hand_copy.remove(piece)
            else:
                return False
        self._hand = hand_copy
        return True

    def has_piece(self, piece: T) -> bool:
        """Check if player has a specific piece.

        Args:
            piece: The piece to check for.

        Returns:
            True if piece is in hand.
        """
        return piece in self._hand

    def has_pieces(self, pieces: list[T]) -> bool:
        """Check if player has all specified pieces.

        Args:
            pieces: List of pieces to check for.

        Returns:
            True if all pieces are in hand.
        """
        hand_copy = self._hand.copy()
        for piece in pieces:
            if piece in hand_copy:
                hand_copy.remove(piece)
            else:
                return False
        return True

    def play_pieces(self, pieces: list[T]) -> list[T]:
        """Remove and return specified pieces from hand.

        Args:
            pieces: List of pieces to play.

        Returns:
            List of pieces that were successfully removed.
        """
        played: list[T] = []
        for piece in pieces:
            if piece in self._hand:
                self._hand.remove(piece)
                played.append(piece)
        return played

    def _sort_hand(self, pieces: list[T]) -> list[T]:
        """Sort the hand. Override for custom sorting.

        Args:
            pieces: List of pieces to sort.

        Returns:
            Sorted list of pieces.
        """
        return sorted(pieces)  # type: ignore[type-var]

    @abstractmethod
    def reset(self) -> None:
        """Reset player state to initial values."""
        self._hand = []
