"""Player module for In a Grove game."""

from __future__ import annotations

from games.in_a_grove.card import SuspectCard
from games.in_a_grove.enums import TileType


class Player:
    """Represents a player in In a Grove game."""

    def __init__(self, player_idx: int) -> None:
        self.__player_idx = player_idx
        self.__hand: list[SuspectCard] = []
        self.__tiles: list[TileType] = [
            TileType.CULPRIT,
            TileType.WITNESS,
            TileType.ACCOMPLICE,
        ]
        self.__score = 0
        self.__current_vote: TileType | None = None

    @property
    def player_idx(self) -> int:
        return self.__player_idx

    @property
    def hand(self) -> tuple[SuspectCard, ...]:
        return tuple(self.__hand)

    @property
    def hand_count(self) -> int:
        return len(self.__hand)

    @property
    def tiles(self) -> tuple[TileType, ...]:
        return tuple(self.__tiles)

    @property
    def score(self) -> int:
        return self.__score

    @property
    def current_vote(self) -> TileType | None:
        return self.__current_vote

    def set_hand(self, cards: list[SuspectCard]) -> None:
        """Set player's hand."""
        self.__hand = list(cards)

    def add_card(self, card: SuspectCard) -> None:
        """Add a card to hand."""
        self.__hand.append(card)

    def play_card(self, card_idx: int) -> SuspectCard:
        """Play a card from hand by index."""
        if card_idx < 0 or card_idx >= len(self.__hand):
            raise ValueError(f"Invalid card index: {card_idx}")
        return self.__hand.pop(card_idx)

    def vote(self, tile_type: TileType) -> None:
        """Cast a vote using a tile."""
        if tile_type not in self.__tiles:
            raise ValueError(f"Don't have tile: {tile_type}")
        self.__tiles.remove(tile_type)
        self.__current_vote = tile_type

    def clear_vote(self) -> None:
        """Clear current vote."""
        self.__current_vote = None

    def add_score(self, points: int) -> None:
        """Add points to score."""
        self.__score += points

    def reset_tiles(self) -> None:
        """Reset tiles for new round."""
        self.__tiles = [TileType.CULPRIT, TileType.WITNESS, TileType.ACCOMPLICE]

    def reset(self) -> None:
        """Reset player for new game."""
        self.__hand = []
        self.__tiles = [TileType.CULPRIT, TileType.WITNESS, TileType.ACCOMPLICE]
        self.__score = 0
        self.__current_vote = None
