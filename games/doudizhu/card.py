"""Card representation for Doudizhu (斗地主) game."""

from __future__ import annotations

from dataclasses import dataclass

from games.doudizhu.enums import CardRank, CardSuit


@dataclass(frozen=True, order=True)
class Card:
    """A single card in Doudizhu.

    Cards are ordered by rank for comparison.
    """

    rank: CardRank
    suit: CardSuit

    def __str__(self) -> str:
        if self.rank == CardRank.RED_JOKER:
            return "RJ"
        if self.rank == CardRank.BLACK_JOKER:
            return "BJ"

        rank_names = {
            CardRank.THREE: "3",
            CardRank.FOUR: "4",
            CardRank.FIVE: "5",
            CardRank.SIX: "6",
            CardRank.SEVEN: "7",
            CardRank.EIGHT: "8",
            CardRank.NINE: "9",
            CardRank.TEN: "10",
            CardRank.JACK: "J",
            CardRank.QUEEN: "Q",
            CardRank.KING: "K",
            CardRank.ACE: "A",
            CardRank.TWO: "2",
        }
        suit_symbols = {
            CardSuit.SPADE: "♠",
            CardSuit.HEART: "♥",
            CardSuit.CLUB: "♣",
            CardSuit.DIAMOND: "♦",
        }
        return f"{suit_symbols[self.suit]}{rank_names[self.rank]}"

    @property
    def card_id(self) -> int:
        """Get unique card ID (0-53)."""
        if self.rank == CardRank.BLACK_JOKER:
            return 52
        if self.rank == CardRank.RED_JOKER:
            return 53
        return int(self.rank) * 4 + int(self.suit)

    @classmethod
    def from_id(cls, card_id: int) -> Card:
        """Create card from unique ID."""
        if card_id == 52:
            return cls(CardRank.BLACK_JOKER, CardSuit.JOKER)
        if card_id == 53:
            return cls(CardRank.RED_JOKER, CardSuit.JOKER)
        rank = CardRank(card_id // 4)
        suit = CardSuit(card_id % 4)
        return cls(rank, suit)


def create_full_deck() -> list[Card]:
    """Create a full deck of 54 cards."""
    deck: list[Card] = []
    # Regular cards (3-2, 4 suits each)
    for rank in range(13):  # 0-12 (THREE to TWO)
        for suit in range(4):
            deck.append(Card(CardRank(rank), CardSuit(suit)))
    # Jokers
    deck.append(Card(CardRank.BLACK_JOKER, CardSuit.JOKER))
    deck.append(Card(CardRank.RED_JOKER, CardSuit.JOKER))
    return deck
