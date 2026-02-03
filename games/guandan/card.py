"""Card representation for Guandan (掼蛋) game."""

from __future__ import annotations

from dataclasses import dataclass

from games.guandan.enums import CardRank, CardSuit


@dataclass(frozen=True)
class Card:
    """A single card in Guandan.

    Cards are ordered by rank for comparison.
    Deck index (0 or 1) distinguishes cards from the two decks.
    """

    rank: CardRank
    suit: CardSuit
    deck: int = 0  # 0 or 1, indicating which deck

    def __lt__(self, other: Card) -> bool:
        if self.rank != other.rank:
            return self.rank < other.rank
        if self.suit != other.suit:
            return self.suit < other.suit
        return self.deck < other.deck

    def __str__(self) -> str:
        if self.rank == CardRank.RED_JOKER:
            return f"RJ{self.deck}"
        if self.rank == CardRank.BLACK_JOKER:
            return f"BJ{self.deck}"

        rank_names = {
            CardRank.TWO: "2",
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
        """Get unique card ID (0-107).

        Layout: deck0 (0-53) + deck1 (54-107)
        Within each deck: rank * 4 + suit for regular, 52/53 for jokers
        """
        base = self.deck * 54
        if self.rank == CardRank.BLACK_JOKER:
            return base + 52
        if self.rank == CardRank.RED_JOKER:
            return base + 53
        return base + int(self.rank) * 4 + int(self.suit)

    @classmethod
    def from_id(cls, card_id: int) -> Card:
        """Create card from unique ID."""
        deck = card_id // 54
        local_id = card_id % 54
        if local_id == 52:
            return cls(CardRank.BLACK_JOKER, CardSuit.JOKER, deck)
        if local_id == 53:
            return cls(CardRank.RED_JOKER, CardSuit.JOKER, deck)
        rank = CardRank(local_id // 4)
        suit = CardSuit(local_id % 4)
        return cls(rank, suit, deck)

    def get_effective_rank(self, level_rank: CardRank) -> int:
        """Get effective rank considering level card (级牌).

        In Guandan, the level card (e.g., 2 when playing 2s) becomes
        the highest regular card, just below jokers.
        """
        if self.rank == CardRank.RED_JOKER:
            return 100
        if self.rank == CardRank.BLACK_JOKER:
            return 99
        if self.rank == level_rank:
            return 98  # Level card is highest regular
        return int(self.rank)


def create_double_deck() -> list[Card]:
    """Create a double deck of 108 cards."""
    deck: list[Card] = []
    for deck_idx in range(2):
        # Regular cards (2-A, 4 suits each)
        for rank in range(13):  # 0-12 (TWO to ACE)
            for suit in range(4):
                deck.append(Card(CardRank(rank), CardSuit(suit), deck_idx))
        # Jokers
        deck.append(Card(CardRank.BLACK_JOKER, CardSuit.JOKER, deck_idx))
        deck.append(Card(CardRank.RED_JOKER, CardSuit.JOKER, deck_idx))
    return deck
