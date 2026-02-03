"""Hand type detection for Guandan (掼蛋) game."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from games.guandan.card import Card
from games.guandan.constants import CardConsts
from games.guandan.enums import CardRank, HandType


@dataclass
class HandInfo:
    """Information about a hand of cards."""

    hand_type: HandType
    rank: int  # Primary rank for comparison
    length: int  # Length for straights/chains

    def can_beat(self, other: HandInfo) -> bool:
        """Check if this hand can beat another hand."""
        # Rocket beats everything
        if self.hand_type == HandType.ROCKET:
            return True
        if other.hand_type == HandType.ROCKET:
            return False

        # Bombs beat non-bombs
        if self._is_bomb() and not other._is_bomb():
            return True
        if other._is_bomb() and not self._is_bomb():
            return False

        # Compare bombs by size then rank
        if self._is_bomb() and other._is_bomb():
            if self.hand_type != other.hand_type:
                return self.hand_type > other.hand_type
            return self.rank > other.rank

        # Same type comparison
        if self.hand_type != other.hand_type:
            return False
        if self.length != other.length:
            return False

        return self.rank > other.rank

    def _is_bomb(self) -> bool:
        """Check if this hand is a bomb."""
        return self.hand_type in (
            HandType.BOMB_4,
            HandType.BOMB_5,
            HandType.BOMB_6,
            HandType.BOMB_7,
            HandType.BOMB_8,
            HandType.STRAIGHT_FLUSH,
        )


class HandDetector:
    """Detects the type of a hand of cards."""

    @staticmethod
    def detect(cards: list[Card], level_rank: CardRank = CardRank.TWO) -> HandInfo:
        """Detect the hand type of given cards.

        Args:
            cards: List of cards to detect
            level_rank: Current level rank (级牌), affects wild card behavior
        """
        if not cards:
            return HandInfo(HandType.PASS, -1, 0)

        n = len(cards)
        # Use effective ranks for comparison
        ranks = sorted([card.get_effective_rank(level_rank) for card in cards])
        rank_counts = Counter(ranks)
        count_values = sorted(rank_counts.values(), reverse=True)

        # Rocket (四王)
        joker_count = sum(
            1 for c in cards if c.rank in (CardRank.BLACK_JOKER, CardRank.RED_JOKER)
        )
        if joker_count == 4:
            return HandInfo(HandType.ROCKET, 100, 4)

        # Single (单张)
        if n == 1:
            return HandInfo(HandType.SINGLE, ranks[0], 1)

        # Pair (对子)
        if n == 2 and count_values == [2]:
            return HandInfo(HandType.PAIR, ranks[0], 1)

        # Triple (三张)
        if n == 3 and count_values == [3]:
            return HandInfo(HandType.TRIPLE, ranks[0], 1)

        # Bombs (炸弹) - 4 to 8 of same rank
        if len(rank_counts) == 1:
            rank = ranks[0]
            if n == 4:
                return HandInfo(HandType.BOMB_4, rank, 1)
            if n == 5:
                return HandInfo(HandType.BOMB_5, rank, 1)
            if n == 6:
                return HandInfo(HandType.BOMB_6, rank, 1)
            if n == 7:
                return HandInfo(HandType.BOMB_7, rank, 1)
            if n == 8:
                return HandInfo(HandType.BOMB_8, rank, 1)

        # Triple with two (三带二)
        if n == 5 and count_values == [3, 2]:
            triple_rank = HandDetector._get_rank_with_count(rank_counts, 3)
            return HandInfo(HandType.TRIPLE_WITH_TWO, triple_rank, 1)

        # Straight (顺子) - at least 5 consecutive singles
        if (
            n >= CardConsts.MIN_STRAIGHT_LENGTH
            and count_values == [1] * n
            and HandDetector._is_consecutive(ranks)
        ):
            # Check for straight flush
            if HandDetector._is_same_suit(cards):
                return HandInfo(HandType.STRAIGHT_FLUSH, max(ranks), n)
            return HandInfo(HandType.STRAIGHT, max(ranks), n)

        # Tube/连对 (at least 3 consecutive pairs)
        if (
            n >= CardConsts.MIN_TUBE_LENGTH * 2
            and n % 2 == 0
            and all(c == 2 for c in count_values)
        ):
            unique_ranks = sorted(rank_counts.keys())
            if HandDetector._is_consecutive(unique_ranks):
                return HandInfo(HandType.TUBE, max(unique_ranks), n // 2)

        # Plate/板子 (at least 2 consecutive triples)
        if (
            n >= CardConsts.MIN_PLATE_LENGTH * 3
            and n % 3 == 0
            and all(c == 3 for c in count_values)
        ):
            unique_ranks = sorted(rank_counts.keys())
            if HandDetector._is_consecutive(unique_ranks):
                return HandInfo(HandType.PLATE, max(unique_ranks), n // 3)

        return HandInfo(HandType.INVALID, -1, 0)

    @staticmethod
    def _get_rank_with_count(rank_counts: Counter, count: int) -> int:
        """Get the rank that has the specified count."""
        for rank, cnt in rank_counts.items():
            if cnt == count:
                return int(rank)
        return -1

    @staticmethod
    def _is_consecutive(ranks: list[int]) -> bool:
        """Check if ranks are consecutive."""
        if not ranks:
            return False
        # Level cards (98) and jokers (99, 100) can't be in straights
        if any(r >= 98 for r in ranks):
            return False
        return all(ranks[i] - ranks[i - 1] == 1 for i in range(1, len(ranks)))

    @staticmethod
    def _is_same_suit(cards: list[Card]) -> bool:
        """Check if all cards have the same suit."""
        if not cards:
            return False
        suits = set(c.suit for c in cards)
        return len(suits) == 1
