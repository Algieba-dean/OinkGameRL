"""Hand type detection for Doudizhu (斗地主) game."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from games.doudizhu.card import Card
from games.doudizhu.constants import CardConsts
from games.doudizhu.enums import CardRank, HandType


@dataclass
class HandInfo:
    """Information about a hand of cards."""

    hand_type: HandType
    rank: int  # Primary rank for comparison (e.g., the triple's rank in 三带一)
    length: int  # Length for straights/chains

    def can_beat(self, other: HandInfo) -> bool:
        """Check if this hand can beat another hand."""
        # Rocket beats everything
        if self.hand_type == HandType.ROCKET:
            return True
        if other.hand_type == HandType.ROCKET:
            return False

        # Bomb beats non-bomb (except rocket)
        if self.hand_type == HandType.BOMB and other.hand_type != HandType.BOMB:
            return True
        if other.hand_type == HandType.BOMB and self.hand_type != HandType.BOMB:
            return False

        # Same type comparison
        if self.hand_type != other.hand_type:
            return False
        if self.length != other.length:
            return False

        return self.rank > other.rank


class HandDetector:
    """Detects the type of a hand of cards."""

    @staticmethod
    def detect(cards: list[Card]) -> HandInfo:
        """Detect the hand type of given cards."""
        if not cards:
            return HandInfo(HandType.PASS, -1, 0)

        n = len(cards)
        ranks = sorted([card.rank for card in cards])
        rank_counts = Counter(ranks)
        count_values = sorted(rank_counts.values(), reverse=True)

        # Rocket (双王)
        if n == 2 and CardRank.BLACK_JOKER in ranks and CardRank.RED_JOKER in ranks:
            return HandInfo(HandType.ROCKET, int(CardRank.RED_JOKER), 2)

        # Single (单张)
        if n == 1:
            return HandInfo(HandType.SINGLE, int(ranks[0]), 1)

        # Pair (对子)
        if n == 2 and count_values == [2]:
            return HandInfo(HandType.PAIR, int(ranks[0]), 1)

        # Triple (三张)
        if n == 3 and count_values == [3]:
            return HandInfo(HandType.TRIPLE, int(ranks[0]), 1)

        # Bomb (炸弹)
        if n == 4 and count_values == [4]:
            return HandInfo(HandType.BOMB, int(ranks[0]), 1)

        # Triple with single (三带一)
        if n == 4 and count_values == [3, 1]:
            triple_rank = HandDetector._get_rank_with_count(rank_counts, 3)
            return HandInfo(HandType.TRIPLE_WITH_SINGLE, triple_rank, 1)

        # Triple with pair (三带二)
        if n == 5 and count_values == [3, 2]:
            triple_rank = HandDetector._get_rank_with_count(rank_counts, 3)
            return HandInfo(HandType.TRIPLE_WITH_PAIR, triple_rank, 1)

        # Straight (顺子)
        if (
            n >= CardConsts.MIN_STRAIGHT_LENGTH
            and count_values == [1] * n
            and HandDetector._is_consecutive(ranks)
            and max(ranks) < CardRank.TWO
        ):
            return HandInfo(HandType.STRAIGHT, int(max(ranks)), n)

        # Straight pair (连对)
        if (
            n >= CardConsts.MIN_STRAIGHT_PAIR_LENGTH * 2
            and n % 2 == 0
            and all(c == 2 for c in count_values)
        ):
            unique_ranks = sorted(rank_counts.keys())
            if (
                HandDetector._is_consecutive(unique_ranks)
                and max(unique_ranks) < CardRank.TWO
            ):
                return HandInfo(HandType.STRAIGHT_PAIR, int(max(unique_ranks)), n // 2)

        # Airplane (飞机) and variants
        airplane_info = HandDetector._detect_airplane(cards, rank_counts, count_values)
        if airplane_info.hand_type != HandType.INVALID:
            return airplane_info

        # Four with two singles (四带二单)
        if n == 6 and count_values == [4, 1, 1]:
            four_rank = HandDetector._get_rank_with_count(rank_counts, 4)
            return HandInfo(HandType.FOUR_WITH_TWO_SINGLES, four_rank, 1)

        # Four with two pairs (四带二对)
        if n == 8 and count_values == [4, 2, 2]:
            four_rank = HandDetector._get_rank_with_count(rank_counts, 4)
            return HandInfo(HandType.FOUR_WITH_TWO_PAIRS, four_rank, 1)

        return HandInfo(HandType.INVALID, -1, 0)

    @staticmethod
    def _get_rank_with_count(rank_counts: Counter, count: int) -> int:
        """Get the rank that has the specified count."""
        for rank, cnt in rank_counts.items():
            if cnt == count:
                return int(rank)
        return -1

    @staticmethod
    def _is_consecutive(ranks: list[CardRank]) -> bool:
        """Check if ranks are consecutive."""
        if not ranks:
            return False
        return all(int(ranks[i]) - int(ranks[i - 1]) == 1 for i in range(1, len(ranks)))

    @staticmethod
    def _detect_airplane(
        cards: list[Card], rank_counts: Counter, count_values: list[int]
    ) -> HandInfo:
        """Detect airplane (飞机) patterns."""
        n = len(cards)

        # Find all triples
        triple_ranks = sorted(
            [rank for rank, count in rank_counts.items() if count >= 3]
        )

        if len(triple_ranks) < CardConsts.MIN_AIRPLANE_LENGTH:
            return HandInfo(HandType.INVALID, -1, 0)

        # Find longest consecutive triple sequence
        best_seq: list[CardRank] = []
        current_seq: list[CardRank] = []

        for rank in triple_ranks:
            if rank >= CardRank.TWO:  # 2 and jokers can't be in airplane
                continue
            if not current_seq or int(rank) == int(current_seq[-1]) + 1:
                current_seq.append(rank)
            else:
                if len(current_seq) > len(best_seq):
                    best_seq = current_seq
                current_seq = [rank]

        if len(current_seq) > len(best_seq):
            best_seq = current_seq

        if len(best_seq) < CardConsts.MIN_AIRPLANE_LENGTH:
            return HandInfo(HandType.INVALID, -1, 0)

        num_triples = len(best_seq)
        remaining = n - num_triples * 3

        # Pure airplane (飞机)
        if remaining == 0:
            return HandInfo(HandType.AIRPLANE, int(max(best_seq)), num_triples)

        # Airplane with singles (飞机带单)
        if remaining == num_triples:
            return HandInfo(
                HandType.AIRPLANE_WITH_SINGLES, int(max(best_seq)), num_triples
            )

        # Airplane with pairs (飞机带对)
        if remaining == num_triples * 2:
            # Simplified check: just verify total count matches for pairs
            return HandInfo(
                HandType.AIRPLANE_WITH_PAIRS, int(max(best_seq)), num_triples
            )

        return HandInfo(HandType.INVALID, -1, 0)
