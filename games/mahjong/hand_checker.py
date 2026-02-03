"""Hand checking for Mahjong (麻将) game."""

from __future__ import annotations

from collections import Counter

from games.mahjong.meld import Meld
from games.mahjong.tile import Tile


class HandChecker:
    """Checks if a hand is a winning hand."""

    @staticmethod
    def is_winning_hand(hand: list[Tile], melds: list[Meld]) -> bool:
        """Check if hand + melds form a winning hand.

        A winning hand consists of 4 melds + 1 pair.
        Hand contains tiles not in melds.
        """
        # Count tiles needed in hand (14 - meld tiles)
        meld_tile_count = sum(len(m.tiles) for m in melds)
        expected_hand_size = 14 - meld_tile_count

        if len(hand) != expected_hand_size:
            return False

        # Try to form 4 melds + 1 pair from hand tiles
        return HandChecker._can_form_winning(hand)

    @staticmethod
    def _can_form_winning(tiles: list[Tile]) -> bool:
        """Check if tiles can form melds + pair."""
        if not tiles:
            return True

        # Count by tile type
        type_counts: Counter[int] = Counter(t.tile_type_id for t in tiles)

        # Try each tile type as the pair
        for pair_type in type_counts:
            if type_counts[pair_type] >= 2:
                remaining = type_counts.copy()
                remaining[pair_type] -= 2
                if remaining[pair_type] == 0:
                    del remaining[pair_type]
                if HandChecker._can_form_melds(remaining):
                    return True

        return False

    @staticmethod
    def _can_form_melds(counts: Counter[int]) -> bool:
        """Check if remaining tiles can all form melds (triplets or sequences)."""
        if not counts:
            return True

        # Get the smallest tile type
        min_type = min(counts.keys())
        count = counts[min_type]

        # Try triplet (刻子)
        if count >= 3:
            remaining = counts.copy()
            remaining[min_type] -= 3
            if remaining[min_type] == 0:
                del remaining[min_type]
            if HandChecker._can_form_melds(remaining):
                return True

        # Try sequence (顺子) - only for numbered suits (types 0-26)
        if min_type < 27 and min_type % 9 <= 6:  # Can form sequence
            next1 = min_type + 1
            next2 = min_type + 2
            # Check same suit (not crossing 9)
            if (
                next1 % 9 != 0
                and next2 % 9 != 0
                and counts.get(next1, 0) >= 1
                and counts.get(next2, 0) >= 1
            ):
                remaining = counts.copy()
                remaining[min_type] -= 1
                remaining[next1] -= 1
                remaining[next2] -= 1
                for t in [min_type, next1, next2]:
                    if remaining[t] == 0:
                        del remaining[t]
                if HandChecker._can_form_melds(remaining):
                    return True

        return False

    @staticmethod
    def can_chi(hand: list[Tile], discard: Tile) -> list[tuple[Tile, Tile]]:
        """Get possible chi combinations for a discarded tile.

        Returns list of (tile1, tile2) pairs from hand that can chi with discard.
        """
        if not discard.is_numbered():
            return []

        result: list[tuple[Tile, Tile]] = []
        suit = discard.suit
        rank = discard.rank

        # Group hand tiles by type
        hand_by_type: dict[int, list[Tile]] = {}
        for t in hand:
            if t.suit == suit:
                if t.rank not in hand_by_type:
                    hand_by_type[t.rank] = []
                hand_by_type[t.rank].append(t)

        # Check three patterns: (rank-2, rank-1), (rank-1, rank+1), (rank+1, rank+2)
        patterns = [
            (rank - 2, rank - 1),
            (rank - 1, rank + 1),
            (rank + 1, rank + 2),
        ]

        for r1, r2 in patterns:
            if (
                1 <= r1 <= 9
                and 1 <= r2 <= 9
                and r1 in hand_by_type
                and r2 in hand_by_type
            ):
                result.append((hand_by_type[r1][0], hand_by_type[r2][0]))

        return result

    @staticmethod
    def can_pong(hand: list[Tile], discard: Tile) -> bool:
        """Check if can pong the discarded tile."""
        count = sum(1 for t in hand if t.same_type(discard))
        return count >= 2

    @staticmethod
    def can_gang(hand: list[Tile], discard: Tile) -> bool:
        """Check if can gang (明杠) the discarded tile."""
        count = sum(1 for t in hand if t.same_type(discard))
        return count >= 3

    @staticmethod
    def can_an_gang(hand: list[Tile]) -> list[int]:
        """Get tile types that can form an_gang (暗杠) from hand."""
        type_counts: Counter[int] = Counter(t.tile_type_id for t in hand)
        return [t for t, c in type_counts.items() if c == 4]

    @staticmethod
    def is_tenpai(hand: list[Tile], melds: list[Meld]) -> list[int]:
        """Get list of tile types that would complete the hand (听牌).

        Returns list of tile_type_ids that would win.
        """
        winning_types: list[int] = []

        for type_id in range(34):
            # Create a hypothetical tile
            test_tile = Tile.from_type_id(type_id, 0)
            test_hand = hand + [test_tile]
            if HandChecker.is_winning_hand(test_hand, melds):
                winning_types.append(type_id)

        return winning_types
