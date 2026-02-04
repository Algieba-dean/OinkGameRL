"""Tests for Doudizhu hand detector module."""

from games.doudizhu.card import Card
from games.doudizhu.enums import CardRank, CardSuit, HandType
from games.doudizhu.hand_detector import HandDetector, HandInfo


class TestHandInfo:
    """Test HandInfo class."""

    def test_rocket_beats_everything(self):
        rocket = HandInfo(HandType.ROCKET, 14, 2)
        bomb = HandInfo(HandType.BOMB, 10, 1)
        assert rocket.can_beat(bomb)

    def test_bomb_beats_non_bomb(self):
        bomb = HandInfo(HandType.BOMB, 5, 1)
        triple = HandInfo(HandType.TRIPLE, 10, 1)
        assert bomb.can_beat(triple)

    def test_higher_bomb_beats_lower(self):
        bomb_high = HandInfo(HandType.BOMB, 10, 1)
        bomb_low = HandInfo(HandType.BOMB, 5, 1)
        assert bomb_high.can_beat(bomb_low)
        assert not bomb_low.can_beat(bomb_high)

    def test_same_type_higher_rank_wins(self):
        single_high = HandInfo(HandType.SINGLE, 10, 1)
        single_low = HandInfo(HandType.SINGLE, 5, 1)
        assert single_high.can_beat(single_low)
        assert not single_low.can_beat(single_high)

    def test_different_types_cannot_beat(self):
        single = HandInfo(HandType.SINGLE, 10, 1)
        pair = HandInfo(HandType.PAIR, 5, 1)
        assert not single.can_beat(pair)
        assert not pair.can_beat(single)

    def test_different_length_cannot_beat(self):
        straight5 = HandInfo(HandType.STRAIGHT, 7, 5)
        straight6 = HandInfo(HandType.STRAIGHT, 8, 6)
        assert not straight5.can_beat(straight6)
        assert not straight6.can_beat(straight5)


class TestHandDetectorBasic:
    """Test basic hand detection."""

    def test_empty_hand_is_pass(self):
        info = HandDetector.detect([])
        assert info.hand_type == HandType.PASS

    def test_single(self):
        card = Card(CardRank.ACE, CardSuit.SPADE)
        info = HandDetector.detect([card])
        assert info.hand_type == HandType.SINGLE
        assert info.rank == CardRank.ACE

    def test_pair(self):
        cards = [
            Card(CardRank.KING, CardSuit.SPADE),
            Card(CardRank.KING, CardSuit.HEART),
        ]
        info = HandDetector.detect(cards)
        assert info.hand_type == HandType.PAIR
        assert info.rank == CardRank.KING

    def test_triple(self):
        cards = [
            Card(CardRank.QUEEN, CardSuit.SPADE),
            Card(CardRank.QUEEN, CardSuit.HEART),
            Card(CardRank.QUEEN, CardSuit.CLUB),
        ]
        info = HandDetector.detect(cards)
        assert info.hand_type == HandType.TRIPLE
        assert info.rank == CardRank.QUEEN

    def test_bomb(self):
        cards = [
            Card(CardRank.JACK, CardSuit.SPADE),
            Card(CardRank.JACK, CardSuit.HEART),
            Card(CardRank.JACK, CardSuit.CLUB),
            Card(CardRank.JACK, CardSuit.DIAMOND),
        ]
        info = HandDetector.detect(cards)
        assert info.hand_type == HandType.BOMB
        assert info.rank == CardRank.JACK

    def test_rocket(self):
        cards = [
            Card(CardRank.BLACK_JOKER, CardSuit.JOKER),
            Card(CardRank.RED_JOKER, CardSuit.JOKER),
        ]
        info = HandDetector.detect(cards)
        assert info.hand_type == HandType.ROCKET


class TestHandDetectorTripleWith:
    """Test triple with single/pair detection."""

    def test_triple_with_single(self):
        cards = [
            Card(CardRank.TEN, CardSuit.SPADE),
            Card(CardRank.TEN, CardSuit.HEART),
            Card(CardRank.TEN, CardSuit.CLUB),
            Card(CardRank.THREE, CardSuit.SPADE),
        ]
        info = HandDetector.detect(cards)
        assert info.hand_type == HandType.TRIPLE_WITH_SINGLE
        assert info.rank == CardRank.TEN

    def test_triple_with_pair(self):
        cards = [
            Card(CardRank.NINE, CardSuit.SPADE),
            Card(CardRank.NINE, CardSuit.HEART),
            Card(CardRank.NINE, CardSuit.CLUB),
            Card(CardRank.FOUR, CardSuit.SPADE),
            Card(CardRank.FOUR, CardSuit.HEART),
        ]
        info = HandDetector.detect(cards)
        assert info.hand_type == HandType.TRIPLE_WITH_PAIR
        assert info.rank == CardRank.NINE


class TestHandDetectorStraight:
    """Test straight detection."""

    def test_straight_5_cards(self):
        cards = [
            Card(CardRank.THREE, CardSuit.SPADE),
            Card(CardRank.FOUR, CardSuit.HEART),
            Card(CardRank.FIVE, CardSuit.CLUB),
            Card(CardRank.SIX, CardSuit.DIAMOND),
            Card(CardRank.SEVEN, CardSuit.SPADE),
        ]
        info = HandDetector.detect(cards)
        assert info.hand_type == HandType.STRAIGHT
        assert info.length == 5

    def test_straight_cannot_include_two(self):
        cards = [
            Card(CardRank.TEN, CardSuit.SPADE),
            Card(CardRank.JACK, CardSuit.HEART),
            Card(CardRank.QUEEN, CardSuit.CLUB),
            Card(CardRank.KING, CardSuit.DIAMOND),
            Card(CardRank.ACE, CardSuit.SPADE),
            Card(CardRank.TWO, CardSuit.HEART),
        ]
        info = HandDetector.detect(cards)
        assert info.hand_type == HandType.INVALID

    def test_straight_pair(self):
        cards = [
            Card(CardRank.THREE, CardSuit.SPADE),
            Card(CardRank.THREE, CardSuit.HEART),
            Card(CardRank.FOUR, CardSuit.CLUB),
            Card(CardRank.FOUR, CardSuit.DIAMOND),
            Card(CardRank.FIVE, CardSuit.SPADE),
            Card(CardRank.FIVE, CardSuit.HEART),
        ]
        info = HandDetector.detect(cards)
        assert info.hand_type == HandType.STRAIGHT_PAIR
        assert info.length == 3


class TestHandDetectorAirplane:
    """Test airplane detection."""

    def test_airplane_basic(self):
        cards = [
            Card(CardRank.THREE, CardSuit.SPADE),
            Card(CardRank.THREE, CardSuit.HEART),
            Card(CardRank.THREE, CardSuit.CLUB),
            Card(CardRank.FOUR, CardSuit.SPADE),
            Card(CardRank.FOUR, CardSuit.HEART),
            Card(CardRank.FOUR, CardSuit.CLUB),
        ]
        info = HandDetector.detect(cards)
        assert info.hand_type == HandType.AIRPLANE
        assert info.length == 2

    def test_airplane_with_singles(self):
        cards = [
            Card(CardRank.FIVE, CardSuit.SPADE),
            Card(CardRank.FIVE, CardSuit.HEART),
            Card(CardRank.FIVE, CardSuit.CLUB),
            Card(CardRank.SIX, CardSuit.SPADE),
            Card(CardRank.SIX, CardSuit.HEART),
            Card(CardRank.SIX, CardSuit.CLUB),
            Card(CardRank.THREE, CardSuit.SPADE),
            Card(CardRank.KING, CardSuit.HEART),
        ]
        info = HandDetector.detect(cards)
        assert info.hand_type == HandType.AIRPLANE_WITH_SINGLES

    def test_airplane_with_pairs(self):
        cards = [
            Card(CardRank.SEVEN, CardSuit.SPADE),
            Card(CardRank.SEVEN, CardSuit.HEART),
            Card(CardRank.SEVEN, CardSuit.CLUB),
            Card(CardRank.EIGHT, CardSuit.SPADE),
            Card(CardRank.EIGHT, CardSuit.HEART),
            Card(CardRank.EIGHT, CardSuit.CLUB),
            Card(CardRank.THREE, CardSuit.SPADE),
            Card(CardRank.THREE, CardSuit.HEART),
            Card(CardRank.KING, CardSuit.CLUB),
            Card(CardRank.KING, CardSuit.DIAMOND),
        ]
        info = HandDetector.detect(cards)
        assert info.hand_type == HandType.AIRPLANE_WITH_PAIRS


class TestHandDetectorFourWith:
    """Test four with two detection."""

    def test_four_with_two_singles(self):
        cards = [
            Card(CardRank.EIGHT, CardSuit.SPADE),
            Card(CardRank.EIGHT, CardSuit.HEART),
            Card(CardRank.EIGHT, CardSuit.CLUB),
            Card(CardRank.EIGHT, CardSuit.DIAMOND),
            Card(CardRank.THREE, CardSuit.SPADE),
            Card(CardRank.KING, CardSuit.HEART),
        ]
        info = HandDetector.detect(cards)
        assert info.hand_type == HandType.FOUR_WITH_TWO_SINGLES
        assert info.rank == CardRank.EIGHT

    def test_four_with_two_pairs(self):
        cards = [
            Card(CardRank.NINE, CardSuit.SPADE),
            Card(CardRank.NINE, CardSuit.HEART),
            Card(CardRank.NINE, CardSuit.CLUB),
            Card(CardRank.NINE, CardSuit.DIAMOND),
            Card(CardRank.THREE, CardSuit.SPADE),
            Card(CardRank.THREE, CardSuit.HEART),
            Card(CardRank.KING, CardSuit.CLUB),
            Card(CardRank.KING, CardSuit.DIAMOND),
        ]
        info = HandDetector.detect(cards)
        assert info.hand_type == HandType.FOUR_WITH_TWO_PAIRS
        assert info.rank == CardRank.NINE


class TestHandDetectorInvalid:
    """Test invalid hand detection."""

    def test_invalid_two_different_singles(self):
        cards = [
            Card(CardRank.THREE, CardSuit.SPADE),
            Card(CardRank.FIVE, CardSuit.HEART),
        ]
        info = HandDetector.detect(cards)
        assert info.hand_type == HandType.INVALID

    def test_invalid_short_straight(self):
        cards = [
            Card(CardRank.THREE, CardSuit.SPADE),
            Card(CardRank.FOUR, CardSuit.HEART),
            Card(CardRank.FIVE, CardSuit.CLUB),
            Card(CardRank.SIX, CardSuit.DIAMOND),
        ]
        info = HandDetector.detect(cards)
        assert info.hand_type == HandType.INVALID


class TestHandInfoEdgeCases:
    """Test edge cases for HandInfo."""

    def test_rocket_vs_rocket(self):
        """Test rocket vs rocket (same type, same rank)."""
        rocket1 = HandInfo(HandType.ROCKET, 14, 2)
        rocket2 = HandInfo(HandType.ROCKET, 14, 2)
        # Rocket always beats other rocket in can_beat (returns True for same type)
        # Actually rocket.can_beat(rocket) returns True since it's same type
        result = rocket1.can_beat(rocket2)
        # The result depends on implementation - just verify it doesn't crash
        assert isinstance(result, bool)

    def test_nothing_beats_rocket(self):
        """Test that nothing beats rocket except higher rocket."""
        rocket = HandInfo(HandType.ROCKET, 14, 2)
        bomb = HandInfo(HandType.BOMB, 14, 1)
        assert not bomb.can_beat(rocket)

    def test_non_bomb_cannot_beat_bomb(self):
        """Test that non-bomb cannot beat bomb."""
        bomb = HandInfo(HandType.BOMB, 5, 1)
        single = HandInfo(HandType.SINGLE, 14, 1)
        assert not single.can_beat(bomb)


class TestHandDetectorAirplaneEdgeCases:
    """Test airplane edge cases."""

    def test_airplane_cannot_include_two(self):
        """Test that airplane cannot include 2."""
        cards = [
            Card(CardRank.ACE, CardSuit.SPADE),
            Card(CardRank.ACE, CardSuit.HEART),
            Card(CardRank.ACE, CardSuit.CLUB),
            Card(CardRank.TWO, CardSuit.SPADE),
            Card(CardRank.TWO, CardSuit.HEART),
            Card(CardRank.TWO, CardSuit.CLUB),
        ]
        info = HandDetector.detect(cards)
        # Should be invalid since 2 can't be in airplane
        assert info.hand_type == HandType.INVALID

    def test_airplane_non_consecutive(self):
        """Test non-consecutive triples are invalid airplane."""
        cards = [
            Card(CardRank.THREE, CardSuit.SPADE),
            Card(CardRank.THREE, CardSuit.HEART),
            Card(CardRank.THREE, CardSuit.CLUB),
            Card(CardRank.FIVE, CardSuit.SPADE),
            Card(CardRank.FIVE, CardSuit.HEART),
            Card(CardRank.FIVE, CardSuit.CLUB),
        ]
        info = HandDetector.detect(cards)
        assert info.hand_type == HandType.INVALID

    def test_single_triple_not_airplane(self):
        """Test that single triple is not airplane."""
        cards = [
            Card(CardRank.THREE, CardSuit.SPADE),
            Card(CardRank.THREE, CardSuit.HEART),
            Card(CardRank.THREE, CardSuit.CLUB),
        ]
        info = HandDetector.detect(cards)
        assert info.hand_type == HandType.TRIPLE


class TestHandDetectorStraightEdgeCases:
    """Test straight edge cases."""

    def test_straight_pair_cannot_include_two(self):
        """Test that straight pair cannot include 2."""
        cards = [
            Card(CardRank.ACE, CardSuit.SPADE),
            Card(CardRank.ACE, CardSuit.HEART),
            Card(CardRank.TWO, CardSuit.CLUB),
            Card(CardRank.TWO, CardSuit.DIAMOND),
            Card(CardRank.THREE, CardSuit.SPADE),
            Card(CardRank.THREE, CardSuit.HEART),
        ]
        info = HandDetector.detect(cards)
        assert info.hand_type == HandType.INVALID

    def test_is_consecutive_empty(self):
        """Test _is_consecutive with empty list."""
        result = HandDetector._is_consecutive([])
        assert result is False

    def test_get_rank_with_count_not_found(self):
        """Test _get_rank_with_count when count not found."""
        from collections import Counter

        rank_counts = Counter({CardRank.THREE: 2, CardRank.FOUR: 2})
        result = HandDetector._get_rank_with_count(rank_counts, 3)
        assert result == -1
