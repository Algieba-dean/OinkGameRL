"""Tests for Guandan hand detector module."""

from games.guandan.card import Card
from games.guandan.enums import CardRank, CardSuit, HandType
from games.guandan.hand_detector import HandDetector, HandInfo


class TestHandInfo:
    """Test HandInfo class."""

    def test_rocket_beats_everything(self):
        rocket = HandInfo(HandType.ROCKET, 100, 4)
        bomb = HandInfo(HandType.BOMB_8, 10, 1)
        assert rocket.can_beat(bomb)

    def test_bigger_bomb_beats_smaller(self):
        bomb8 = HandInfo(HandType.BOMB_8, 5, 1)
        bomb4 = HandInfo(HandType.BOMB_4, 10, 1)
        assert bomb8.can_beat(bomb4)

    def test_bomb_beats_non_bomb(self):
        bomb = HandInfo(HandType.BOMB_4, 5, 1)
        triple = HandInfo(HandType.TRIPLE, 10, 1)
        assert bomb.can_beat(triple)

    def test_same_type_higher_rank_wins(self):
        single_high = HandInfo(HandType.SINGLE, 10, 1)
        single_low = HandInfo(HandType.SINGLE, 5, 1)
        assert single_high.can_beat(single_low)
        assert not single_low.can_beat(single_high)

    def test_different_types_cannot_beat(self):
        single = HandInfo(HandType.SINGLE, 10, 1)
        pair = HandInfo(HandType.PAIR, 5, 1)
        assert not single.can_beat(pair)

    def test_different_length_cannot_beat(self):
        straight5 = HandInfo(HandType.STRAIGHT, 7, 5)
        straight6 = HandInfo(HandType.STRAIGHT, 8, 6)
        assert not straight5.can_beat(straight6)

    def test_straight_flush_is_bomb(self):
        sf = HandInfo(HandType.STRAIGHT_FLUSH, 10, 5)
        assert sf._is_bomb()


class TestHandDetectorBasic:
    """Test basic hand detection."""

    def test_empty_hand_is_pass(self):
        info = HandDetector.detect([])
        assert info.hand_type == HandType.PASS

    def test_single(self):
        card = Card(CardRank.ACE, CardSuit.SPADE, 0)
        info = HandDetector.detect([card])
        assert info.hand_type == HandType.SINGLE

    def test_pair(self):
        cards = [
            Card(CardRank.KING, CardSuit.SPADE, 0),
            Card(CardRank.KING, CardSuit.HEART, 0),
        ]
        info = HandDetector.detect(cards)
        assert info.hand_type == HandType.PAIR

    def test_triple(self):
        cards = [
            Card(CardRank.QUEEN, CardSuit.SPADE, 0),
            Card(CardRank.QUEEN, CardSuit.HEART, 0),
            Card(CardRank.QUEEN, CardSuit.CLUB, 0),
        ]
        info = HandDetector.detect(cards)
        assert info.hand_type == HandType.TRIPLE


class TestHandDetectorBombs:
    """Test bomb detection."""

    def test_bomb_4(self):
        cards = [
            Card(CardRank.JACK, CardSuit.SPADE, 0),
            Card(CardRank.JACK, CardSuit.HEART, 0),
            Card(CardRank.JACK, CardSuit.CLUB, 0),
            Card(CardRank.JACK, CardSuit.DIAMOND, 0),
        ]
        info = HandDetector.detect(cards)
        assert info.hand_type == HandType.BOMB_4

    def test_bomb_5(self):
        cards = [
            Card(CardRank.TEN, CardSuit.SPADE, 0),
            Card(CardRank.TEN, CardSuit.HEART, 0),
            Card(CardRank.TEN, CardSuit.CLUB, 0),
            Card(CardRank.TEN, CardSuit.DIAMOND, 0),
            Card(CardRank.TEN, CardSuit.SPADE, 1),
        ]
        info = HandDetector.detect(cards)
        assert info.hand_type == HandType.BOMB_5

    def test_bomb_6(self):
        cards = [
            Card(CardRank.NINE, CardSuit.SPADE, 0),
            Card(CardRank.NINE, CardSuit.HEART, 0),
            Card(CardRank.NINE, CardSuit.CLUB, 0),
            Card(CardRank.NINE, CardSuit.DIAMOND, 0),
            Card(CardRank.NINE, CardSuit.SPADE, 1),
            Card(CardRank.NINE, CardSuit.HEART, 1),
        ]
        info = HandDetector.detect(cards)
        assert info.hand_type == HandType.BOMB_6

    def test_rocket(self):
        cards = [
            Card(CardRank.BLACK_JOKER, CardSuit.JOKER, 0),
            Card(CardRank.BLACK_JOKER, CardSuit.JOKER, 1),
            Card(CardRank.RED_JOKER, CardSuit.JOKER, 0),
            Card(CardRank.RED_JOKER, CardSuit.JOKER, 1),
        ]
        info = HandDetector.detect(cards)
        assert info.hand_type == HandType.ROCKET


class TestHandDetectorTripleWith:
    """Test triple with two detection."""

    def test_triple_with_two(self):
        cards = [
            Card(CardRank.TEN, CardSuit.SPADE, 0),
            Card(CardRank.TEN, CardSuit.HEART, 0),
            Card(CardRank.TEN, CardSuit.CLUB, 0),
            Card(CardRank.THREE, CardSuit.SPADE, 0),
            Card(CardRank.THREE, CardSuit.HEART, 0),
        ]
        info = HandDetector.detect(cards)
        assert info.hand_type == HandType.TRIPLE_WITH_TWO


class TestHandDetectorStraight:
    """Test straight detection."""

    def test_straight_5_cards(self):
        cards = [
            Card(CardRank.THREE, CardSuit.SPADE, 0),
            Card(CardRank.FOUR, CardSuit.HEART, 0),
            Card(CardRank.FIVE, CardSuit.CLUB, 0),
            Card(CardRank.SIX, CardSuit.DIAMOND, 0),
            Card(CardRank.SEVEN, CardSuit.SPADE, 1),
        ]
        info = HandDetector.detect(cards)
        assert info.hand_type == HandType.STRAIGHT
        assert info.length == 5

    def test_straight_flush(self):
        cards = [
            Card(CardRank.THREE, CardSuit.HEART, 0),
            Card(CardRank.FOUR, CardSuit.HEART, 0),
            Card(CardRank.FIVE, CardSuit.HEART, 0),
            Card(CardRank.SIX, CardSuit.HEART, 0),
            Card(CardRank.SEVEN, CardSuit.HEART, 0),
        ]
        info = HandDetector.detect(cards)
        assert info.hand_type == HandType.STRAIGHT_FLUSH


class TestHandDetectorTube:
    """Test tube (连对) detection."""

    def test_tube_3_pairs(self):
        cards = [
            Card(CardRank.THREE, CardSuit.SPADE, 0),
            Card(CardRank.THREE, CardSuit.HEART, 0),
            Card(CardRank.FOUR, CardSuit.CLUB, 0),
            Card(CardRank.FOUR, CardSuit.DIAMOND, 0),
            Card(CardRank.FIVE, CardSuit.SPADE, 1),
            Card(CardRank.FIVE, CardSuit.HEART, 1),
        ]
        info = HandDetector.detect(cards)
        assert info.hand_type == HandType.TUBE
        assert info.length == 3


class TestHandDetectorPlate:
    """Test plate (板子) detection."""

    def test_plate_2_triples(self):
        cards = [
            Card(CardRank.THREE, CardSuit.SPADE, 0),
            Card(CardRank.THREE, CardSuit.HEART, 0),
            Card(CardRank.THREE, CardSuit.CLUB, 0),
            Card(CardRank.FOUR, CardSuit.SPADE, 1),
            Card(CardRank.FOUR, CardSuit.HEART, 1),
            Card(CardRank.FOUR, CardSuit.CLUB, 1),
        ]
        info = HandDetector.detect(cards)
        assert info.hand_type == HandType.PLATE
        assert info.length == 2


class TestHandDetectorInvalid:
    """Test invalid hand detection."""

    def test_invalid_two_different_singles(self):
        cards = [
            Card(CardRank.THREE, CardSuit.SPADE, 0),
            Card(CardRank.FIVE, CardSuit.HEART, 0),
        ]
        info = HandDetector.detect(cards)
        assert info.hand_type == HandType.INVALID

    def test_invalid_short_straight(self):
        cards = [
            Card(CardRank.THREE, CardSuit.SPADE, 0),
            Card(CardRank.FOUR, CardSuit.HEART, 0),
            Card(CardRank.FIVE, CardSuit.CLUB, 0),
            Card(CardRank.SIX, CardSuit.DIAMOND, 0),
        ]
        info = HandDetector.detect(cards)
        assert info.hand_type == HandType.INVALID


class TestHandDetectorLevelCard:
    """Test level card (级牌) handling."""

    def test_level_card_effective_rank(self):
        # When level is 5, 5s become highest regular
        cards = [Card(CardRank.FIVE, CardSuit.SPADE, 0)]
        info = HandDetector.detect(cards, CardRank.FIVE)
        assert info.rank == 98  # Level card rank

    def test_level_card_not_in_straight(self):
        # Level cards can't be in straights
        cards = [
            Card(CardRank.THREE, CardSuit.SPADE, 0),
            Card(CardRank.FOUR, CardSuit.HEART, 0),
            Card(CardRank.FIVE, CardSuit.CLUB, 0),  # Level card
            Card(CardRank.SIX, CardSuit.DIAMOND, 0),
            Card(CardRank.SEVEN, CardSuit.SPADE, 1),
        ]
        info = HandDetector.detect(cards, CardRank.FIVE)
        assert info.hand_type == HandType.INVALID
