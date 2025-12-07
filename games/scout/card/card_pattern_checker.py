from games.scout.card.cards import Card
from games.scout.enums import CardPattern


class CardPatternChecker:
    @staticmethod
    def get_pattern(cards: list[Card]) -> CardPattern:
        if not cards:
            return CardPattern.INVALID_PATTERN
        if CardPatternChecker.__is_same_rank(cards=cards):
            return CardPattern.SAME_RANK
        if CardPatternChecker.__is_sequence(cards=cards):
            return CardPattern.SEQUENCE
        return CardPattern.INVALID_PATTERN

    @staticmethod
    def __is_same_rank(cards: list[Card]) -> bool:
        return len({card.top for card in cards}) == 1

    @staticmethod
    def __is_sequence(cards: list[Card]) -> bool:
        allowed_gaps = [-1, 1]
        # as one card will be recognized as same_rank, we at least we have 2 cards
        current_gap = cards[1].top - cards[0].top

        if current_gap not in allowed_gaps:
            return False

        previous_card = None
        for card in cards:
            if previous_card is None:
                previous_card = card
                continue

            if card.top - previous_card.top != current_gap:
                return False

            previous_card = card

        return True
