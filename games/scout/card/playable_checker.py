from games.scout.card.card_pattern_checker import CardPatternChecker
from games.scout.card.cards import Card
from games.scout.enums import CardPattern


class PlayableChecker:
    @staticmethod
    def is_playable(board_cards: list[Card], target_cards: list[Card]) -> bool:
        target_pattern = CardPatternChecker.get_pattern(cards=target_cards)

        if target_pattern == CardPattern.INVALID_PATTERN:
            return False

        # check lengh first
        if len(target_cards) > len(board_cards):
            return True
        if len(target_cards) < len(board_cards):
            return False

        # check parttern (SAME_RANK is bigger than SEQUENCE)
        board_pattern = CardPatternChecker.get_pattern(cards=board_cards)
        if (
            target_pattern == CardPattern.SAME_RANK
            and board_pattern == CardPattern.SEQUENCE
        ):
            return True
        if (
            target_pattern == CardPattern.SEQUENCE
            and board_pattern == CardPattern.SAME_RANK
        ):
            return False
        # check values
        return PlayableChecker.__is_value_bigger(
            board_cards=board_cards, target_cards=target_cards
        )

    @staticmethod
    def get_all_playable_subsets(
        board_cards: list[Card], target_cards: list[Card]
    ) -> list[tuple[int, int]]:
        playable_subsets = []
        for start in range(len(target_cards)):
            for end in range(len(target_cards)):
                subset = target_cards[start : end + 1]
                pattern = CardPatternChecker.get_pattern(cards=subset)
                if (
                    pattern == CardPattern.INVALID_PATTERN
                    or not PlayableChecker.is_playable(
                        board_cards=board_cards, target_cards=subset
                    )
                ):
                    continue
                playable_subsets.append((start, end))
        return playable_subsets

    @staticmethod
    def __is_value_bigger(board_cards: list[Card], target_cards: list[Card]) -> bool:
        # both pattern should be the same, both length should be same
        pattern = CardPatternChecker.get_pattern(cards=board_cards)
        if pattern == CardPattern.SAME_RANK:
            return target_cards[0].top > board_cards[0].top

        # compare SEQUENCE pattern
        biggest_board_top = (
            board_cards[0].top
            if board_cards[0].top > board_cards[-1].top
            else board_cards[-1].top
        )
        biggest_target_top = (
            target_cards[0].top
            if target_cards[0].top > target_cards[-1].top
            else target_cards[-1].top
        )
        return biggest_target_top > biggest_board_top
