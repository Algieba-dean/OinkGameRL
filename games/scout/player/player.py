from games.scout.card.cards import Card


class Player:
    """Represents a player in the Scout game.

    Manages the player's hand of cards and Scout-and-Show token state.
    """

    def __init__(self, player_idx: int, cards: list[Card]) -> None:
        self.__player_idx: int = player_idx
        self.__hand: list[Card] = list(cards)
        self.__scout_and_show_token: bool = True

    @property
    def player_idx(self) -> int:
        return self.__player_idx

    @property
    def hand(self) -> tuple[Card, ...]:
        return tuple(self.__hand)

    @property
    def hand_count(self) -> int:
        return len(self.__hand)

    @property
    def scout_and_show_token(self) -> bool:
        return self.__scout_and_show_token

    def play_cards(self, start_idx: int, end_idx: int) -> list[Card]:
        """Remove and return cards from hand in the specified range.

        Args:
            start_idx: Start index (inclusive)
            end_idx: End index (inclusive)

        Returns:
            List of played cards

        Raises:
            ValueError: If indices are invalid
        """
        if start_idx < 0 or end_idx >= len(self.__hand) or start_idx > end_idx:
            raise ValueError(
                f"Invalid indices: start={start_idx}, end={end_idx}, "
                f"hand_size={len(self.__hand)}"
            )

        played_cards = self.__hand[start_idx : end_idx + 1]
        self.__hand = self.__hand[:start_idx] + self.__hand[end_idx + 1 :]
        return played_cards

    def insert_card(self, card: Card, position: int) -> None:
        """Insert a card at the specified position in hand.

        Args:
            card: The card to insert
            position: Position to insert at (0 to hand_count inclusive)

        Raises:
            ValueError: If position is invalid
        """
        if position < 0 or position > len(self.__hand):
            raise ValueError(
                f"Invalid position: {position}, valid range: 0 to {len(self.__hand)}"
            )
        self.__hand.insert(position, card)

    def use_scout_and_show_token(self) -> None:
        """Use the Scout-and-Show token.

        Raises:
            ValueError: If token is already used
        """
        if not self.__scout_and_show_token:
            raise ValueError("Scout and Show token already used")
        self.__scout_and_show_token = False

    def reset_token(self) -> None:
        """Reset the Scout-and-Show token to available state."""
        self.__scout_and_show_token = True

    def reset(self, cards: list[Card]) -> None:
        """Reset player state with new cards.

        Args:
            cards: New hand of cards
        """
        self.__hand = list(cards)
        self.__scout_and_show_token = True
