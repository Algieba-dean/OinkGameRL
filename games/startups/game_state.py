"""GameState module for Startups game."""

from __future__ import annotations

from games.startups.card import Card
from games.startups.constants import CompanyConsts, PlayerConsts
from games.startups.dealer import Dealer
from games.startups.enums import Company
from games.startups.player import Player


class GameState:
    """Manages the complete state of a Startups game."""

    CARDS_PER_PLAYER = {3: 6, 4: 5, 5: 4, 6: 3, 7: 3}

    def __init__(self, player_num: int) -> None:
        if player_num not in PlayerConsts.ALLOWED_PLAYER_NUM:
            raise ValueError(
                f"Invalid player num {player_num}, "
                f"allowed: {PlayerConsts.ALLOWED_PLAYER_NUM}"
            )

        self.__player_num = player_num
        self.__players = [Player(player_idx=i) for i in range(player_num)]
        self.__market: list[Card] = []
        self.__current_player_idx = 0
        self.__dealer = Dealer()

    @property
    def player_num(self) -> int:
        return self.__player_num

    @property
    def players(self) -> tuple[Player, ...]:
        return tuple(self.__players)

    @property
    def current_player_idx(self) -> int:
        return self.__current_player_idx

    @property
    def market(self) -> tuple[Card, ...]:
        return tuple(self.__market)

    @property
    def dealer(self) -> Dealer:
        return self.__dealer

    @property
    def is_terminated(self) -> bool:
        """Game ends when deck is empty and a player empties their hand."""
        if self.__dealer.deck_count > 0:
            return False
        return any(p.hand_count == 0 for p in self.__players)

    def get_player(self, player_idx: int) -> Player:
        return self.__players[player_idx]

    def get_current_player(self) -> Player:
        return self.__players[self.__current_player_idx]

    def next_player(self) -> None:
        self.__current_player_idx = (self.__current_player_idx + 1) % self.__player_num

    def add_to_market(self, card: Card) -> None:
        """Add a card to the market."""
        self.__market.append(card)

    def take_from_market(self, card_idx: int) -> Card:
        """Take a card from the market."""
        if card_idx < 0 or card_idx >= len(self.__market):
            raise ValueError(f"Invalid market index: {card_idx}")
        return self.__market.pop(card_idx)

    def calculate_scores(self) -> list[int]:
        """Calculate final scores for all players."""
        scores = []
        company_values = dict(zip(Company, CompanyConsts.COMPANY_VALUES, strict=True))

        for player in self.__players:
            score = player.coins
            for company in Company:
                count = player.get_company_count(company)
                if count > 0:
                    majority = self._has_majority(player.player_idx, company)
                    if majority:
                        score += company_values[company] * count
            scores.append(score)
        return scores

    def _has_majority(self, player_idx: int, company: Company) -> bool:
        """Check if player has majority (most cards) for a company."""
        player_count = self.__players[player_idx].get_company_count(company)
        if player_count == 0:
            return False
        for i, p in enumerate(self.__players):
            if i != player_idx and p.get_company_count(company) >= player_count:
                return False
        return True

    def get_winner(self) -> int | None:
        """Get winner (highest score)."""
        if not self.is_terminated:
            return None
        scores = self.calculate_scores()
        max_score = max(scores)
        winners = [i for i, s in enumerate(scores) if s == max_score]
        return winners[0] if len(winners) == 1 else None

    def reset(self, rng) -> None:
        """Reset game state."""
        for player in self.__players:
            player.reset()
        self.__market = []
        self.__current_player_idx = 0

        self.__dealer.reset(random_generator=rng)
        self.__dealer.create_and_shuffle_deck()

        cards_per = self.CARDS_PER_PLAYER[self.__player_num]
        hands = self.__dealer.deal_to_players(self.__player_num, cards_per)
        for i, hand in enumerate(hands):
            self.__players[i].set_hand(hand)

        for _ in range(self.__player_num):
            card = self.__dealer.deal_one()
            if card:
                self.__market.append(card)
