"""Reward shaping utilities for multi-player games.

This module provides utilities for computing rewards in multi-player games,
addressing common RL training challenges like sparse rewards and non-zero-sum
reward structures.

Key concepts:
    - Ranking rewards: Rewards based on final placement (1st, 2nd, etc.)
    - Relative score rewards: Rewards based on score vs. opponents' average
    - Win/lose rewards: Simple binary rewards for winning/losing
    - Step rewards: Small intermediate rewards for successful actions

Example:
    >>> from games.core.reward_shaping import RewardShaping
    >>> # 4-player game where player 0 wins
    >>> rewards = RewardShaping.ranking_reward(num_players=4, winner_idx=0)
    >>> print(rewards)  # [1.0, -0.33, -0.33, -0.33] (approximately)
"""

from __future__ import annotations


class RewardShaping:
    """Utility class for computing shaped rewards in multi-player games.

    All methods are static and return rewards that sum to zero (zero-sum),
    which is important for stable self-play training.
    """

    @staticmethod
    def ranking_reward(
        num_players: int,
        rankings: list[int] | None = None,
        winner_idx: int | None = None,
        win_reward: float = 1.0,
        lose_reward: float = -1.0,
    ) -> list[float]:
        """Compute rewards based on final rankings.

        For 2-player games, this is equivalent to win/lose rewards.
        For multi-player games, rewards are interpolated based on rank.

        Args:
            num_players: Total number of players.
            rankings: List where rankings[i] is player i's rank (1 = first place).
                If None, winner_idx must be provided.
            winner_idx: Index of the winning player (for simple win/lose).
                Only used if rankings is None.
            win_reward: Reward for first place.
            lose_reward: Reward for last place.

        Returns:
            List of rewards for each player, summing to approximately zero.
        """
        if rankings is None:
            if winner_idx is None:
                raise ValueError("Either rankings or winner_idx must be provided")
            # Simple win/lose case
            rankings = [2 if i != winner_idx else 1 for i in range(num_players)]

        rewards: list[float] = []
        for i in range(num_players):
            rank = rankings[i]
            if num_players == 1:
                rewards.append(0.0)
            elif num_players == 2:
                rewards.append(win_reward if rank == 1 else lose_reward)
            else:
                # Interpolate between win_reward and lose_reward based on rank
                # rank 1 -> win_reward, rank N -> lose_reward
                t = (rank - 1) / (num_players - 1)
                reward = win_reward * (1 - t) + lose_reward * t
                rewards.append(reward)

        # Normalize to zero-sum
        mean_reward = sum(rewards) / len(rewards)
        rewards = [r - mean_reward for r in rewards]

        return rewards

    @staticmethod
    def relative_score_reward(
        scores: list[float], normalize: bool = False
    ) -> list[float]:
        """Compute rewards based on score relative to opponents' average.

        This encourages maximizing the gap between your score and opponents,
        not just maximizing absolute score.

        Args:
            scores: List of scores for each player.
            normalize: If True, normalize rewards to [-1, 1] range.

        Returns:
            List of rewards (score - average_opponent_score) for each player.
        """
        num_players = len(scores)
        total_score = sum(scores)
        rewards: list[float] = []

        for i in range(num_players):
            if num_players == 1:
                rewards.append(0.0)
            else:
                # Average of opponents' scores
                opponent_avg = (total_score - scores[i]) / (num_players - 1)
                reward = scores[i] - opponent_avg
                rewards.append(reward)

        if normalize and rewards:
            max_abs = max(abs(r) for r in rewards)
            if max_abs > 0:
                rewards = [r / max_abs for r in rewards]

        return rewards

    @staticmethod
    def win_lose_reward(
        num_players: int,
        winner_idx: int | None,
        win_reward: float = 1.0,
        lose_reward: float = -1.0,
    ) -> list[float]:
        """Compute simple win/lose rewards.

        Args:
            num_players: Total number of players.
            winner_idx: Index of the winning player, or None for draw/no winner.
            win_reward: Reward for the winner.
            lose_reward: Reward for losers.

        Returns:
            List of rewards for each player.
        """
        if winner_idx is None:
            return [0.0] * num_players

        return [
            win_reward if i == winner_idx else lose_reward for i in range(num_players)
        ]

    @staticmethod
    def step_reward(
        action_success: bool,
        success_reward: float = 0.0,
        failure_reward: float = 0.0,
    ) -> float:
        """Compute intermediate step reward for an action.

        Use sparingly to avoid changing the optimal policy. Small values
        (e.g., 0.01) are recommended.

        Args:
            action_success: Whether the action was successful.
            success_reward: Reward for successful action.
            failure_reward: Reward for failed action.

        Returns:
            The step reward.
        """
        return success_reward if action_success else failure_reward

    @staticmethod
    def pbrs(
        env_reward: float,
        potential_current: float,
        potential_next: float,
        gamma: float = 0.99,
    ) -> float:
        """Compute Potential-Based Reward Shaping (PBRS).

        PBRS is theoretically guaranteed to not change the optimal policy.
        Formula: R_shaped = R_env + gamma * Phi(S_next) - Phi(S_current)

        Args:
            env_reward: The original environment reward.
            potential_current: Potential function value for current state.
            potential_next: Potential function value for next state.
                Should be 0 for terminal states.
            gamma: Discount factor (should match RL algorithm's gamma).

        Returns:
            The shaped reward.

        Example:
            # Define a potential function for hand quality
            def hand_potential(hand: list[Card]) -> float:
                return count_consecutive_cards(hand) / max_hand_size

            # In step():
            phi_current = hand_potential(old_hand)
            phi_next = hand_potential(new_hand) if not terminated else 0.0
            shaped_reward = RewardShaping.pbrs(reward, phi_current, phi_next)
        """
        return env_reward + gamma * potential_next - potential_current

    @staticmethod
    def curriculum_blend(
        dense_reward: float,
        sparse_reward: float,
        alpha: float,
    ) -> float:
        """Blend dense and sparse rewards for curriculum learning.

        Start training with alpha=1.0 (dense rewards) and gradually
        decrease to alpha=0.0 (sparse rewards only).

        Args:
            dense_reward: Dense reward (e.g., step-by-step shaping).
            sparse_reward: Sparse reward (e.g., win/lose at end).
            alpha: Blending factor in [0, 1]. 1.0 = all dense, 0.0 = all sparse.

        Returns:
            Blended reward: alpha * dense + (1 - alpha) * sparse

        Example:
            # Training schedule
            for epoch in range(num_epochs):
                alpha = max(0.0, 1.0 - epoch / warmup_epochs)
                reward = RewardShaping.curriculum_blend(dense, sparse, alpha)
        """
        return alpha * dense_reward + (1 - alpha) * sparse_reward

    @staticmethod
    def action_specific_reward(
        action_type: str,
        bonus_config: dict[str, float],
        cards_played: int = 0,
    ) -> float:
        """Compute action-specific dense rewards.

        Provides small intermediate rewards for specific actions to accelerate
        learning. Use sparingly to avoid changing optimal policy.

        Args:
            action_type: Type of action performed. Supported types:
                - "clear_hand": Player cleared their hand (triggers end game)
                - "play_cards": Player played cards (bonus per card)
                - "use_token": Player used special token (e.g., Scout & Play)
                - "dominate": Player forced opponent to pass
            bonus_config: Dict mapping action types to bonus values.
                For "play_cards", use "play_cards_per_card" key.
            cards_played: Number of cards played (for "play_cards" action).

        Returns:
            The action-specific reward bonus.

        Example:
            config = {
                "clear_hand": 0.5,
                "play_cards_per_card": 0.01,
                "use_token": 0.1,
                "dominate": 0.05,
            }
            reward = RewardShaping.action_specific_reward(
                action_type="play_cards",
                bonus_config=config,
                cards_played=4,
            )  # Returns 0.04
        """
        if action_type == "play_cards":
            per_card_bonus = bonus_config.get("play_cards_per_card", 0.0)
            return per_card_bonus * cards_played
        return bonus_config.get(action_type, 0.0)

    @staticmethod
    def hand_quality_potential(
        hand_values: list[int],
        max_hand_size: int,
        consecutive_weight: float = 0.6,
        pair_weight: float = 0.4,
    ) -> float:
        """Compute potential function based on hand quality.

        Evaluates hand quality for PBRS. Higher potential means better hand
        (more consecutive cards, more pairs/triples).

        Args:
            hand_values: List of card values/ranks in hand.
            max_hand_size: Maximum possible hand size (for normalization).
            consecutive_weight: Weight for consecutive card bonus.
            pair_weight: Weight for pair/triple bonus.

        Returns:
            Potential value in [0, 1] range.

        Example:
            # Hand with consecutive cards
            potential = RewardShaping.hand_quality_potential(
                hand_values=[3, 4, 5, 6, 7],
                max_hand_size=13,
            )  # High potential
        """
        if not hand_values:
            return 0.0

        sorted_values = sorted(hand_values)
        n = len(sorted_values)

        # Count consecutive sequences
        consecutive_count = 0
        current_seq = 1
        for i in range(1, n):
            if sorted_values[i] == sorted_values[i - 1] + 1:
                current_seq += 1
            else:
                if current_seq >= 2:
                    consecutive_count += current_seq
                current_seq = 1
        if current_seq >= 2:
            consecutive_count += current_seq

        # Count pairs/triples
        from collections import Counter

        value_counts = Counter(sorted_values)
        pair_count = sum(1 for count in value_counts.values() if count >= 2)
        triple_count = sum(1 for count in value_counts.values() if count >= 3)

        # Normalize scores
        max_consecutive = max_hand_size
        max_pairs = max_hand_size // 2

        consecutive_score = min(consecutive_count / max_consecutive, 1.0)
        pair_score = min((pair_count + triple_count * 2) / max_pairs, 1.0)

        # Weighted combination
        potential = consecutive_weight * consecutive_score + pair_weight * pair_score
        return min(potential, 1.0)

    @staticmethod
    def game_progress_potential(
        cards_remaining: int,
        initial_cards: int,
    ) -> float:
        """Compute potential based on game progress (cards played).

        Higher potential when closer to clearing hand.

        Args:
            cards_remaining: Number of cards remaining in hand.
            initial_cards: Initial number of cards dealt.

        Returns:
            Potential value in [0, 1] range.
        """
        if initial_cards <= 0:
            return 0.0
        return 1.0 - (cards_remaining / initial_cards)

    @staticmethod
    def score_lead_potential(
        my_score: float,
        opponent_scores: list[float],
        normalize_factor: float = 10.0,
    ) -> float:
        """Compute potential based on score lead over opponents.

        Args:
            my_score: Current player's score.
            opponent_scores: List of opponent scores.
            normalize_factor: Factor to normalize the lead.

        Returns:
            Potential value (positive if leading, negative if behind).
        """
        if not opponent_scores:
            return 0.0
        avg_opponent = sum(opponent_scores) / len(opponent_scores)
        lead = my_score - avg_opponent
        return lead / normalize_factor

    @staticmethod
    def exploration_bonus(
        state_key: str,
        state_counts: dict[str, int],
        bonus_scale: float = 0.1,
    ) -> float:
        """Compute exploration bonus for curiosity-driven learning.

        Encourages visiting novel states using count-based exploration.
        Formula: bonus = scale / sqrt(count)

        Args:
            state_key: String representation of current state.
            state_counts: Dict mapping state keys to visit counts.
            bonus_scale: Scale factor for the bonus.

        Returns:
            Exploration bonus (higher for less-visited states).
        """
        count = state_counts.get(state_key, 0)
        if count == 0:
            return bonus_scale
        return bonus_scale / (count**0.5)

    @staticmethod
    def prediction_reward(
        predicted: bool,
        actual: bool,
        correct_reward: float = 0.05,
        incorrect_penalty: float = -0.02,
    ) -> float:
        """Compute reward for auxiliary prediction tasks.

        Used for training auxiliary heads (e.g., predicting opponent actions).

        Args:
            predicted: The predicted value.
            actual: The actual value.
            correct_reward: Reward for correct prediction.
            incorrect_penalty: Penalty for incorrect prediction.

        Returns:
            Reward based on prediction accuracy.
        """
        return correct_reward if predicted == actual else incorrect_penalty

    @staticmethod
    def normalize_to_zero_sum(rewards: list[float]) -> list[float]:
        """Normalize rewards to sum to zero.

        Args:
            rewards: List of rewards for each player.

        Returns:
            Normalized rewards that sum to zero.
        """
        if not rewards:
            return []
        if len(rewards) == 1:
            return [0.0]
        mean_reward = sum(rewards) / len(rewards)
        return [r - mean_reward for r in rewards]
