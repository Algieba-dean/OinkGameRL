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
