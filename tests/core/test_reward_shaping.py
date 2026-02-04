"""Tests for RewardShaping utility class."""

from games.core.reward_shaping import RewardShaping


class TestRankingReward:
    """Test ranking-based reward calculation."""

    def test_two_players_winner_gets_positive(self):
        # Player 0 wins (rank 1), Player 1 loses (rank 2)
        rewards = RewardShaping.ranking_reward(num_players=2, winner_idx=0)
        assert rewards[0] > 0  # Winner gets positive
        assert rewards[1] < 0  # Loser gets negative

    def test_two_players_symmetric(self):
        rewards = RewardShaping.ranking_reward(num_players=2, winner_idx=0)
        assert rewards[0] == -rewards[1]  # Zero-sum

    def test_four_players_ranking(self):
        # Rankings: [1, 2, 3, 4] means player 0 is 1st, player 1 is 2nd, etc.
        rankings = [1, 2, 3, 4]
        rewards = RewardShaping.ranking_reward(num_players=4, rankings=rankings)
        assert rewards[0] > rewards[1] > rewards[2] > rewards[3]

    def test_four_players_sum_to_zero(self):
        rankings = [1, 2, 3, 4]
        rewards = RewardShaping.ranking_reward(num_players=4, rankings=rankings)
        assert abs(sum(rewards)) < 1e-6  # Zero-sum

    def test_custom_winner_loser_values(self):
        rewards = RewardShaping.ranking_reward(
            num_players=2, winner_idx=0, win_reward=10.0, lose_reward=-10.0
        )
        assert rewards[0] == 10.0
        assert rewards[1] == -10.0


class TestRelativeScoreReward:
    """Test relative score reward calculation."""

    def test_positive_when_above_average(self):
        scores = [100, 50, 50]  # Player 0 above average (66.67)
        rewards = RewardShaping.relative_score_reward(scores)
        assert rewards[0] > 0
        assert rewards[1] < 0
        assert rewards[2] < 0

    def test_zero_sum(self):
        scores = [100, 80, 60, 40]
        rewards = RewardShaping.relative_score_reward(scores)
        assert abs(sum(rewards)) < 1e-6

    def test_equal_scores_zero_reward(self):
        scores = [50, 50, 50]
        rewards = RewardShaping.relative_score_reward(scores)
        for r in rewards:
            assert abs(r) < 1e-6

    def test_with_normalization(self):
        scores = [100, 0]
        rewards = RewardShaping.relative_score_reward(scores, normalize=True)
        # After normalization, rewards should be in [-1, 1] range
        assert -1.0 <= rewards[0] <= 1.0
        assert -1.0 <= rewards[1] <= 1.0


class TestWinLoseReward:
    """Test simple win/lose reward."""

    def test_winner_gets_win_reward(self):
        rewards = RewardShaping.win_lose_reward(
            num_players=4, winner_idx=2, win_reward=1.0, lose_reward=-1.0
        )
        assert rewards[2] == 1.0
        assert rewards[0] == -1.0
        assert rewards[1] == -1.0
        assert rewards[3] == -1.0

    def test_no_winner_all_zero(self):
        rewards = RewardShaping.win_lose_reward(
            num_players=3, winner_idx=None, win_reward=1.0, lose_reward=-1.0
        )
        assert all(r == 0.0 for r in rewards)


class TestStepReward:
    """Test intermediate step rewards."""

    def test_positive_action_reward(self):
        reward = RewardShaping.step_reward(
            action_success=True, success_reward=0.1, failure_reward=-0.05
        )
        assert reward == 0.1

    def test_negative_action_reward(self):
        reward = RewardShaping.step_reward(
            action_success=False, success_reward=0.1, failure_reward=-0.05
        )
        assert reward == -0.05

    def test_default_values(self):
        reward = RewardShaping.step_reward(action_success=True)
        assert reward == 0.0  # Default is no step reward


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_ranking_reward_no_rankings_no_winner_raises(self):
        import pytest

        with pytest.raises(ValueError, match="Either rankings or winner_idx"):
            RewardShaping.ranking_reward(num_players=4)

    def test_ranking_reward_single_player(self):
        rewards = RewardShaping.ranking_reward(num_players=1, rankings=[1])
        assert rewards[0] == 0.0

    def test_relative_score_single_player(self):
        rewards = RewardShaping.relative_score_reward([100])
        assert rewards[0] == 0.0


class TestPBRS:
    """Test Potential-Based Reward Shaping (PBRS)."""

    def test_pbrs_basic(self):
        r_env = 0.0
        phi_current = 0.5
        phi_next = 0.8
        gamma = 0.99
        r_shaped = RewardShaping.pbrs(
            env_reward=r_env,
            potential_current=phi_current,
            potential_next=phi_next,
            gamma=gamma,
        )
        expected = r_env + gamma * phi_next - phi_current
        assert abs(r_shaped - expected) < 1e-6

    def test_pbrs_terminal_state(self):
        r_shaped = RewardShaping.pbrs(
            env_reward=1.0,
            potential_current=0.5,
            potential_next=0.0,
            gamma=0.99,
        )
        expected = 1.0 + 0.99 * 0.0 - 0.5
        assert abs(r_shaped - expected) < 1e-6

    def test_pbrs_positive_shaping(self):
        r_shaped = RewardShaping.pbrs(
            env_reward=0.0,
            potential_current=0.2,
            potential_next=0.8,
            gamma=1.0,
        )
        assert r_shaped > 0

    def test_pbrs_negative_shaping(self):
        r_shaped = RewardShaping.pbrs(
            env_reward=0.0,
            potential_current=0.8,
            potential_next=0.2,
            gamma=1.0,
        )
        assert r_shaped < 0


class TestCurriculumReward:
    """Test curriculum learning reward blending."""

    def test_blend_dense_only(self):
        blended = RewardShaping.curriculum_blend(
            dense_reward=1.0, sparse_reward=0.0, alpha=1.0
        )
        assert blended == 1.0

    def test_blend_sparse_only(self):
        blended = RewardShaping.curriculum_blend(
            dense_reward=1.0, sparse_reward=0.5, alpha=0.0
        )
        assert blended == 0.5

    def test_blend_mixed(self):
        blended = RewardShaping.curriculum_blend(
            dense_reward=1.0, sparse_reward=0.0, alpha=0.5
        )
        assert blended == 0.5
