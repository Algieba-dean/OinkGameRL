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


class TestActionSpecificReward:
    """Test action-specific dense rewards."""

    def test_clear_hand_bonus(self):
        """Test bonus for clearing hand (triggering end game)."""
        reward = RewardShaping.action_specific_reward(
            action_type="clear_hand",
            bonus_config={"clear_hand": 0.5},
        )
        assert reward == 0.5

    def test_play_multiple_cards_bonus(self):
        """Test bonus proportional to cards played."""
        reward = RewardShaping.action_specific_reward(
            action_type="play_cards",
            cards_played=4,
            bonus_config={"play_cards_per_card": 0.01},
        )
        assert abs(reward - 0.04) < 1e-6

    def test_use_special_token_bonus(self):
        """Test bonus for using special token (e.g., Scout & Play)."""
        reward = RewardShaping.action_specific_reward(
            action_type="use_token",
            bonus_config={"use_token": 0.1},
        )
        assert reward == 0.1

    def test_dominate_opponent_bonus(self):
        """Test bonus for forcing opponent to pass."""
        reward = RewardShaping.action_specific_reward(
            action_type="dominate",
            bonus_config={"dominate": 0.05},
        )
        assert reward == 0.05

    def test_unknown_action_no_bonus(self):
        """Test that unknown action types return 0."""
        reward = RewardShaping.action_specific_reward(
            action_type="unknown_action",
            bonus_config={"clear_hand": 0.5},
        )
        assert reward == 0.0

    def test_missing_config_no_bonus(self):
        """Test that missing config returns 0."""
        reward = RewardShaping.action_specific_reward(
            action_type="clear_hand",
            bonus_config={},
        )
        assert reward == 0.0


class TestPotentialFunctions:
    """Test potential function utilities."""

    def test_hand_quality_consecutive(self):
        """Test hand quality based on consecutive cards."""
        # Hand with consecutive cards [1,2,3,4,5] should have positive potential
        hand_values = [1, 2, 3, 4, 5]
        potential = RewardShaping.hand_quality_potential(
            hand_values=hand_values,
            max_hand_size=10,
        )
        assert potential > 0.0

    def test_hand_quality_scattered(self):
        """Test hand quality with scattered cards."""
        # Hand with scattered cards [1,3,5,7,9] should have lower potential
        # than consecutive cards
        scattered_values = [1, 3, 5, 7, 9]
        consecutive_values = [1, 2, 3, 4, 5]
        scattered_potential = RewardShaping.hand_quality_potential(
            hand_values=scattered_values,
            max_hand_size=10,
        )
        consecutive_potential = RewardShaping.hand_quality_potential(
            hand_values=consecutive_values,
            max_hand_size=10,
        )
        assert scattered_potential < consecutive_potential

    def test_hand_quality_pairs(self):
        """Test hand quality with pairs."""
        # Hand with pairs [2,2,5,5,8] should have positive potential
        hand_values = [2, 2, 5, 5, 8]
        potential = RewardShaping.hand_quality_potential(
            hand_values=hand_values,
            max_hand_size=10,
        )
        assert potential > 0.0

    def test_hand_quality_empty(self):
        """Test hand quality for empty hand."""
        potential = RewardShaping.hand_quality_potential(
            hand_values=[],
            max_hand_size=10,
        )
        assert potential == 0.0

    def test_hand_quality_normalized(self):
        """Test that hand quality is in [0, 1] range."""
        hand_values = [1, 1, 1, 2, 2, 2, 3, 3, 3]
        potential = RewardShaping.hand_quality_potential(
            hand_values=hand_values,
            max_hand_size=10,
        )
        assert 0.0 <= potential <= 1.0

    def test_game_progress_potential(self):
        """Test game progress potential."""
        # Early game (many cards left)
        early_potential = RewardShaping.game_progress_potential(
            cards_remaining=10,
            initial_cards=13,
        )
        # Late game (few cards left)
        late_potential = RewardShaping.game_progress_potential(
            cards_remaining=2,
            initial_cards=13,
        )
        assert late_potential > early_potential

    def test_game_progress_empty_hand(self):
        """Test game progress potential with empty hand."""
        potential = RewardShaping.game_progress_potential(
            cards_remaining=0,
            initial_cards=13,
        )
        assert potential == 1.0

    def test_score_lead_potential(self):
        """Test score lead potential."""
        # Leading by 5 points
        potential = RewardShaping.score_lead_potential(
            my_score=15,
            opponent_scores=[10, 8, 12],
        )
        assert potential > 0

    def test_score_lead_potential_behind(self):
        """Test score lead potential when behind."""
        potential = RewardShaping.score_lead_potential(
            my_score=5,
            opponent_scores=[10, 15, 20],
        )
        assert potential < 0

    def test_score_lead_potential_no_opponents(self):
        """Test score lead potential with no opponents."""
        potential = RewardShaping.score_lead_potential(
            my_score=10,
            opponent_scores=[],
        )
        assert potential == 0.0

    def test_game_progress_potential_zero_initial(self):
        """Test game progress potential with zero initial cards."""
        potential = RewardShaping.game_progress_potential(
            cards_remaining=5,
            initial_cards=0,
        )
        assert potential == 0.0

    def test_game_progress_potential_negative_initial(self):
        """Test game progress potential with negative initial cards."""
        potential = RewardShaping.game_progress_potential(
            cards_remaining=5,
            initial_cards=-1,
        )
        assert potential == 0.0


class TestIntrinsicMotivation:
    """Test intrinsic motivation rewards."""

    def test_exploration_bonus_novel_state(self):
        """Test exploration bonus for novel states."""
        state_counts = {"state_a": 1, "state_b": 10}
        bonus = RewardShaping.exploration_bonus(
            state_key="state_a",
            state_counts=state_counts,
            bonus_scale=0.1,
        )
        # Novel state should get higher bonus
        assert bonus > 0

    def test_exploration_bonus_visited_state(self):
        """Test exploration bonus for frequently visited states."""
        state_counts = {"state_a": 1, "state_b": 100}
        bonus = RewardShaping.exploration_bonus(
            state_key="state_b",
            state_counts=state_counts,
            bonus_scale=0.1,
        )
        # Frequently visited state should get lower bonus
        assert bonus < 0.02

    def test_exploration_bonus_new_state(self):
        """Test exploration bonus for completely new state."""
        state_counts = {"state_a": 1}
        bonus = RewardShaping.exploration_bonus(
            state_key="new_state",
            state_counts=state_counts,
            bonus_scale=0.1,
        )
        # New state should get maximum bonus
        assert bonus == 0.1

    def test_prediction_reward(self):
        """Test reward for correct predictions."""
        reward = RewardShaping.prediction_reward(
            predicted=True,
            actual=True,
            correct_reward=0.05,
            incorrect_penalty=-0.02,
        )
        assert reward == 0.05

    def test_prediction_penalty(self):
        """Test penalty for incorrect predictions."""
        reward = RewardShaping.prediction_reward(
            predicted=True,
            actual=False,
            correct_reward=0.05,
            incorrect_penalty=-0.02,
        )
        assert reward == -0.02


class TestZeroSumNormalization:
    """Test zero-sum normalization utilities."""

    def test_normalize_to_zero_sum(self):
        """Test normalizing rewards to zero-sum."""
        rewards = [10.0, 5.0, 3.0, 2.0]
        normalized = RewardShaping.normalize_to_zero_sum(rewards)
        assert abs(sum(normalized)) < 1e-6

    def test_normalize_preserves_order(self):
        """Test that normalization preserves reward ordering."""
        rewards = [10.0, 5.0, 3.0, 2.0]
        normalized = RewardShaping.normalize_to_zero_sum(rewards)
        assert normalized[0] > normalized[1] > normalized[2] > normalized[3]

    def test_normalize_empty_list(self):
        """Test normalizing empty list."""
        normalized = RewardShaping.normalize_to_zero_sum([])
        assert normalized == []

    def test_normalize_single_player(self):
        """Test normalizing single player reward."""
        normalized = RewardShaping.normalize_to_zero_sum([5.0])
        assert normalized == [0.0]
