import pytest
import gymnasium as gym
import random
from typing import List, Dict, Tuple, Any

from games.AutoOpponentWrapper import AutoOpponentWrapper
from games.OinkGame import OinkGameEnv
from games.GameAgent import GameAgent


class ScriptedEnv(OinkGameEnv):

    def __init__(
        self,
        player_sequence: List[int],
        reward_sequence: List[int] = None,
        render_mode=None,
    ):
        super().__init__(render_mode)
        self._player_seq = player_sequence
        self._reward_seq = (
            reward_sequence
            if reward_sequence is not None
            else [1] * len(player_sequence)
        )
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Discrete(10)
        self.current_seq_idx = 0

        # Initialize current_player_idx
        self._update_current_player()

    def _update_current_player(self):
        if self._player_seq and self.current_seq_idx < len(self._player_seq):
            self._current_player_idx = self._player_seq[self.current_seq_idx]
        else:
            self._current_player_idx = 0

    def _apply_action(self, action):
        # Boundary protection
        if self.current_seq_idx >= len(self._player_seq):
            return 0, True

        reward = self._reward_seq[self.current_seq_idx]
        self.current_seq_idx += 1

        # Check if terminated
        terminated = self.current_seq_idx == len(self._player_seq)
        if not terminated:
            self._update_current_player()

        return reward, terminated

    def _get_action_mask(self, player_idx):
        return [1]

    def _render_text(self):
        return ""

    def _get_global_state(self):
        return {1: "0"}

    def _get_observation(self, player_idx):
        return ""

    def _reset_logic(self, seed=0, options=None):
        self.current_seq_idx = 0
        self._update_current_player()


@pytest.fixture
def bot_factory(mocker):
    """Factory to produce Mock Bots."""

    def _create(return_value=1):
        bot = mocker.Mock(spec=GameAgent)
        bot.predict.return_value = return_value
        return bot

    return _create


@pytest.fixture
def game_builder(bot_factory):
    """
    Core Builder Fixture: Assembles Env, Bots, and Wrapper in one step.
    Returns a tuple for simplicity (could be a namedtuple or dict).
    """

    def _build(
        player_seq: List[int],
        reward_seq: List[int] = None,
        bot_indices: List[int] = None,
        ego_idx: int = 0,
    ) -> Tuple[AutoOpponentWrapper, ScriptedEnv, Dict[int, Any]]:

        # 1. Create Bots
        if bot_indices is None:
            bot_indices = []
        bots = {idx: bot_factory() for idx in bot_indices}

        # 2. Create Env
        env = ScriptedEnv(player_sequence=player_seq, reward_sequence=reward_seq)

        # 3. Create Wrapper
        wrapper = AutoOpponentWrapper(env=env, bots=bots, ego_player_idx=ego_idx)

        return wrapper, env, bots

    return _build


def gen_bot_cases():
    """Generate test cases for Bot configurations."""
    # Use a fixed seed to ensure determinism
    rng = random.Random(42)
    return [
        pytest.param([], id="empty_bots"),
        pytest.param([1, 5], id="non_consecutive"),
        pytest.param([2, 4, 1, 3], id="shuffled"),
        # Use rng instead of global random
        pytest.param(rng.sample(range(1, 100), 10), id="random_sample"),
    ]


def gen_gameplay_cases():
    """Generate test cases for complex gameplay flows."""
    return [
        pytest.param(
            {"seq": [0, 0, 0], "bots": [1], "rewards": [1, 2, 3]}, id="ego_only"
        ),
        pytest.param(
            {"seq": [0, 1, 0, 1], "bots": [1], "rewards": [10, 11, 12, 13]},
            id="alternating",
        ),
        pytest.param(
            {"seq": [1, 1, 1], "bots": [1], "rewards": [0, 0, 0]}, id="bots_only"
        ),
    ]


class TestWrapperContract:
    """Tests basic contracts like properties and type checks."""

    def test_property_immutability(self, game_builder):
        # Use builder to quickly create a minimal viable environment
        wrapper, _, bots = game_builder(player_seq=[0, 1], bot_indices=[1])

        # Verify `bots` property
        assert wrapper.bots is bots
        with pytest.raises(AttributeError, match="has no setter"):
            wrapper.bots = {}

        # Verify `ego_player_idx` property
        assert wrapper.ego_player_idx == 0
        with pytest.raises(AttributeError, match="has no setter"):
            wrapper.ego_player_idx = 1

    def test_ego_idx_occupied_error(self, game_builder):
        """Should raise an error if Ego's position is occupied by a Bot."""
        with pytest.raises(ValueError, match="occupied"):
            # Attempt to place a Bot at index 0 while setting Ego to 0
            game_builder(player_seq=[0], bot_indices=[0], ego_idx=0)


class TestWrapperGameplay:
    """Tests actual gameplay flow logic."""

    def test_regular_step_flow(self, game_builder):
        """Verify standard one-step interaction: Bot moves automatically -> Ego moves."""
        wrapper, _, bots = game_builder(
            player_seq=[1, 0, 0], reward_seq=[10, 20, 30], bot_indices=[1]
        )
        wrapper.reset()
        bot = bots[1]

        # --- Step 1 ---
        # Sequence is [1, 0, 0].
        # When calling wrapper.step(action):
        # 1. Internally processes index=1 (Bot). Bot is called.
        # 2. Environment rotates to index=0 (Ego).
        # 3. Execute Ego's action, returning Observation.
        _, reward, terminated, _, _ = wrapper.step(action=99)

        assert reward == 20  # Reward corresponding to the step (Ego's turn)
        assert not terminated
        assert bot.predict.call_count == 1  # Bot moved once

        # --- Step 2 ---
        # Remaining sequence [0].
        # Execute Ego action directly.
        _, reward, terminated, _, _ = wrapper.step(action=99)

        assert reward == 30
        assert terminated
        assert bot.predict.call_count == 1  # Bot did not move again

    def test_reset_clears_state(self, game_builder):
        """Verify game can restart after reset."""
        wrapper, _, bots = game_builder(player_seq=[1, 0], bot_indices=[1])
        bot = bots[1]

        # First round
        wrapper.reset()
        wrapper.step(0)
        assert bot.predict.call_count == 1

        # Second round (Reset)
        bot.predict.reset_mock()  # Reset mock counter
        wrapper.reset()

        wrapper.step(0)
        assert bot.predict.call_count == 1

    def test_opponent_starts_and_wins(self, game_builder):
        """Verify scenario where Bot wins immediately at the start."""
        # Sequence [1]: Bot moves once, game ends immediately.
        wrapper, _, bots = game_builder(player_seq=[1], bot_indices=[1])
        bot = bots[1]

        wrapper.reset()
        # Depending on Wrapper implementation, it might finish in reset or require a step trigger.
        # Assuming wrapper.step handles the progression.
        _, _, terminated, _, _ = wrapper.step(0)

        assert terminated
        assert bot.predict.call_count == 1

    def test_ego_is_not_zero(self, game_builder):
        """Verify scenario where Ego is not player 0."""
        # Sequence [0, 0, 1], Ego is 1, Bot is 0.
        wrapper, env, bots = game_builder(
            player_seq=[0, 0, 1], reward_seq=[10, 20, 30], bot_indices=[0], ego_idx=1
        )
        wrapper.reset()

        # Call step; wrapper should automatically finish the turns for the first two Bots (0)
        _, reward, terminated, _, _ = wrapper.step(action=99)

        assert reward == 30
        assert terminated
        assert bots[0].predict.call_count == 2
        assert env._current_player_idx == 1

    @pytest.mark.parametrize("bot_indices", gen_bot_cases())
    def test_bot_initialization(self, game_builder, bot_indices):
        """Verify initialization with different Bot counts/orders does not raise errors."""
        wrapper, _, _ = game_builder(player_seq=[0], bot_indices=bot_indices)
        assert len(wrapper.bots) == len(bot_indices)

    @pytest.mark.parametrize("case", gen_gameplay_cases())
    def test_complex_gameplay_scenarios(self, game_builder, case):
        """
        Verify gameplay flow under complex sequences.
        Core logic: Calculate expected Ego moves, loop execution, and verification.
        """
        seq = case["seq"]
        bot_indices = case["bots"]
        wrapper, _, bots = game_builder(
            player_seq=seq, reward_seq=case["rewards"], bot_indices=bot_indices
        )
        wrapper.reset()

        # Expected move counts and rewards for Ego (Player 0)
        expected_moves = seq.count(0)
        expected_rewards = [
            reward for idx, reward in zip(seq, case["rewards"]) if idx == 0
        ]

        for i in range(expected_moves):
            _, reward, terminated, _, _ = wrapper.step(0)

            assert reward == expected_rewards[i]
            if i == expected_moves - 1:
                assert terminated
            else:
                assert not terminated

        # Verify call counts for all Bots
        for idx, bot in bots.items():
            assert bot.predict.call_count == seq.count(idx)
