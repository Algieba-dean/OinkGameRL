import pytest
import gymnasium as gym
import numpy as np
import random

from games.AutoOpponentWrapper import AutoOpponentWrapper
from games.OinkGame import OinkGameEnv
from games.GameAgent import GameAgent


class MockBot(GameAgent):
    def __init__(self):
        self.call_count = 0

    def predict(self, observation, action_mask):
        self.call_count += 1
        return 1


class ScriptedEnv(OinkGameEnv):
    def __init__(
        self,
        player_sequence: list[int],
        reward_sequence: list[int] = None,
        render_mode=None,
    ):
        super().__init__(render_mode)
        self.__player_sequence = player_sequence
        self.__reward_sequence = (
            reward_sequence
            if reward_sequence is not None
            else [1 for _ in self.__player_sequence]
        )
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Discrete(10)
        self.current_sequence_idx = 0
        self._current_player_idx = self.__player_sequence[self.current_sequence_idx]

    def _apply_action(self, action):
        if self.current_sequence_idx == len(self.__player_sequence):
            return 0, True
        # do some action and get reward
        reward = self.__reward_sequence[self.current_sequence_idx]

        self.current_sequence_idx += 1
        if self.current_sequence_idx == len(self.__player_sequence):
            return reward, True
        self._current_player_idx = self.__player_sequence[self.current_sequence_idx]
        return reward, False

    def _get_action_mask(self, player_idx):
        return [1]

    def _render_text(self):
        return ""

    def _get_global_state(self):
        return {1: "0"}

    def _get_observation(self, player_idx):
        return ""

    def _reset_logic(self, seed=0, options=None):
        self.current_sequence_idx = 0
        self._current_player_idx = self.__player_sequence[self.current_sequence_idx]


class TestAutoOppoentWrapperContract:
    BOTS = {1: MockBot()}
    EGO_PLAYER_IDX = 0
    PLAYER_SEQUENCE = [0, 1]

    @pytest.fixture
    def wrapped_env(self):
        return AutoOpponentWrapper(
            env=ScriptedEnv(player_sequence=self.PLAYER_SEQUENCE),
            bots=self.BOTS,
            ego_player_idx=self.EGO_PLAYER_IDX,
        )

    def test_bots_property(self, wrapped_env):
        assert isinstance(wrapped_env.bots, dict)
        assert wrapped_env.bots is self.BOTS
        with pytest.raises(
            AttributeError,
            match="property 'bots' of '.*' object has no setter",
        ):
            wrapped_env.bots = {2: MockBot()}

    def test_ego_idx_property(self, wrapped_env):
        assert isinstance(wrapped_env.ego_player_idx, int)
        with pytest.raises(
            AttributeError,
            match="property 'ego_player_idx' of '.*' object has no setter",
        ):
            wrapped_env.ego_player_idx = 3


class TestAutoOpponentWrapperInUse:

    def test_regular_step(self):
        bot = MockBot()
        bots = {
            1: bot,
        }
        player_sequence = [1, 0, 0]
        reward_sequence = [1, 2, 3]
        env = ScriptedEnv(
            player_sequence=player_sequence, reward_sequence=reward_sequence
        )
        wrappered_env = AutoOpponentWrapper(env=env, bots=bots, ego_player_idx=0)
        wrappered_env.reset()

        _, reward, terminated, _, _ = wrappered_env.step(action=1)
        assert reward == reward_sequence[1]
        assert not terminated
        assert bot.call_count == 1

        _, reward, terminated, _, _ = wrappered_env.step(action=1)
        assert reward == reward_sequence[2]
        assert terminated
        assert bot.call_count == 1

    def test_regular_reset(self):
        bot = MockBot()
        bots = {
            1: bot,
        }
        env = ScriptedEnv(player_sequence=[1, 0, 0])
        wrappered_env = AutoOpponentWrapper(env=env, bots=bots, ego_player_idx=0)
        wrappered_env.reset()
        _, _, terminated, _, _ = wrappered_env.step(action=1)
        assert not terminated
        _, _, terminated, _, _ = wrappered_env.step(action=1)
        assert terminated
        wrappered_env.reset()
        _, _, terminated, _, _ = wrappered_env.step(action=1)
        assert not terminated

    def test_opponent_starts_game(self):
        bot = MockBot()
        bots = {
            1: bot,
        }
        env = ScriptedEnv(player_sequence=[1, 0])
        wrappered_env = AutoOpponentWrapper(env=env, bots=bots, ego_player_idx=0)
        wrappered_env.reset()
        _, _, terminated, _, _ = wrappered_env.step(action=1)
        assert terminated
        assert env.current_player_idx == 0
        assert bot.call_count == 1

    def test_opponent_starts_win(self):
        # no exception raised and no death loop should be here
        bot = MockBot()
        bots = {
            1: bot,
        }
        env = ScriptedEnv(player_sequence=[1])
        wrappered_env = AutoOpponentWrapper(env=env, bots=bots, ego_player_idx=0)
        wrappered_env.reset()
        _, _, terminated, _, _ = wrappered_env.step(action=1)
        assert terminated

    def test_ego_idx(self):
        bot = MockBot()
        bots = {
            0: bot,
        }
        player_sequence = [0, 0, 1]
        reward_sequence = [1, 2, 3]
        env = ScriptedEnv(
            player_sequence=player_sequence, reward_sequence=reward_sequence
        )
        wrappered_env = AutoOpponentWrapper(env=env, bots=bots, ego_player_idx=1)
        wrappered_env.reset()

        _, reward, terminated, _, _ = wrappered_env.step(action=1)
        assert reward == reward_sequence[-1]
        assert terminated
        assert bot.call_count == 2

    test_bots_params = [
        {"id": "empty bots", "bots": {}, "exception": ""},
        {
            "id": "non-consecutive bots idx",
            "bots": {1: MockBot(), 5: MockBot()},
        },
        {
            "id": "shuffled bots idx",
            "bots": {i: MockBot() for i in random.sample(list(range(5)), 5) if i != 0},
        },
        {
            "id": "multibots idx",
            "bots": {i: MockBot() for i in range(5000) if i != 0},
        },
    ]

    @pytest.mark.parametrize(
        "case", test_bots_params, ids=[case["id"] for case in test_bots_params]
    )
    def test_bots_success(self, case):
        player_sequence = (
            list(case["bots"].keys()) if len(case["bots"].keys()) > 0 else [0]
        )
        env = ScriptedEnv(player_sequence=player_sequence)
        AutoOpponentWrapper(env=env, bots=case["bots"], ego_player_idx=0)
        # no assertion, if no error should be fine

    def test_ego_idx_occupied(self):
        env = ScriptedEnv(player_sequence=[0, 1])
        with pytest.raises(
            ValueError, match="ego player idx is occupied by bots idx.*"
        ):
            AutoOpponentWrapper(env=env, bots={0: MockBot()}, ego_player_idx=0)

    test_players_params = [
        {
            "id": "ego turns",
            "player_sequence": [0, 0, 0, 0],
            "bots": {1: MockBot()},
            "reward_sequence": [1, 2, 3, 4],
        },
        {
            "id": "long ego turns",
            "player_sequence": [0 for _ in range(5000)],
            "bots": {1: MockBot()},
            "reward_sequence": [i**2 for i in range(5000)],
        },
        {
            "id": "2 players turns",
            "player_sequence": [0, 1],
            "bots": {1: MockBot()},
            "reward_sequence": [1, 2],
        },
        {
            "id": "long 2 players turns",
            "player_sequence": [i % 2 for i in range(5000)],
            "bots": {1: MockBot()},
            "reward_sequence": [i * 2 for i in range(5000)],
        },
        {
            "id": "long 2 players turns random",
            "player_sequence": [random.randint(0, 1) for i in range(5000)],
            "bots": {1: MockBot()},
            "reward_sequence": [i * 2 for i in range(5000)],
        },
        {
            "id": "3 players",
            "player_sequence": [i % 3 for i in range(5000)],
            "bots": {1: MockBot(), 2: MockBot()},
            "reward_sequence": [i * 2 for i in range(5000)],
        },
        {
            "id": "long 3 players",
            "player_sequence": [i % 3 for i in range(5000)],
            "bots": {1: MockBot(), 2: MockBot()},
            "reward_sequence": [i * 2 for i in range(5000)],
        },
        {
            "id": "long 3 players random",
            "player_sequence": [random.randint(0, 2) for i in range(5000)],
            "bots": {1: MockBot(), 2: MockBot()},
            "reward_sequence": [i * 2 for i in range(5000)],
        },
        {
            "id": "4 players",
            "player_sequence": [i % 4 for i in range(5000)],
            "bots": {1: MockBot(), 2: MockBot(), 3: MockBot()},
            "reward_sequence": [i * 2 for i in range(5000)],
        },
        {
            "id": "long 4 players",
            "player_sequence": [i % 4 for i in range(5000)],
            "bots": {1: MockBot(), 2: MockBot(), 3: MockBot()},
            "reward_sequence": [i * 2 for i in range(5000)],
        },
        {
            "id": "long 4 players random",
            "player_sequence": [random.randint(0, 3) for i in range(5000)],
            "bots": {1: MockBot(), 2: MockBot(), 3: MockBot()},
            "reward_sequence": [i * 2 for i in range(5000)],
        },
    ]

    @pytest.mark.parametrize(
        "case", test_players_params, ids=[param["id"] for param in test_players_params]
    )
    def test_players_play(self, case):
        expected_ego_movement_counts = case["player_sequence"].count(0)
        expected_ego_rewards = [
            reward
            for player_idx, reward in zip(
                case["player_sequence"], case["reward_sequence"]
            )
            if player_idx == 0
        ]

        expected_bot_movement_counts_dict = {}
        for bot_idx in case["bots"]:
            expected_bot_movement_counts_dict[bot_idx] = case["player_sequence"].count(
                bot_idx
            )
        env = ScriptedEnv(
            player_sequence=case["player_sequence"],
            reward_sequence=case["reward_sequence"],
        )
        wrappered_env = AutoOpponentWrapper(env=env, bots=case["bots"])
        wrappered_env.reset()

        # move
        for i in range(expected_ego_movement_counts):
            assert (
                env.current_player_idx == 0
            )  # for wrappered environment, will only be players view, all other bot's turns will be collapse
            _, reward, terminated, _, _ = wrappered_env.step(action=0)

            if i == expected_ego_movement_counts - 1:
                assert terminated
            else:
                assert not terminated

            assert reward == expected_ego_rewards[i]

        for bot_idx, bot in case["bots"].items():
            assert bot.call_count == expected_bot_movement_counts_dict[bot_idx]

    def test_ego_turn_only(self):
        bot = MockBot()
        bots = {
            1: bot,
        }
        player_sequence = [0, 0, 0, 0]
        reward_sequence = [0, 0, 0, 0]
        env = ScriptedEnv(
            player_sequence=player_sequence, reward_sequence=reward_sequence
        )
        wrappered_env = AutoOpponentWrapper(env=env, bots=bots)
        wrappered_env.reset()

        for i, player_idx in enumerate(player_sequence):
            _, _, terminated, _, _ = wrappered_env.step(action=0)

            if i == len(player_sequence) - 1:
                assert terminated
            else:
                assert not terminated
            assert env.current_player_idx == player_idx
            assert bot.call_count == 0
