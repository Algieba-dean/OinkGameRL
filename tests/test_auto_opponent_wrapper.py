import pytest
import gymnasium as gym
import random

from games.AutoOpponentWrapper import AutoOpponentWrapper
from games.OinkGame import OinkGameEnv
from games.GameAgent import GameAgent


@pytest.fixture
def bot_factory(mocker):
    def _create(return_value=1):
        bot = mocker.Mock(spec=GameAgent)
        bot.predict.return_value = return_value
        return bot

    return _create


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

        if self.__player_sequence:
            self._current_player_idx = self.__player_sequence[self.current_sequence_idx]
        else:
            self._current_player_idx = 0

    def _apply_action(self, action):
        if self.current_sequence_idx >= len(self.__player_sequence):
            return 0, True
        # do some action and get reward
        reward = self.__reward_sequence[self.current_sequence_idx]
        self.current_sequence_idx += 1

        # game terminate condition
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
        if self.__player_sequence:
            self._current_player_idx = self.__player_sequence[self.current_sequence_idx]


class TestAutoOppoentWrapperContract:
    EGO_PLAYER_IDX = 0
    PLAYER_SEQUENCE = [0, 1]

    def test_bots_property(self, bot_factory):
        bots = {1: bot_factory()}
        wrapped_env = AutoOpponentWrapper(
            env=ScriptedEnv(player_sequence=self.PLAYER_SEQUENCE),
            bots=bots,
            ego_player_idx=self.EGO_PLAYER_IDX,
        )
        assert isinstance(wrapped_env.bots, dict)
        assert wrapped_env.bots is bots
        with pytest.raises(
            AttributeError,
            match="property 'bots' of '.*' object has no setter",
        ):
            wrapped_env.bots = {2: bot_factory()}

    def test_ego_idx_property(self):
        bots = {}
        wrapped_env = AutoOpponentWrapper(
            env=ScriptedEnv(player_sequence=self.PLAYER_SEQUENCE),
            bots=bots,
            ego_player_idx=self.EGO_PLAYER_IDX,
        )
        assert isinstance(wrapped_env.ego_player_idx, int)
        with pytest.raises(
            AttributeError,
            match="property 'ego_player_idx' of '.*' object has no setter",
        ):
            wrapped_env.ego_player_idx = 3


class TestAutoOpponentWrapperInUse:

    def test_regular_step(self, bot_factory):
        bot = bot_factory()
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
        assert bot.predict.call_count == 1

        _, reward, terminated, _, _ = wrappered_env.step(action=1)
        assert reward == reward_sequence[2]
        assert terminated
        assert bot.predict.call_count == 1

    def test_regular_reset(self, bot_factory):
        bot = bot_factory()
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

    def test_opponent_starts_game(self, bot_factory):
        bot = bot_factory()
        bots = {
            1: bot,
        }
        env = ScriptedEnv(player_sequence=[1, 0])
        wrappered_env = AutoOpponentWrapper(env=env, bots=bots, ego_player_idx=0)
        wrappered_env.reset()
        _, _, terminated, _, _ = wrappered_env.step(action=1)
        assert terminated
        assert env.current_player_idx == 0
        assert bot.predict.call_count == 1

    def test_opponent_starts_win(self, bot_factory):
        # no exception raised and no death loop should be here
        bot = bot_factory()
        bots = {
            1: bot,
        }
        env = ScriptedEnv(player_sequence=[1])
        wrappered_env = AutoOpponentWrapper(env=env, bots=bots, ego_player_idx=0)
        wrappered_env.reset()
        _, _, terminated, _, _ = wrappered_env.step(action=1)
        assert terminated

    def test_ego_idx(self, bot_factory):
        bot = bot_factory()
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
        assert bot.predict.call_count == 2

    test_bots_params = [
        {"id": "empty bots", "bots_idx": [], "exception": ""},
        {
            "id": "non-consecutive bots idx",
            "bots_idx": [1, 5],
        },
        {
            "id": "shuffled bots idx",
            "bots_idx": [i for i in random.sample(list(range(5)), 5) if i != 0],
        },
        {
            "id": "multibots idx",
            "bots_idx": list(range(1, 5000)),
        },
    ]

    @pytest.mark.parametrize("case", test_bots_params, ids=lambda case: case["id"])
    def test_bots_success(self, case, bot_factory):
        bots = {idx: bot_factory() for idx in case["bots_idx"]}
        player_sequence = list(bots.keys()) if len(bots.keys()) > 0 else [0]

        env = ScriptedEnv(player_sequence=player_sequence)
        AutoOpponentWrapper(env=env, bots=bots, ego_player_idx=0)
        # no assertion, if no error should be fine

    def test_ego_idx_occupied(self, bot_factory):
        env = ScriptedEnv(player_sequence=[0, 1])
        with pytest.raises(
            ValueError, match="ego player idx is occupied by bots idx.*"
        ):
            AutoOpponentWrapper(env=env, bots={0: bot_factory()}, ego_player_idx=0)

    test_players_params = [
        {
            "id": "ego turns",
            "player_sequence": [0, 0, 0, 0],
            "bots_idx": [1],
            "reward_sequence": [1, 2, 3, 4],
        },
        {
            "id": "long ego turns",
            "player_sequence": [0 for _ in range(5000)],
            "bots_idx": [1],
            "reward_sequence": [i**2 for i in range(5000)],
        },
        {
            "id": "2 players turns",
            "player_sequence": [0, 1],
            "bots_idx": [1],
            "reward_sequence": [1, 2],
        },
        {
            "id": "long 2 players turns",
            "player_sequence": [i % 2 for i in range(5000)],
            "bots_idx": [1],
            "reward_sequence": [i * 2 for i in range(5000)],
        },
        {
            "id": "long 2 players turns random",
            "player_sequence": [random.randint(0, 1) for i in range(5000)],
            "bots_idx": [1],
            "reward_sequence": [i * 2 for i in range(5000)],
        },
        {
            "id": "3 players",
            "player_sequence": [i % 3 for i in range(5000)],
            "bots_idx": [1, 2],
            "reward_sequence": [i * 2 for i in range(5000)],
        },
        {
            "id": "long 3 players",
            "player_sequence": [i % 3 for i in range(5000)],
            "bots_idx": [1, 2],
            "reward_sequence": [i * 2 for i in range(5000)],
        },
        {
            "id": "long 3 players random",
            "player_sequence": [random.randint(0, 2) for i in range(5000)],
            "bots_idx": [1, 2],
            "reward_sequence": [i * 2 for i in range(5000)],
        },
        {
            "id": "4 players",
            "player_sequence": [i % 4 for i in range(5000)],
            "bots_idx": [1, 2, 3],
            "reward_sequence": [i * 2 for i in range(5000)],
        },
        {
            "id": "long 4 players",
            "player_sequence": [i % 4 for i in range(5000)],
            "bots_idx": [1, 2, 3],
            "reward_sequence": [i * 2 for i in range(5000)],
        },
        {
            "id": "long 4 players random",
            "player_sequence": [random.randint(0, 3) for i in range(5000)],
            "bots_idx": [1, 2, 3],
            "reward_sequence": [i * 2 for i in range(5000)],
        },
    ]

    @pytest.mark.parametrize("case", test_players_params, ids=lambda case: case["id"])
    def test_players_play(self, case, bot_factory):
        bots = {idx: bot_factory() for idx in case["bots_idx"]}
        expected_ego_movement_counts = case["player_sequence"].count(0)
        expected_ego_rewards = [
            reward
            for player_idx, reward in zip(
                case["player_sequence"], case["reward_sequence"]
            )
            if player_idx == 0
        ]

        expected_bot_movement_counts_dict = {}
        for bot_idx in bots:
            expected_bot_movement_counts_dict[bot_idx] = case["player_sequence"].count(
                bot_idx
            )
        env = ScriptedEnv(
            player_sequence=case["player_sequence"],
            reward_sequence=case["reward_sequence"],
        )
        wrappered_env = AutoOpponentWrapper(env=env, bots=bots)
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

        for bot_idx, bot in bots.items():
            assert bot.predict.call_count == expected_bot_movement_counts_dict[bot_idx]

    def test_ego_turn_only(self, bot_factory):
        bot = bot_factory()
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
            assert bot.predict.call_count == 0
