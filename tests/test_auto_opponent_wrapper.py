import pytest
import gymnasium as gym
import numpy as np

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
        self.__reward_sequnce = (
            reward_sequence
            if reward_sequence is not None
            else [1 for _ in self.__player_sequence]
        )
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Discrete(10)
        self.current_sequence_id = 0
        self.current_player_idx = self.__player_sequence[self.current_sequence_id]

    def _apply_action(self, action):
        reward = self.__reward_sequnce[self.current_sequence_id]
        if self.current_sequence_id == len(self.__player_sequence) - 1:
            return reward, True

        self.current_sequence_id += 1
        self.current_player_idx = self.__player_sequence[self.current_sequence_id]
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
        self.current_sequence_id = 0
        self.current_player_idx = self.__player_sequence[self.current_sequence_id]


class TestAutoOpponentWrapper:

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

    @pytest.mark.skip("TODO")
    def test_ego_idx(self): ...  # TODO

    @pytest.mark.skip("TODO")
    def test_bots(self):  # TODO
        # empty bots
        # wrong idx bots
        # wrong bots type
        ...

    test_players_params = [
        {
            "id": "ego turn only case",
            "player_sequence": [0, 0, 0, 0],
            "bots": {1: MockBot()},
            "reward_sequence": [1, 2, 3, 4],
        },
        {
            "id": "long ego turn only case",
            "player_sequence": [0 for _ in range(5000)],
            "bots": {1: MockBot()},
            "reward_sequence": [i**2 for i in range(5000)],
        },
        # {
        #     "id": "",
        #     "player_sequence": [],
        #     "bots": {1: MockBot()},
        #     "reward_sequence": [],
        # },
    ]

    @pytest.mark.parametrize(
        "case", test_players_params, ids=[param["id"] for param in test_players_params]
    )
    def test_players_play(self, case):
        expected_ego_movement_counts = len(case["player_sequence"])
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
            _, reward, terminated, _, _ = wrappered_env.step(action=0)

            if i == expected_ego_movement_counts - 1:
                assert terminated
            else:
                assert not terminated
            assert (
                env.current_player_idx == 0
            )  # for wrappered environment, will only be players view, all other bot's turns will be collapse

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
