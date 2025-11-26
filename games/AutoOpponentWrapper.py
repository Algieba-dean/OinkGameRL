from __future__ import annotations
from typing import Dict, Tuple, Any, TYPE_CHECKING, List
import gymnasium as gym


if TYPE_CHECKING:  # pragma: no cover
    from GameAgent import GameAgent
    from OinkGame import OinkGameEnv


class AutoOpponentWrapper(gym.Wrapper):
    def __init__(
        self, env: OinkGameEnv, bots: Dict[int, GameAgent], ego_player_idx: int = 0
    ):
        super().__init__(env)
        self.__bots: Dict[int, GameAgent] = bots
        self.__ego_player_idx: int = ego_player_idx
        if self.__ego_player_idx in self.__bots:
            raise ValueError(
                f"ego player idx is occupied by bots idx, ego:{self.__ego_player_idx}, bot:{self.__bots.keys()}"
            )

    @property
    def ego_player_idx(self) -> int:
        return self.__ego_player_idx

    @property
    def bots(self) -> Dict[int, GameAgent]:
        return self.__bots

    def reset(self, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        terminated = False
        truncated = False

        # check once after reset ego player is not the current
        while self.__need_process_opponent_turns(
            terminated=terminated,
            truncted=truncated,
            current_player_idx=self.env.current_player_idx,
            ego_player_idx=self.ego_player_idx,
        ):
            # let it move til current player is ego player, no need to know bot's score
            observation, _, terminated, truncated, info = self.__process_opponent_turns(
                previous_observation=observation,
                terminated=terminated,
                truncated=terminated,
                previous_info=info,
            )
        return observation, info

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:

        # the step is now only for ego player movement
        observation, reward, terminated, truncated, info = self.env.step(action=action)

        while self.__need_process_opponent_turns(
            terminated=terminated,
            truncted=truncated,
            current_player_idx=self.env.current_player_idx,
            ego_player_idx=self.ego_player_idx,
        ):
            # let bots run
            observation, _, terminated, truncated, info = self.__process_opponent_turns(
                previous_observation=observation,
                terminated=terminated,
                truncated=truncated,
                previous_info=info,
            )

        return observation, reward, terminated, truncated, info

    def __process_opponent_turns(
        self,
        previous_observation: Any,
        terminated: bool,
        truncated: bool,
        previous_info: Dict[str, Any],
    ) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        observation = previous_observation
        info = previous_info
        while self.__need_process_opponent_turns(
            terminated=terminated,
            truncted=truncated,
            current_player_idx=self.env.current_player_idx,
            ego_player_idx=self.ego_player_idx,
        ):
            current_player_idx = self.env.current_player_idx
            bot = self.bots[current_player_idx]
            bot_observation = self.env.unwrapped._get_observation(current_player_idx)
            bot_action_mask = self.env.unwrapped._get_action_mask(current_player_idx)
            bot_action = bot.predict(
                observation=bot_observation, action_mask=bot_action_mask
            )
            observation, reward, terminated, truncated, info = self.env.step(
                action=bot_action
            )
        return observation, reward, terminated, terminated, info

    @staticmethod
    def __need_process_opponent_turns(
        terminated: bool, truncted: bool, current_player_idx: int, ego_player_idx: int
    ) -> bool:
        return not terminated and not truncted and current_player_idx != ego_player_idx
