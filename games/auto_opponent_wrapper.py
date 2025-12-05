from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import gymnasium as gym

if TYPE_CHECKING:  # pragma: no cover
    from games.game_agent import GameAgent
    from games.oink_game import OinkGameEnv


class AutoOpponentWrapper(gym.Wrapper):
    env: OinkGameEnv  # just use to pass MyPy check

    def __init__(
        self, env: OinkGameEnv, bots: dict[int, GameAgent], ego_player_idx: int = 0
    ):
        super().__init__(env)
        self.__bots: dict[int, GameAgent] = bots
        self.__ego_player_idx: int = ego_player_idx
        if self.__ego_player_idx in self.__bots:
            raise ValueError(
                f"ego player idx is occupied by bots idx, ego:{self.__ego_player_idx}, bot:{self.__bots.keys()}"
            )

    @property
    def _unwrapped_env(self) -> OinkGameEnv:
        """
        a specific attribution to as for AutoOpponentWrapper, the env will be OinkGameEnv
        """
        from games.oink_game import (
            OinkGameEnv,
        )  # we have to do such a lazy import to pass ruff check

        return cast(OinkGameEnv, self.env.unwrapped)

    @property
    def ego_player_idx(self) -> int:
        return self.__ego_player_idx

    @property
    def bots(self) -> dict[int, GameAgent]:
        return self.__bots

    def reset(self, **kwargs) -> tuple[Any, dict[str, Any]]:
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

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
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

        return observation, float(reward), terminated, truncated, info

    def __process_opponent_turns(
        self,
        previous_observation: Any,
        terminated: bool,
        truncated: bool,
        previous_info: dict[str, Any],
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
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
            bot_observation = self._unwrapped_env._get_observation(current_player_idx)
            bot_action_mask = self._unwrapped_env._get_action_mask(current_player_idx)
            bot_action = bot.predict(
                observation=bot_observation, action_mask=bot_action_mask
            )
            observation, reward, terminated, truncated, info = self.env.step(
                action=bot_action
            )
        return observation, float(reward), terminated, terminated, info

    @staticmethod
    def __need_process_opponent_turns(
        terminated: bool, truncted: bool, current_player_idx: int, ego_player_idx: int
    ) -> bool:
        return not terminated and not truncted and current_player_idx != ego_player_idx
